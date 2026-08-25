"""Residual block bootstrap -- the primary dependence model.

The idea in one sentence: resample whole historical days, keeping all apps and
all 36 intervals together, so the intra-day shape and the cross-app coincidence
come along for free.

Why this rather than a copula. A Gaussian copula on PIT-transformed residuals
is more principled and it is implemented alongside this (see copula.py), but it
assumes the dependence is Gaussian -- which for a 35-dimensional joint whose
whole purpose is the *tail* is a substantial assumption, and one that is hard
to defend in a review where the audience is a capacity manager and a systems
programmer rather than a statistician. The block bootstrap assumes nothing
about the distribution. It says: "days like this have happened; here is what
they looked like." That sentence survives an interview.

What it costs: the bootstrap can only produce coincidence patterns that have
actually occurred. With ~750 usable days that is a reasonable pool for a
two-year horizon, but it cannot invent a pattern the history has never shown,
and the number of distinct days available is a hard limit on the diversity of
the tail. That limitation is real, it is reported in the manifest, and it is
the reason the copula comparison exists at all.

About 80 lines of numpy, as promised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from capplan.logging_utils import get_logger
from capplan.sim.residuals import ResidualPanel

LOG = get_logger(__name__)


@dataclass
class DayStream:
    """One chunk's residual draws, resolved one day at a time.

    `indices[p, d]` is the historical day feeding path p's day d -- which also
    makes provenance free: the composition of any simulated path can be traced
    back to real dates.
    """

    pool: np.ndarray                 # (pool_days, apps, intervals)
    indices: np.ndarray              # (size, n_days) int32

    @property
    def n_days(self) -> int:
        return self.indices.shape[1]

    def day(self, d: int) -> np.ndarray:
        """(size, apps, intervals) residuals for horizon day `d`."""
        return self.pool[self.indices[:, d]]

    def nbytes(self) -> int:
        return int(self.indices.nbytes)


@dataclass
class BlockBootstrap:
    """Sampler over aligned residual day-blocks.

    `pool` has shape (n_usable_days, n_apps, n_intervals). One draw returns one
    whole historical day across every app -- never a per-app or per-interval
    recombination, which would break exactly the structure this exists to
    preserve.
    """

    pool: np.ndarray
    block_days: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.pool.ndim != 3:
            raise ValueError("residual pool must be (days, apps, intervals)")
        if len(self.pool) == 0:
            raise ValueError("residual pool is empty; no clean aligned days to resample")
        if self.block_days < 1:
            raise ValueError("block_days must be at least 1")
        self._rng = np.random.default_rng(self.seed)

    @classmethod
    def from_panel(cls, panel: ResidualPanel, block_days: int = 1, seed: int = 0):
        return cls(pool=panel.usable_block(), block_days=block_days, seed=seed)

    @property
    def n_days(self) -> int:
        return len(self.pool)

    @property
    def n_apps(self) -> int:
        return self.pool.shape[1]

    @property
    def n_intervals(self) -> int:
        return self.pool.shape[2]

    def n_starts(self) -> int:
        """Distinct block start positions. The real diversity of the sampler."""
        return max(1, self.n_days - self.block_days + 1)

    def draw(self, n_days: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw residuals for `n_days` consecutive future business days.

        Returns (n_days, n_apps, n_intervals).

        With `block_days > 1` consecutive historical days are kept together, so
        day-to-day persistence survives as well -- a week that ran hot stays a
        week that ran hot. Blocks are drawn with replacement and the tail is
        truncated to the requested length.
        """
        rng = rng or self._rng
        if self.block_days == 1:
            picks = rng.integers(0, self.n_days, size=n_days)
            return self.pool[picks]

        n_blocks = int(np.ceil(n_days / self.block_days))
        starts = rng.integers(0, self.n_starts(), size=n_blocks)
        idx = (starts[:, None] + np.arange(self.block_days)[None, :]).reshape(-1)
        idx = np.minimum(idx, self.n_days - 1)[:n_days]
        return self.pool[idx]

    def draw_indices(self, n_days: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Which historical day fed each future day. Kept for provenance.

        Being able to answer "the 99th-percentile path is made of 14 March, 2
        August and 19 November" is worth the array it takes to store.
        """
        rng = rng or self._rng
        if self.block_days == 1:
            return rng.integers(0, self.n_days, size=n_days)
        n_blocks = int(np.ceil(n_days / self.block_days))
        starts = rng.integers(0, self.n_starts(), size=n_blocks)
        idx = (starts[:, None] + np.arange(self.block_days)[None, :]).reshape(-1)
        return np.minimum(idx, self.n_days - 1)[:n_days]

    def stream(self, size: int, n_days: int, rng: np.random.Generator) -> "DayStream":
        """Lazy per-day view for `size` paths over `n_days`.

        Materialising the residual draws for a whole chunk would be
        size x days x apps x intervals -- 2.6 GB at the default settings, which
        defeats the entire point of chunking. Only the day-indices are stored
        (a few MB); each day's residual slab is produced on demand and thrown
        away, so the resident cost is one (size, apps, intervals) array.
        """
        idx = np.stack([self.draw_indices(n_days, rng) for _ in range(size)])
        return DayStream(pool=self.pool, indices=idx.astype(np.int32))

    def diagnostics(self) -> dict[str, float]:
        return {
            "pool_days": self.n_days,
            "block_days": self.block_days,
            "distinct_block_starts": self.n_starts(),
            "n_apps": self.n_apps,
            "n_intervals": self.n_intervals,
            # The honest limitation: no coincidence pattern outside this pool
            # can ever be produced.
            "max_distinct_day_patterns": self.n_days,
        }


def apply_residuals(
    median: np.ndarray,
    residuals: np.ndarray,
    spread: np.ndarray | None,
    scaling: str,
    floor: float = 0.0,
) -> np.ndarray:
    """Turn residual draws back into MIPS.

    `median`     (apps, intervals) Stage 1 median for the target day
    `residuals`  (apps, intervals) a drawn residual block
    `spread`     (apps, intervals) half the (q10, q90) width for that day
    """
    if scaling == "spread":
        if spread is None:
            raise ValueError("spread scaling requires the forecast spread")
        out = median + residuals * spread
    else:
        out = median + residuals
    return np.maximum(out, floor)
