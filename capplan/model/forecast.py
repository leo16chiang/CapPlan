"""The Stage 1 output object.

A `ForecastCube` is the entire contract between Stage 1 and Stage 2: marginal
predictive quantiles for every (app, future interval). It is deliberately not a
point forecast with error bars bolted on, and it deliberately contains no peak.

Layout: `q[app, day, interval, quantile]`. For 35 apps x 512 business days x 36
intervals x 9 quantiles that is ~46MB in float32 -- small enough to hold, large
enough that float64 would be wasteful.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class ForecastCube:
    """Marginal predictive quantiles per (app, day, interval)."""

    q: np.ndarray               # (apps, days, intervals, quantiles), float32
    quantiles: np.ndarray       # (quantiles,) ascending, e.g. 0.01 .. 0.99
    apps: list[str]
    days: list[date]
    n_intervals: int
    backend: str = "unknown"
    calibrated: bool = False

    def __post_init__(self) -> None:
        self.q = np.asarray(self.q, dtype=np.float32)
        self.quantiles = np.asarray(self.quantiles, dtype=np.float64)
        expected = (len(self.apps), len(self.days), self.n_intervals, len(self.quantiles))
        if self.q.shape != expected:
            raise ValueError(f"forecast cube shape {self.q.shape} != expected {expected}")

    # -- accessors -------------------------------------------------------

    @property
    def median_index(self) -> int:
        return int(np.argmin(np.abs(self.quantiles - 0.5)))

    @property
    def median(self) -> np.ndarray:
        """(apps, days, intervals) central forecast. Not a peak forecast."""
        return self.q[..., self.median_index]

    def spread(self, lo: float = 0.1, hi: float = 0.9) -> np.ndarray:
        """Interquantile width, used to scale bootstrap residuals."""
        i_lo = int(np.argmin(np.abs(self.quantiles - lo)))
        i_hi = int(np.argmin(np.abs(self.quantiles - hi)))
        width = self.q[..., i_hi] - self.q[..., i_lo]
        # A degenerate interval would make scaled residuals blow up or vanish;
        # floor it at something below measurement resolution.
        return np.maximum(width, 1e-6)

    def app_pos(self, app: str) -> int:
        return self.apps.index(app)

    def enforce_monotone(self) -> "ForecastCube":
        """Repair quantile crossing in place.

        Independently fitted quantiles cross, especially in the tails and
        especially after conformal widening. Sorting along the quantile axis is
        the standard fix and cannot make calibration worse.
        """
        self.q = np.sort(self.q, axis=-1)
        return self

    def clip_nonnegative(self) -> "ForecastCube":
        """MIPS cannot be negative; a lower tail that goes there is an artefact."""
        np.clip(self.q, 0.0, None, out=self.q)
        return self

    # -- persistence -----------------------------------------------------

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            q=self.q,
            quantiles=self.quantiles,
            apps=np.array(self.apps, dtype=object),
            days=np.array([d.isoformat() for d in self.days], dtype=object),
            n_intervals=self.n_intervals,
            backend=self.backend,
            calibrated=self.calibrated,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "ForecastCube":
        with np.load(path, allow_pickle=True) as data:
            return cls(
                q=data["q"],
                quantiles=data["quantiles"],
                apps=[str(a) for a in data["apps"]],
                days=[date.fromisoformat(str(d)) for d in data["days"]],
                n_intervals=int(data["n_intervals"]),
                backend=str(data["backend"]),
                calibrated=bool(data["calibrated"]),
            )

    # -- reporting -------------------------------------------------------

    def to_long(self, quantiles: Sequence[float] | None = None) -> pd.DataFrame:
        """Long form for the Dash app and the custodian pack.

        Only the requested quantiles, because the full cube in long form is
        ~5.8M rows and nothing on the serving side wants that.
        """
        wanted = list(quantiles) if quantiles is not None else list(self.quantiles)
        cols = [int(np.argmin(np.abs(self.quantiles - q))) for q in wanted]
        n_a, n_d, n_i = len(self.apps), len(self.days), self.n_intervals
        frames = []
        for q, col in zip(wanted, cols):
            frames.append(
                pd.DataFrame(
                    {
                        "app_id": np.repeat(np.array(self.apps, dtype=object), n_d * n_i),
                        "business_date": np.tile(
                            np.repeat(np.array(self.days, dtype=object), n_i), n_a
                        ),
                        "interval_idx": np.tile(np.arange(n_i, dtype="int16"), n_a * n_d),
                        "quantile": q,
                        "mips": self.q[..., col].reshape(-1),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    def describe(self) -> str:
        return (
            f"ForecastCube[{self.backend}{'/calibrated' if self.calibrated else ''}] "
            f"{len(self.apps)} apps x {len(self.days)} days x {self.n_intervals} intervals "
            f"x {len(self.quantiles)} quantiles, {self.days[0]} .. {self.days[-1]}, "
            f"{self.q.nbytes / 1e6:.0f}MB"
        )
