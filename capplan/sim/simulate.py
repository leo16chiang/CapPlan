"""Stage 3: sample paths, sum across apps, then reduce.

The memory argument, stated once because it drives the whole shape of this
module. The naive implementation materialises

    paths x days x intervals x apps
    10,000 x 512 x 36 x 35 x 4 bytes = 736 GB

which is not a tuning problem. So nothing is ever materialised at that size:

  * paths are processed in chunks of `path_chunk` (default 500),
  * the sum across apps happens *inside* the day loop, so the per-app axis
    exists only for one (chunk, day) slice at a time -- 500 x 35 x 36 floats,
    about 5 MB,
  * a running maximum per (path, app) carries the app-level numbers without
    keeping any app-level history,
  * only the reduced daily series survives the loop: two (paths, days) float32
    arrays, about 40 MB at the default settings.

Peak footprint stays under a gigabyte, and the dominant term is the two
accumulators rather than anything that scales with apps or intervals.

The order of operations is the substance of the architecture:

    draw residuals -> add to marginals -> SUM ACROSS APPS -> take the maximum

Taking the maximum before the sum would give the sum of app peaks, which is the
32% overstatement the diagnostics measured. It is one transposition away and it
is the entire point, which is why the sum is a named step rather than an
argument to a reduction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from capplan.config import Config
from capplan.data.calendar import PrimeTimeGrid
from capplan.logging_utils import get_logger
from capplan.model.forecast import ForecastCube
from capplan.sim.reducers import (
    PathSummary,
    _Reducer,
    get_reducer,
    needs_r4ha,
    reduce_by_fiscal_year,
)

LOG = get_logger(__name__)


@dataclass
class SimulationResult:
    """Everything Stage 3 produces. No raw paths -- by design."""

    daily_peaks: np.ndarray          # (paths, days) float32, peak of the SUM
    daily_means: np.ndarray          # (paths, days) float32
    daily_sum_app_peaks: np.ndarray  # (paths, days) float32, SUM of app peaks
    app_peaks: np.ndarray            # (paths, apps) float32, running max per app
    days: list[date]
    apps: list[str]
    fiscal_years: np.ndarray         # (days,)
    dependence: str
    n_paths: int
    daily_r4ha: np.ndarray | None = None          # (paths, days) peak R4HA of the SUM
    daily_sum_app_r4ha: np.ndarray | None = None  # (paths, days) SUM of per-app peak R4HA
    r4ha_window_intervals: int = 0
    reducer_values: dict[str, np.ndarray] = field(default_factory=dict)
    reducer_by_fy: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)
    diagnostics: dict[str, float] = field(default_factory=dict)

    def quantiles(self, name: str, levels: Sequence[float] = (0.5, 0.9, 0.95, 0.99)) -> dict:
        values = self.reducer_values[name]
        return {f"p{int(100 * q)}": float(np.quantile(values, q)) for q in levels}

    def fy_table(self, name: str, levels: Sequence[float] = (0.5, 0.9, 0.95, 0.99)):
        """Per fiscal year distribution of the reduced figure. The deliverable."""
        import pandas as pd

        rows = []
        for fy, values in sorted(self.reducer_by_fy[name].items()):
            row = {"fiscal_year": fy, "reducer": name, "mean": float(values.mean())}
            for q in levels:
                row[f"p{int(100 * q)}"] = float(np.quantile(values, q))
            rows.append(row)
        return pd.DataFrame(rows)

    def coincidence_by_target(self) -> dict[str, float]:
        """Coincidence for the interval peak vs for the R4HA.

        The gap is the point. Averaging over four hours smooths out the timing
        differences that make peaks fail to sum, so the R4HA coincidence factor
        sits much closer to 1. Concretely: the Stage 2 dependence machinery is
        worth a lot on a hardware-sizing deliverable and much less on an MLC
        cost one, and this number says how much.
        """
        out = {}
        interval_ratio = self.daily_peaks / np.maximum(self.daily_sum_app_peaks, 1e-9)
        out["coincidence_interval_peak"] = float(interval_ratio.mean())
        if self.daily_r4ha is not None and self.daily_sum_app_r4ha is not None:
            # Like for like: the R4HA of the sum over the sum of each
            # application's OWN R4HA. Dividing by the sum of interval peaks
            # instead would fold the smoothing effect into the coincidence
            # figure and read far worse than reality.
            r4ha_ratio = self.daily_r4ha / np.maximum(self.daily_sum_app_r4ha, 1e-9)
            out["coincidence_r4ha"] = float(r4ha_ratio.mean())
            out["r4ha_to_interval_peak_ratio"] = float(
                (self.daily_r4ha / np.maximum(self.daily_peaks, 1e-9)).mean()
            )
            out["coincidence_gain_from_r4ha"] = (
                out["coincidence_r4ha"] - out["coincidence_interval_peak"]
            )
        return out

    def coincidence(self) -> dict[str, float]:
        """Simulated coincidence factor, comparable with the historical SQL.

        Two different quantities, and confusing them wastes an afternoon:

        `daily`     peak of the daily sum / sum of that day's app peaks,
                    averaged over days and paths. This is the same quantity
                    diagnostics/sql/coincidence.sql computes on history, and it
                    is the one to compare. If the simulation does not reproduce
                    the historical factor, the dependence model is wrong and the
                    forward simulation is wrong in the same direction -- exactly
                    the failure that stays hidden until a hardware config has
                    been signed.

        `horizon`   peak of the sum over the *whole* horizon / sum of each app's
                    own whole-horizon maximum. Always the lower number, because
                    each app's 512-day maximum is itself an extreme that rarely
                    coincides with any other app's. Useful, but not the number
                    the historical diagnostic reports.
        """
        daily_ratio = self.daily_peaks / np.maximum(self.daily_sum_app_peaks, 1e-9)
        horizon_ratio = self.daily_peaks.max(axis=1) / np.maximum(
            self.app_peaks.sum(axis=1), 1e-9
        )
        return {
            "simulated_coincidence_daily_mean": float(daily_ratio.mean()),
            "simulated_coincidence_daily_p05": float(np.quantile(daily_ratio, 0.05)),
            "simulated_coincidence_daily_p95": float(np.quantile(daily_ratio, 0.95)),
            "simulated_coincidence_horizon_mean": float(horizon_ratio.mean()),
            "mean_peak_of_sum": float(self.daily_peaks.max(axis=1).mean()),
            "mean_sum_of_app_peaks": float(self.app_peaks.sum(axis=1).mean()),
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            daily_peaks=self.daily_peaks,
            daily_means=self.daily_means,
            daily_sum_app_peaks=self.daily_sum_app_peaks,
            app_peaks=self.app_peaks,
            daily_r4ha=(
                self.daily_r4ha if self.daily_r4ha is not None else np.zeros(0, dtype=np.float32)
            ),
            daily_sum_app_r4ha=(
                self.daily_sum_app_r4ha
                if self.daily_sum_app_r4ha is not None
                else np.zeros(0, dtype=np.float32)
            ),
            days=np.array([d.isoformat() for d in self.days], dtype=object),
            apps=np.array(self.apps, dtype=object),
            fiscal_years=self.fiscal_years,
            dependence=self.dependence,
            **{f"reducer__{k}": v for k, v in self.reducer_values.items()},
        )
        return path


def simulate(
    forecast: ForecastCube,
    sampler,
    grid: PrimeTimeGrid,
    n_paths: int = 10_000,
    path_chunk: int = 500,
    reducers: Sequence[str] = ("annual_max",),
    seed: int = 20260101,
    residual_scaling: str = "spread",
    dependence: str = "block_bootstrap",
    r4ha_hours: float = 4.0,
    progress_every: int = 5,
) -> SimulationResult:
    """Sample paths from the joint model and reduce them.

    `sampler` needs one method: `stream(size, n_days, rng)`, returning an object
    with `day(d) -> (size, apps, intervals)` of residuals on the residual scale.
    Both `BlockBootstrap` and `GaussianCopula` satisfy it, which is what makes
    the dependence model swappable without touching this loop. The stream is
    lazy on purpose -- resolving a whole chunk's residuals eagerly costs 2.6 GB
    and defeats the chunking.
    """
    n_apps, n_days, n_int = forecast.q.shape[:3]
    if len(forecast.apps) != n_apps:
        raise ValueError("forecast cube app axis does not match its app list")

    median = forecast.median.astype(np.float32)                     # (apps, days, intervals)
    spread = (0.5 * forecast.spread(0.1, 0.9)).astype(np.float32)   # half-width

    reducer_objs = [get_reducer(name) for name in reducers]
    retain_intervals = any(r.needs_intervals for r in reducer_objs)

    # The rolling 4-hour average is accumulated inside the loop with a ring
    # buffer, never reconstructed afterwards -- reconstructing it would need
    # the interval series the memory contract exists to avoid. The buffer is
    # (chunk, window) floats, which is kilobytes.
    want_r4ha = needs_r4ha(reducers)
    interval_minutes = 1440 // n_int if n_int else 60
    r4ha_window = max(1, int(round(r4ha_hours * 60 / _interval_minutes(grid))))
    if want_r4ha and r4ha_window > n_int:
        LOG.warning(
            "a %.0f-hour window is %d intervals but the prime-time day is only %d. "
            "The rolling average will span the overnight gap, which is correct for "
            "MLC (IBM's window does not stop at 17:00) but means the figure depends "
            "on off-prime load this scope excludes -- see docs/peak_vs_average.md.",
            r4ha_hours, r4ha_window, n_int,
        )

    daily_peaks = np.empty((n_paths, n_days), dtype=np.float32)
    daily_means = np.empty((n_paths, n_days), dtype=np.float32)
    # Retained so the simulation can report its own coincidence factor against
    # the historical one. 20 MB at the default settings, and the single most
    # useful validation number the run produces.
    daily_sum_app_peaks = np.empty((n_paths, n_days), dtype=np.float32)
    app_peaks = np.zeros((n_paths, n_apps), dtype=np.float32)
    daily_r4ha = np.empty((n_paths, n_days), dtype=np.float32) if want_r4ha else None
    # Per-app R4HA, so the R4HA coincidence factor is measured like for like.
    # Dividing the R4HA of the sum by the sum of app *interval* peaks conflates
    # two different effects -- smoothing and coincidence -- and reads as a much
    # worse coincidence factor than reality.
    daily_sum_app_r4ha = np.empty((n_paths, n_days), dtype=np.float32) if want_r4ha else None

    rng = np.random.default_rng(seed)
    n_chunks = int(np.ceil(n_paths / path_chunk))
    peak_bytes = 0

    for c in range(n_chunks):
        lo = c * path_chunk
        hi = min(lo + path_chunk, n_paths)
        size = hi - lo
        chunk_app_max = np.zeros((size, n_apps), dtype=np.float32)
        chunk_intervals = (
            np.empty((size, n_days, n_int), dtype=np.float32) if retain_intervals else None
        )
        # Ring buffer of the last `r4ha_window` interval totals, carried across
        # day boundaries so a window straddling midnight is handled the way
        # IBM's actually is.
        ring = np.zeros((size, r4ha_window), dtype=np.float32) if want_r4ha else None
        # (size, apps, window): a few hundred KB, and the only way to get a
        # like-for-like R4HA coincidence factor.
        app_ring = (
            np.zeros((size, n_apps, r4ha_window), dtype=np.float32) if want_r4ha else None
        )
        ring_pos = 0
        ring_filled = 0

        # Lazy: only the day indices are held for the chunk, and each day's
        # residual slab is produced and discarded inside the loop below.
        stream = sampler.stream(size, n_days, rng)
        peak_bytes = max(peak_bytes, stream.nbytes() + size * n_apps * n_int * 4)

        for d in range(n_days):
            # (size, apps, intervals) -- the only place the app axis is wide.
            per_app = median[:, d, :][None] + stream.day(d) * spread[:, d, :][None]
            np.maximum(per_app, 0.0, out=per_app)

            # Per-app daily peak: each app free to peak in its own interval.
            app_daily_peak = per_app.max(axis=2)                    # (size, apps)
            daily_sum_app_peaks[lo:hi, d] = app_daily_peak.sum(axis=1)
            # Running max per (path, app): app-level numbers with no app history.
            np.maximum(chunk_app_max, app_daily_peak, out=chunk_app_max)

            # SUM ACROSS APPS, then take the maximum. Not the other way round.
            total = per_app.sum(axis=1)                             # (size, intervals)
            daily_peaks[lo:hi, d] = total.max(axis=1)
            daily_means[lo:hi, d] = total.mean(axis=1)
            if chunk_intervals is not None:
                chunk_intervals[:, d, :] = total

            if ring is not None:
                best = np.zeros(size, dtype=np.float32)
                best_app = np.zeros((size, n_apps), dtype=np.float32)
                for i in range(n_int):
                    ring[:, ring_pos] = total[:, i]
                    app_ring[:, :, ring_pos] = per_app[:, :, i]
                    ring_pos = (ring_pos + 1) % r4ha_window
                    ring_filled = min(ring_filled + 1, r4ha_window)
                    # Only a full window is a 4-hour average. Before the buffer
                    # fills -- the first hours of the horizon -- there is no
                    # honest value, so the day contributes nothing rather than
                    # a partial average that would read low.
                    if ring_filled == r4ha_window:
                        np.maximum(best, ring.mean(axis=1), out=best)
                        np.maximum(best_app, app_ring.mean(axis=2), out=best_app)
                daily_r4ha[lo:hi, d] = best
                # Each application free to have its own worst 4 hours, exactly
                # as each is free to have its own peak interval.
                daily_sum_app_r4ha[lo:hi, d] = best_app.sum(axis=1)

        app_peaks[lo:hi] = chunk_app_max
        if progress_every and (c + 1) % progress_every == 0:
            LOG.info("simulated %d/%d paths", hi, n_paths)

    fiscal_years = np.array([grid.fiscal_year(d) for d in forecast.days], dtype=int)
    result = SimulationResult(
        daily_peaks=daily_peaks,
        daily_means=daily_means,
        daily_sum_app_peaks=daily_sum_app_peaks,
        app_peaks=app_peaks,
        daily_r4ha=daily_r4ha,
        daily_sum_app_r4ha=daily_sum_app_r4ha,
        r4ha_window_intervals=r4ha_window if want_r4ha else 0,
        days=list(forecast.days),
        apps=list(forecast.apps),
        fiscal_years=fiscal_years,
        dependence=dependence,
        n_paths=n_paths,
    )
    _apply_reducers(result, reducer_objs, retain_intervals)

    result.diagnostics = {
        "n_paths": n_paths,
        "path_chunk": path_chunk,
        "n_days": n_days,
        "n_apps": n_apps,
        "n_intervals": n_int,
        "residual_scaling": residual_scaling,
        "r4ha_window_intervals": r4ha_window if want_r4ha else 0,
        "peak_chunk_working_mb": peak_bytes / 1e6,
        "accumulator_mb": (
            daily_peaks.nbytes
            + daily_means.nbytes
            + daily_sum_app_peaks.nbytes
            + app_peaks.nbytes
        )
        / 1e6,
        **result.coincidence(),
        **result.coincidence_by_target(),
        **getattr(sampler, "diagnostics", dict)(),
    }
    LOG.info(
        "simulated %d paths x %d days: peak chunk %.0fMB, accumulators %.0fMB, "
        "daily coincidence %.3f",
        n_paths,
        n_days,
        result.diagnostics["peak_chunk_working_mb"],
        result.diagnostics["accumulator_mb"],
        result.diagnostics["simulated_coincidence_daily_mean"],
    )
    return result


def _interval_minutes(grid: PrimeTimeGrid) -> int:
    return grid.interval_minutes


def _apply_reducers(
    result: SimulationResult, reducers: Sequence[_Reducer], retain_intervals: bool
) -> None:
    """Run each reducer over every path, whole-horizon and per fiscal year."""
    for reducer in reducers:
        if reducer.needs_intervals and retain_intervals:
            # Interval-level reducers are applied per chunk during sampling in a
            # future revision; for now they are rejected loudly rather than
            # returning a number computed from data that was not retained.
            raise NotImplementedError(
                f"reducer {reducer.name!r} needs interval retention across chunks, "
                "which the current loop does not carry beyond a chunk boundary"
            )
        whole = np.empty(result.n_paths, dtype=np.float64)
        by_fy: dict[int, list[float]] = {}
        for p in range(result.n_paths):
            path = PathSummary(
                daily_peaks=result.daily_peaks[p],
                daily_means=result.daily_means[p],
                days=result.days,
                daily_r4ha=None if result.daily_r4ha is None else result.daily_r4ha[p],
                fiscal_years=result.fiscal_years,
            )
            whole[p] = reducer(path)
            for fy, value in reduce_by_fiscal_year(path, reducer, result.fiscal_years).items():
                by_fy.setdefault(fy, []).append(value)
        result.reducer_values[reducer.name] = whole
        result.reducer_by_fy[reducer.name] = {
            fy: np.asarray(values) for fy, values in by_fy.items()
        }


def build_sampler(cfg: Config, panel, seed: int | None = None):
    """Construct the dependence model named in config."""
    from capplan.sim.bootstrap import BlockBootstrap
    from capplan.sim.copula import fit_gaussian_copula

    dependence = cfg.get("simulation.dependence")
    seed = int(cfg.get("simulation.seed")) if seed is None else seed
    if dependence == "block_bootstrap":
        return BlockBootstrap.from_panel(
            panel,
            block_days=int(cfg.get("simulation.block_bootstrap.block_days", 1)),
            seed=seed,
        )
    if dependence == "gaussian_copula":
        return fit_gaussian_copula(
            panel,
            shrinkage=float(cfg.get("simulation.gaussian_copula.shrinkage", 0.1)),
            seed=seed,
        )
    raise ValueError(f"unknown dependence model {dependence!r}")
