"""Backtest the simulation, not just the point forecast.

This is the module that decides whether any of the rest is trustworthy.

Procedure, per fold:
  1. Cut history at an origin date. Fit Stage 1 on everything before it.
  2. Build residuals, fit the dependence model -- on the training side only.
  3. Simulate the *already realised* window forward.
  4. Compare the simulated distribution of the fiscal-year figure against what
     the LPAR actually did, taken from SMF 70-1.

The comparison is deliberately against the realised LPAR peak rather than a
reconstruction from app rows. The LPAR figure is measured at the LPAR, so it
already contains the true coincidence; reconstructing it by summing app rows
would test the simulation against the same assumption the simulation makes,
which is not a test.

One trap, and the backtest walked straight into it on the first run: SMF 70-1
contains the DR exercise. DR, IST and GCC SDF are explicitly out of scope as
forecast targets, so the simulation does not model them -- and comparing a
simulation that excludes them against a realised maximum that includes them
gives PIT 1.0 on every fold and a "the simulation runs cold" verdict that is
pure measurement error. The realised figure is therefore taken on anomaly-free
days, and the anomaly-inclusive figure is reported alongside it: "your realised
annual maximum was 10,001 MIPS on a DR Saturday, and the in-scope forecast is
6,100" is a true and useful sentence. Quietly dropping either half is not.

What comes out:
  `pit`       where in the simulated distribution the realised value fell.
              Across folds these should look uniform. Systematically high means
              the simulation runs cold and the plan will under-provision.
  `coverage`  did the realised value fall inside the simulated 50/80/90/95
              intervals at roughly the advertised rate.
  `coincidence_error` simulated daily coincidence minus historical. The
              mechanism check: if this is off, the peak will be off in the same
              direction and everything else here is downstream of it.

Folds are few -- a fiscal-year figure needs a fiscal year of realised data, so
three years of history yields a handful of honest origins, not fifty. That is a
real limitation and it is reported rather than papered over with overlapping
windows that would share most of their data and pretend to be independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd

from capplan.config import Config
from capplan.data.calendar import PrimeTimeGrid
from capplan.logging_utils import get_logger
from capplan.model.features import anomaly_mask
from capplan.model.train import fit_stage1, fitted_quantiles, forecast_horizon
from capplan.sim.reducers import get_reducer
from capplan.sim.residuals import compute_residuals
from capplan.sim.simulate import build_sampler, simulate

LOG = get_logger(__name__)


@dataclass
class FoldResult:
    origin: date
    eval_days: tuple[date, ...]
    reducer: str
    realised: float                      # anomaly-free days only: what is in scope
    realised_with_anomalies: float       # what SMF 70-1 actually recorded
    n_anomaly_days_excluded: int
    simulated_mean: float
    simulated_p50: float
    simulated_p90: float
    simulated_p95: float
    pit: float
    simulated_coincidence: float
    historical_coincidence: float
    n_paths: int

    @property
    def coincidence_error(self) -> float:
        return self.simulated_coincidence - self.historical_coincidence


@dataclass
class SimBacktestResult:
    folds: list[FoldResult] = field(default_factory=list)
    reducer: str = "annual_max"
    dependence: str = "block_bootstrap"

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([f.__dict__ | {"coincidence_error": f.coincidence_error} for f in self.folds])

    def summary(self) -> dict[str, float]:
        if not self.folds:
            return {"n_folds": 0}
        frame = self.frame()
        pit = frame["pit"].to_numpy()
        inside_80 = float(((pit > 0.1) & (pit < 0.9)).mean())
        return {
            "n_folds": len(self.folds),
            "reducer": self.reducer,
            "pit_mean": float(pit.mean()),
            "pit_min": float(pit.min()),
            "pit_max": float(pit.max()),
            "share_inside_80pct_interval": inside_80,
            "mean_relative_error": float(
                ((frame["simulated_p50"] - frame["realised"]) / frame["realised"]).mean()
            ),
            "mean_coincidence_error": float(frame["coincidence_error"].mean()),
            "max_abs_coincidence_error": float(frame["coincidence_error"].abs().max()),
            "anomaly_days_excluded": int(frame["n_anomaly_days_excluded"].sum()),
            # How much of the realised maximum is out-of-scope event load. A
            # large gap is not a model problem, but it is the first question a
            # capacity manager will ask about the headline number.
            "mean_anomaly_uplift_pct": float(
                100.0
                * (frame["realised_with_anomalies"] / frame["realised"] - 1.0).mean()
            ),
        }

    def verdict(self, coincidence_tolerance: float = 0.03) -> str:
        """Plain-language read for the go/no-go conversation.

        Deliberately reports magnitude and direction rather than a pass/fail
        flag. With three years of history the folds overlap and there are only
        a handful of them, so neither a pass nor a failure here carries much
        statistical weight -- but a *direction* that is consistent across folds
        is still actionable, and the size of the gap is the number that goes
        into the headroom decision.
        """
        if not self.folds:
            return "NO FOLDS: not enough history to backtest the simulation."
        stats = self.summary()
        lines = [
            f"{stats['n_folds']} fold(s), reducer {stats['reducer']}, dependence "
            f"{self.dependence}. Folds overlap, so treat this as a direction check, "
            "not a hypothesis test."
        ]

        # The mechanism check comes first: everything else is downstream of it.
        if abs(stats["mean_coincidence_error"]) > coincidence_tolerance:
            direction = "over" if stats["mean_coincidence_error"] > 0 else "under"
            lines.append(
                f"MECHANISM FAIL: the simulation {direction}states the daily coincidence "
                f"factor by {abs(stats['mean_coincidence_error']):.3f}. The forward peak "
                "is wrong in the same direction and nothing below this line matters "
                "until it is fixed."
            )
        else:
            lines.append(
                f"Mechanism OK: daily coincidence reproduced to "
                f"{abs(stats['mean_coincidence_error']):.4f}."
            )

        error = stats["mean_relative_error"]
        pit = stats["pit_mean"]
        if pit > 0.75:
            lines.append(
                f"LEVEL: realised peaks land at PIT {pit:.2f}; the simulated median sits "
                f"{abs(100 * error):.1f}% below what actually happened. The simulation "
                f"runs cold. {_cold_explanation(self.dependence)} Size against an upper "
                "quantile rather than the median, and check whether the other dependence "
                "model agrees before adding headroom by hand -- agreement between the "
                "bootstrap and the copula is worth more than either number alone."
            )
        elif pit < 0.25:
            lines.append(
                f"LEVEL: realised peaks land at PIT {pit:.2f}; the simulated median sits "
                f"{abs(100 * error):.1f}% above what actually happened. The simulation runs "
                "hot and the plan will over-provision."
            )
        else:
            lines.append(
                f"LEVEL OK: realised peaks land at PIT {pit:.2f} "
                f"({100 * error:+.1f}% median error)."
            )

        if stats.get("anomaly_days_excluded"):
            lines.append(
                f"Scope: {stats['anomaly_days_excluded']} anomaly day(s) excluded from the "
                f"realised figure. Including DR/IST/GCC SDF would raise it by "
                f"{stats['mean_anomaly_uplift_pct']:.0f}% -- out of scope to forecast, but "
                "the hardware still has to survive it."
            )
        return "\n".join(lines)


def _cold_explanation(dependence: str) -> str:
    if dependence == "block_bootstrap":
        return (
            "Expected direction for a block bootstrap: it can only reproduce coincidence "
            "patterns that have actually occurred, so it under-samples the extreme tail."
        )
    if dependence == "gaussian_copula":
        return (
            "Expected direction for a Gaussian copula: it has no tail dependence, so "
            "coincident extremes across apps are systematically under-generated."
        )
    return "Investigate the dependence model before trusting the tail."


def historical_coincidence(intervals: pd.DataFrame, days: Sequence[date]) -> float:
    """Realised daily coincidence over a window: peak of sum / sum of app peaks."""
    window = intervals[intervals["business_date"].isin(set(days))]
    if "is_anomaly" in window.columns:
        window = window[~window["is_anomaly"]]
    if window.empty:
        return float("nan")
    peak_of_sum = (
        window.groupby(["business_date", "interval_idx"])["mips"].sum().groupby("business_date").max()
    )
    sum_of_app_peaks = (
        window.groupby(["business_date", "app_id"])["mips"].max().groupby("business_date").sum()
    )
    return float((peak_of_sum / sum_of_app_peaks).mean())


def anomaly_days(intervals: pd.DataFrame) -> set:
    """Business dates carrying any labelled DR / IST / GCC SDF activity."""
    if "is_anomaly" not in intervals.columns:
        return set()
    return set(intervals.loc[intervals["is_anomaly"], "business_date"].unique())


def realised_lpar_figure(
    lpar_totals: pd.DataFrame,
    days: Sequence[date],
    reducer_name: str,
    exclude_days: set | None = None,
) -> tuple[float, float, int]:
    """Apply the reducer to the realised SMF 70-1 LPAR peak series.

    Returns (in_scope, including_anomalies, n_days_excluded).

    LPARs are summed per day before reducing, because the simulation covers
    every scoped app across every production LPAR. Reducing each LPAR
    separately and adding the results would reintroduce a sum-of-peaks error at
    the LPAR level -- the same mistake one layer up.
    """
    from capplan.sim.reducers import PathSummary

    window = lpar_totals[lpar_totals["business_date"].isin(set(days))]
    if window.empty:
        return float("nan"), float("nan"), 0
    per_day = window.groupby("business_date")["peak_mips"].sum().sort_index()

    def reduce_series(series: pd.Series) -> float:
        if series.empty:
            return float("nan")
        path = PathSummary(
            daily_peaks=series.to_numpy(dtype=float),
            daily_means=series.to_numpy(dtype=float),
            days=list(series.index),
        )
        return float(get_reducer(reducer_name)(path))

    with_anomalies = reduce_series(per_day)
    excluded = set(exclude_days or ())
    clean = per_day[~per_day.index.isin(excluded)]
    return reduce_series(clean), with_anomalies, int(len(per_day) - len(clean))


def backtest_simulation(
    intervals: pd.DataFrame,
    lpar_totals: pd.DataFrame,
    grid: PrimeTimeGrid,
    cfg: Config,
    origins: Sequence[date] | None = None,
    eval_horizon_days: int = 250,
    n_paths: int = 2000,
    reducer: str = "annual_max",
) -> SimBacktestResult:
    """Run the simulation backtest over one or more origins."""
    all_days = sorted(intervals["business_date"].unique())
    origins = list(origins) if origins else default_origins(all_days, eval_horizon_days)
    result = SimBacktestResult(reducer=reducer, dependence=str(cfg.get("simulation.dependence")))

    for origin in origins:
        train_days = [d for d in all_days if d <= origin]
        eval_days = [d for d in all_days if d > origin][:eval_horizon_days]
        if len(eval_days) < eval_horizon_days // 2:
            LOG.warning("origin %s has only %d evaluation days; skipping", origin, len(eval_days))
            continue
        min_train = int(cfg.get("evaluation.rolling_origin.min_train_days", 250))
        if len(train_days) < min_train:
            LOG.warning(
                "origin %s has only %d training days (need %d); skipping",
                origin, len(train_days), min_train,
            )
            continue

        LOG.info(
            "backtest fold: train %s..%s (%d days), evaluate %s..%s (%d days)",
            train_days[0], train_days[-1], len(train_days),
            eval_days[0], eval_days[-1], len(eval_days),
        )
        fold = _run_fold(
            intervals, lpar_totals, grid, cfg, train_days, eval_days, n_paths, reducer
        )
        if fold is not None:
            result.folds.append(fold)

    LOG.info("%s", result.verdict())
    return result


def _run_fold(
    intervals, lpar_totals, grid, cfg, train_days, eval_days, n_paths, reducer
) -> FoldResult | None:
    train_frame = intervals[intervals["business_date"].isin(set(train_days))]
    art = fit_stage1(train_frame, grid, cfg)

    panel = compute_residuals(
        observed=art.cube,
        predicted_q=fitted_quantiles(art),
        quantiles=art.quantiles,
        days=art.index.days,
        apps=art.index.apps,
        anomaly=anomaly_mask(train_frame, art.index),
        scaling=cfg.get("simulation.block_bootstrap.residual_scale", "spread"),
    )
    if panel.n_usable < 30:
        LOG.warning("fold has only %d usable residual days; skipping", panel.n_usable)
        return None

    forecast = forecast_horizon(art, eval_days, progress_every=0)
    sampler = build_sampler(cfg, panel)
    sim = simulate(
        forecast,
        sampler,
        grid,
        n_paths=n_paths,
        path_chunk=min(int(cfg.get("simulation.path_chunk")), n_paths),
        reducers=(reducer,),
        seed=int(cfg.get("simulation.seed")),
        dependence=cfg.get("simulation.dependence"),
        progress_every=0,
    )

    simulated = sim.reducer_values[reducer]
    realised, realised_all, n_excluded = realised_lpar_figure(
        lpar_totals, eval_days, reducer, exclude_days=anomaly_days(intervals)
    )
    if not np.isfinite(realised):
        LOG.warning("no realised LPAR figure for %s..%s; skipping", eval_days[0], eval_days[-1])
        return None
    if n_excluded:
        LOG.info(
            "excluded %d anomaly day(s) from the realised figure: in-scope %.0f MIPS vs "
            "%.0f MIPS including DR/IST/GCC SDF",
            n_excluded, realised, realised_all,
        )

    return FoldResult(
        origin=train_days[-1],
        eval_days=(eval_days[0], eval_days[-1]),
        reducer=reducer,
        realised=realised,
        realised_with_anomalies=realised_all,
        n_anomaly_days_excluded=n_excluded,
        simulated_mean=float(simulated.mean()),
        simulated_p50=float(np.quantile(simulated, 0.5)),
        simulated_p90=float(np.quantile(simulated, 0.9)),
        simulated_p95=float(np.quantile(simulated, 0.95)),
        pit=float((simulated <= realised).mean()),
        simulated_coincidence=float(sim.diagnostics["simulated_coincidence_daily_mean"]),
        historical_coincidence=historical_coincidence(intervals, eval_days),
        n_paths=n_paths,
    )


def default_origins(all_days: Sequence[date], eval_horizon_days: int) -> list[date]:
    """Origins that leave a full evaluation window of realised data after them.

    Spaced by half the evaluation horizon. Folds still overlap -- with three
    years of history there is no way around that -- so they are not independent
    and the effective number of tests is smaller than the fold count suggests.
    Saying so is part of the result.
    """
    n = len(all_days)
    last_usable = n - eval_horizon_days
    if last_usable <= 0:
        return []
    step = max(eval_horizon_days // 2, 1)
    positions = list(range(last_usable - 1, 0, -step))
    return [all_days[p] for p in sorted(positions)]
