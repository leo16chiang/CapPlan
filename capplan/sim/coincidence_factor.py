"""Transferring the coincidence factor from a short hourly sample.

The problem this solves. If the forecast has to run on daily data -- because
the hourly spine has no application code, or is a view that filters, or simply
does not exist -- then the coincidence factor cannot be measured from the
forecasting data at all. A daily table gives one number per application per
day; even a daily maximum does not say which hour it landed in, so two daily
maxima cannot be distinguished from two simultaneous ones.

Two bad options and one reasonable one.

  BAD: sum the daily application peaks. That is the 25-35% overstatement the
  whole project exists to remove.

  BAD: forecast the daily LPAR total directly. Correct at the LPAR level, but
  it has no application attribution, so there is nothing to take to a custodian
  and nothing to run a scenario against.

  REASONABLE: forecast the applications daily, sum their peaks, and multiply by
  a coincidence factor whose *distribution* was estimated from however much
  hourly data exists -- even one month. The factor is a ratio, and ratios
  transfer across periods far better than levels do.

What this costs, stated plainly:

  * The factor is assumed stable between the sample window and the forecast
    horizon. If the application mix changes materially, it is not.
  * A single month of sample gives a wide interval on the mean factor.
    `sample_adequacy` reports how wide, so the assumption is sized rather than
    waved at.
  * Day-to-day *variation* in the factor is captured (it is resampled, not
    applied as a constant), but any seasonality in it is not unless the sample
    spans the seasons.

Which is worse than measuring it directly, and much better than pretending
peaks sum.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from capplan.data.calendar import PrimeTimeGrid
from capplan.logging_utils import get_logger

LOG = get_logger(__name__)


@dataclass
class CoincidenceFactor:
    """Empirical distribution of peak-of-sum / sum-of-app-peaks.

    Resampled per (path, day) rather than applied as a constant: the factor
    varies day to day, and collapsing it to its mean would understate the
    spread of the forecast peak by exactly that variation.
    """

    samples: np.ndarray            # one factor per sampled day
    source_days: int
    grain_minutes: int
    n_apps: int
    by_weekday: dict[int, np.ndarray] = field(default_factory=dict)
    seed: int = 0

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=np.float64)
        self.samples = self.samples[np.isfinite(self.samples)]
        if self.samples.size == 0:
            raise ValueError("no usable coincidence samples")

    def draw(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Resample factors with replacement."""
        return self.samples[rng.integers(0, self.samples.size, size=size)]

    def draw_for_weekday(
        self, weekday: int, size: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Weekday-conditional draw, falling back to the pooled distribution.

        Monday and Friday genuinely differ at some sites -- a batch tail or an
        early close moves the intra-day shape and with it the coincidence. The
        fallback threshold is deliberately conservative: 20 days is already a
        thin estimate of a distribution, and below that the pooled version is
        less wrong than a weekday-specific one.
        """
        pool = self.by_weekday.get(weekday)
        if pool is None or pool.size < 20:
            pool = self.samples
        return pool[rng.integers(0, pool.size, size=size)]

    def summary(self) -> dict[str, float]:
        return {
            "coincidence_mean": float(self.samples.mean()),
            "coincidence_sd": float(self.samples.std(ddof=1)) if self.samples.size > 1 else 0.0,
            "coincidence_p05": float(np.quantile(self.samples, 0.05)),
            "coincidence_p50": float(np.quantile(self.samples, 0.5)),
            "coincidence_p95": float(np.quantile(self.samples, 0.95)),
            "sample_days": int(self.samples.size),
            "source_days": self.source_days,
            "grain_minutes": self.grain_minutes,
            "n_apps": self.n_apps,
        }

    def sample_adequacy(self, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
        """Bootstrap interval on the mean factor: is the sample big enough?

        The number to look at is `mean_ci_width_pct`. It is the uncertainty the
        transfer adds to every forecast figure, before any modelling
        uncertainty. Under ~2% it is noise beside everything else; over ~5% it
        is the dominant term and more hourly sample is the cheapest available
        improvement.
        """
        rng = np.random.default_rng(seed)
        n = self.samples.size
        means = self.samples[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
        lo, hi = np.quantile(means, [0.025, 0.975])
        mean = float(self.samples.mean())
        width_pct = 100.0 * (hi - lo) / mean if mean else float("nan")
        out = {
            "mean_ci_lo": float(lo),
            "mean_ci_hi": float(hi),
            "mean_ci_width_pct": float(width_pct),
            "sample_days": int(n),
        }
        out["verdict"] = _adequacy_verdict(width_pct, n)
        LOG.info(
            "coincidence sample: %d days, mean %.3f (95%% CI %.3f-%.3f, width %.1f%%) -- %s",
            n, mean, lo, hi, width_pct, out["verdict"],
        )
        return out


def _adequacy_verdict(width_pct: float, n_days: int) -> str:
    if n_days < 20:
        return (
            f"TOO THIN: {n_days} days. Below about 20 the factor's own variation is "
            "indistinguishable from sampling noise. Get more hourly sample before "
            "relying on this."
        )
    if width_pct > 5.0:
        return (
            f"WIDE: the 95% interval on the mean factor spans {width_pct:.1f}% of it, "
            "which will dominate the modelling uncertainty. More hourly sample is the "
            "cheapest improvement available to this forecast."
        )
    if width_pct > 2.0:
        return (
            f"WORKABLE: {width_pct:.1f}% interval on the mean factor. Material but not "
            "dominant; state it as an assumption in the pack."
        )
    return (
        f"ADEQUATE: {width_pct:.1f}% interval on the mean factor, small beside the "
        "forecast's own uncertainty."
    )


def estimate_from_intervals(
    intervals: pd.DataFrame,
    grid: PrimeTimeGrid,
    exclude_anomalies: bool = True,
    lpar_totals: pd.DataFrame | None = None,
) -> CoincidenceFactor:
    """Measure the factor from whatever sub-daily data exists.

    `intervals` needs `business_date`, `interval_idx`, `app_id`, `mips`. One
    month is enough to start; a fiscal year is enough to trust.

    When `lpar_totals` is supplied the numerator comes from the realised LPAR
    peak instead of the reconstructed sum, which additionally captures work not
    attributed to any scoped application. That is the better measurement, and
    the difference between the two is itself worth reporting.
    """
    frame = intervals
    if exclude_anomalies and "is_anomaly" in frame.columns:
        dirty = set(frame.loc[frame["is_anomaly"], "business_date"].unique())
        frame = frame[~frame["business_date"].isin(dirty)]
    if frame.empty:
        raise ValueError("no clean interval rows to estimate a coincidence factor from")

    sum_of_app_peaks = (
        frame.groupby(["business_date", "app_id"])["mips"].max().groupby("business_date").sum()
    )
    if lpar_totals is not None and not lpar_totals.empty:
        peak_of_sum = lpar_totals.groupby("business_date")["peak_mips"].sum()
        source = "realised LPAR peak (SMF 70-1)"
    else:
        peak_of_sum = (
            frame.groupby(["business_date", "interval_idx"])["mips"]
            .sum()
            .groupby("business_date")
            .max()
        )
        source = "reconstructed peak of the summed applications"

    joined = pd.DataFrame({"num": peak_of_sum, "den": sum_of_app_peaks}).dropna()
    joined = joined[joined["den"] > 0]
    factors = (joined["num"] / joined["den"]).to_numpy()

    weekday = {}
    for day, value in zip(joined.index, factors):
        weekday.setdefault(pd.Timestamp(day).weekday(), []).append(value)

    result = CoincidenceFactor(
        samples=factors,
        source_days=int(len(joined)),
        grain_minutes=grid.interval_minutes,
        n_apps=int(frame["app_id"].nunique()),
        by_weekday={k: np.asarray(v) for k, v in weekday.items()},
    )
    LOG.info(
        "coincidence factor from %d days at %d-minute grain, numerator = %s: "
        "mean %.3f (p05 %.3f, p95 %.3f)",
        result.source_days, grid.interval_minutes, source,
        *[result.summary()[k] for k in ("coincidence_mean", "coincidence_p05", "coincidence_p95")],
    )
    if grid.interval_minutes >= 60:
        LOG.warning(
            "the sample is %d-minute grain, so this factor is an UPPER BOUND on the "
            "true one: two applications peaking 20 minutes apart look simultaneous at "
            "hourly grain. The real overstatement from summing application peaks is "
            "larger than this says.",
            grid.interval_minutes,
        )
    return result


def save(factor: CoincidenceFactor, path) -> None:
    np.savez_compressed(
        path,
        samples=factor.samples,
        source_days=factor.source_days,
        grain_minutes=factor.grain_minutes,
        n_apps=factor.n_apps,
        **{f"weekday_{k}": v for k, v in factor.by_weekday.items()},
    )


def load(path) -> CoincidenceFactor:
    with np.load(path, allow_pickle=False) as data:
        by_weekday = {
            int(k.split("_")[1]): data[k] for k in data.files if k.startswith("weekday_")
        }
        return CoincidenceFactor(
            samples=data["samples"],
            source_days=int(data["source_days"]),
            grain_minutes=int(data["grain_minutes"]),
            n_apps=int(data["n_apps"]),
            by_weekday=by_weekday,
        )
