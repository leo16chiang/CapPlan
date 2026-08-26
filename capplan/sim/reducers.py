"""Pluggable reduction from a simulated path to the fiscal-year figure.

This is the seam that matters most, because the question it answers is not
settled: nobody has yet pinned down whether "the FY27 number" means a single
annual maximum, the mean of the twelve monthly peaks, or the 95th percentile of
daily peaks. Those are materially different numbers -- on the synthetic panel
they differ by well over 10% -- and each has a constituency.

So it is written as `reduce(path) -> scalar` and the architecture stops caring.
When the definition is settled, one function is added here and nothing else
changes. If two definitions end up being needed for two audiences, both are
computed in the same run at no extra simulation cost.

Three families, not one
-----------------------

The original framing -- "is it an annual max, a mean of monthly peaks, or a
percentile of daily peaks?" -- assumed all candidates were reductions of an
interval peak series. They are not, and the one with a hard external definition
is not a peak at all.

**Interval peak.** The highest single interval. Sizes hardware: the machine has
to survive the moment, not the hour. Most sensitive to coincidence, and most
sensitive to interval length -- an hourly "peak" is an hourly *average*, and the
true peak inside that hour is higher.

**Rolling 4-hour average (R4HA).** IBM's sub-capacity MLC billing takes, for
each day, the highest 4-hour rolling average MSU, and charges on the highest
such value across the month. That is an externally fixed definition: it is not a
convention CapPlan gets to choose, and if the deliverable is software cost it is
the only correct answer. Averaging over four hours smooths the timing
differences that make peaks fail to sum, so its coincidence factor sits much
closer to 1 than the interval peak's -- which means the Stage 2 machinery earns
much less on an R4HA deliverable than on a hardware one. `coincidence_by_target`
in eval/ measures exactly that gap.

**Sustained level.** Percentiles and monthly means of daily peaks. Smoother,
estimated from hundreds of observations rather than one, and the right answer
for trending and chargeback. Not a sizing number.

Pick per decision, not per project. A hardware refresh and an MLC forecast are
different questions and honestly have different answers.

Memory contract
---------------
A reducer receives a `PathSummary`, not the raw interval cube. Materialising
paths x days x intervals x apps is 10k x 512 x 36 x 35 floats = 736 GB, which
is not a tuning problem, it is an architecture problem. The simulator therefore
reduces *inside* the sampling loop and hands the reducer:

    daily_peaks   (n_days,)  peak of the summed-across-apps total, per day
    daily_means   (n_days,)  mean of that total, per day
    daily_r4ha    (n_days,)  highest 4-hour rolling average that day
    days                     the business dates, so calendar-aware reducers work

`daily_r4ha` is accumulated inside the sampling loop with a four-interval
ring buffer rather than reconstructed afterwards, because reconstructing it
would need the interval series the memory contract exists to avoid.

Every reducer stated above is expressible from the daily peak series. One that
genuinely is not -- say a duration-above-threshold measure -- declares
`needs_intervals = True` and the simulator retains the interval series for that
path within the chunk, which is affordable per chunk and is not affordable
globally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Protocol, Sequence

import numpy as np


@dataclass
class PathSummary:
    """One simulated path, already reduced across apps.

    `daily_peaks[i]` is the maximum over the 36 prime-time intervals of the
    total across all apps on day i. It is a peak of a sum. Nothing in this
    object is a sum of peaks.
    """

    daily_peaks: np.ndarray
    daily_means: np.ndarray
    days: Sequence[date]
    daily_r4ha: np.ndarray | None = None
    intervals: np.ndarray | None = None   # (days, intervals), only if requested
    fiscal_years: np.ndarray | None = None

    def require_r4ha(self) -> np.ndarray:
        if self.daily_r4ha is None:
            raise ValueError(
                "this reducer needs the rolling 4-hour average, which the "
                "simulation did not accumulate. Pass an r4ha reducer to "
                "simulate() so the ring buffer is enabled."
            )
        return self.daily_r4ha

    def month_keys(self) -> np.ndarray:
        return np.array([d.year * 100 + d.month for d in self.days])

    def restrict(self, mask: np.ndarray) -> "PathSummary":
        return PathSummary(
            daily_peaks=self.daily_peaks[mask],
            daily_means=self.daily_means[mask],
            days=[d for d, keep in zip(self.days, mask) if keep],
            daily_r4ha=None if self.daily_r4ha is None else self.daily_r4ha[mask],
            intervals=None if self.intervals is None else self.intervals[mask],
            fiscal_years=None if self.fiscal_years is None else self.fiscal_years[mask],
        )


class Reducer(Protocol):
    """`reduce(path) -> scalar`. That is the entire contract."""

    name: str
    description: str
    needs_intervals: bool

    def __call__(self, path: PathSummary) -> float: ...


@dataclass
class _Reducer:
    name: str
    description: str
    fn: Callable[[PathSummary], float]
    needs_intervals: bool = False

    def __call__(self, path: PathSummary) -> float:
        return float(self.fn(path))


REDUCERS: dict[str, _Reducer] = {}


def register_reducer(
    name: str, description: str, needs_intervals: bool = False
) -> Callable:
    def wrap(fn: Callable[[PathSummary], float]) -> _Reducer:
        reducer = _Reducer(
            name=name, description=description, fn=fn, needs_intervals=needs_intervals
        )
        REDUCERS[name] = reducer
        return reducer

    return wrap


def get_reducer(name: str) -> _Reducer:
    if name not in REDUCERS:
        raise KeyError(f"unknown reducer {name!r}; available: {sorted(REDUCERS)}")
    return REDUCERS[name]


# --------------------------------------------------------------------------
# The candidate definitions of "the fiscal year number"
# --------------------------------------------------------------------------


@register_reducer(
    "annual_max",
    "Single highest prime-time interval in the fiscal year. The most "
    "conservative reading, and the one a hardware config is usually signed "
    "against. Also the noisiest: it is one observation out of ~9,200.",
)
def annual_max(path: PathSummary) -> float:
    return float(np.max(path.daily_peaks))


@register_reducer(
    "mean_of_monthly_peaks",
    "Mean of the twelve monthly maxima. Smoother than the annual max and less "
    "sensitive to a single unusual afternoon, at the cost of sitting below the "
    "level the hardware actually has to survive.",
)
def mean_of_monthly_peaks(path: PathSummary) -> float:
    keys = path.month_keys()
    peaks = [path.daily_peaks[keys == k].max() for k in np.unique(keys)]
    return float(np.mean(peaks))


@register_reducer(
    "max_of_monthly_peaks",
    "Maximum of the monthly maxima -- identical to annual_max, provided as an "
    "explicit alias so a pack can name the convention it used.",
)
def max_of_monthly_peaks(path: PathSummary) -> float:
    keys = path.month_keys()
    return float(max(path.daily_peaks[keys == k].max() for k in np.unique(keys)))


@register_reducer(
    "p95_of_daily_peaks",
    "95th percentile of the daily prime-time peaks: the level exceeded on "
    "roughly one business day a month. Often the most defensible choice, "
    "because it is estimated from ~250 observations rather than one.",
)
def p95_of_daily_peaks(path: PathSummary) -> float:
    return float(np.quantile(path.daily_peaks, 0.95))


@register_reducer(
    "p99_of_daily_peaks",
    "99th percentile of the daily prime-time peaks -- roughly the worst two or "
    "three business days of the year.",
)
def p99_of_daily_peaks(path: PathSummary) -> float:
    return float(np.quantile(path.daily_peaks, 0.99))


@register_reducer(
    "mean_of_daily_peaks",
    "Mean daily prime-time peak. Not a sizing number; useful as the denominator "
    "for a peak-to-average ratio.",
)
def mean_of_daily_peaks(path: PathSummary) -> float:
    return float(np.mean(path.daily_peaks))


@register_reducer(
    "top4_mean_of_daily_peaks",
    "Mean of the four highest daily peaks. A common utility-billing convention "
    "and a reasonable compromise between annual_max and a percentile.",
)
def top4_mean_of_daily_peaks(path: PathSummary) -> float:
    top = np.sort(path.daily_peaks)[-4:]
    return float(np.mean(top))


@register_reducer(
    "mean_daily_mean",
    "Mean prime-time utilisation across the fiscal year. Not a peak at all -- "
    "included so a pack can show the headroom between average and peak.",
)
def mean_daily_mean(path: PathSummary) -> float:
    return float(np.mean(path.daily_means))


# -- R4HA: the externally defined one ---------------------------------------


@register_reducer(
    "monthly_peak_r4ha",
    "Highest rolling 4-hour average in the month, averaged across the months "
    "of the fiscal year. This is IBM's sub-capacity MLC billing basis, so if "
    "the deliverable is software cost this is not a convention to choose but a "
    "definition to match.",
)
def monthly_peak_r4ha(path: PathSummary) -> float:
    r4ha = path.require_r4ha()
    keys = path.month_keys()
    return float(np.mean([r4ha[keys == k].max() for k in np.unique(keys)]))


@register_reducer(
    "annual_peak_r4ha",
    "Highest rolling 4-hour average anywhere in the fiscal year -- the worst "
    "month's MLC bill rather than the average one. The number to size a "
    "capacity-group cap against.",
)
def annual_peak_r4ha(path: PathSummary) -> float:
    return float(np.max(path.require_r4ha()))


@register_reducer(
    "p95_of_daily_r4ha",
    "95th percentile of the daily peak R4HA. Less exposed to a single unusual "
    "day than annual_peak_r4ha, and estimated from ~250 observations.",
)
def p95_of_daily_r4ha(path: PathSummary) -> float:
    return float(np.quantile(path.require_r4ha(), 0.95))


@register_reducer(
    "mean_of_daily_r4ha",
    "Mean daily peak R4HA. Not a billing figure; the denominator for showing "
    "how much of the MLC bill is driven by the worst few days of the month.",
)
def mean_of_daily_r4ha(path: PathSummary) -> float:
    return float(np.mean(path.require_r4ha()))


@register_reducer(
    "intervals_above_p99",
    "Count of prime-time intervals above the path's own 99th percentile. A "
    "duration measure rather than a level, and the example of a reducer that "
    "genuinely needs the interval series.",
    needs_intervals=True,
)
def intervals_above_p99(path: PathSummary) -> float:
    if path.intervals is None:
        raise ValueError("intervals_above_p99 requires retain_intervals=True")
    threshold = np.quantile(path.intervals, 0.99)
    return float((path.intervals > threshold).sum())


def reduce_by_fiscal_year(
    path: PathSummary, reducer: _Reducer, fiscal_years: np.ndarray
) -> dict[int, float]:
    """Apply a reducer separately within each fiscal year of the path.

    The deliverable is per fiscal year, so this is the form the pack actually
    consumes; the whole-path reducer is the building block.
    """
    out: dict[int, float] = {}
    for fy in np.unique(fiscal_years):
        mask = fiscal_years == fy
        out[int(fy)] = reducer(path.restrict(mask))
    return out


# Which decision each reducer is for. The pack prints this, because "which
# number do you want" is a question about the decision being made, not about
# statistics.
TARGET_FAMILY = {
    "annual_max": "hardware",
    "max_of_monthly_peaks": "hardware",
    "p99_of_daily_peaks": "hardware",
    "top4_mean_of_daily_peaks": "hardware",
    "mean_of_monthly_peaks": "sustained",
    "p95_of_daily_peaks": "sustained",
    "mean_of_daily_peaks": "sustained",
    "mean_daily_mean": "sustained",
    "monthly_peak_r4ha": "software_cost",
    "annual_peak_r4ha": "software_cost",
    "p95_of_daily_r4ha": "software_cost",
    "mean_of_daily_r4ha": "software_cost",
    "intervals_above_p99": "duration",
}

FAMILY_NOTES = {
    "hardware": (
        "Sizing the machine. Most exposed to coincidence and to interval "
        "length -- an hourly figure is an hourly average, so apply the "
        "peak-to-hourly uplift before signing anything."
    ),
    "software_cost": (
        "IBM sub-capacity MLC. Externally defined, not a choice. Four-hour "
        "averaging smooths timing differences, so coincidence matters much "
        "less here than for hardware."
    ),
    "sustained": "Trending and chargeback. Estimated from many observations, not one.",
    "duration": "How long, not how high.",
}


def needs_r4ha(names) -> bool:
    return any(TARGET_FAMILY.get(n) == "software_cost" for n in names)


def describe_reducers() -> str:
    """Table for the pack, so the convention in use is stated, not implied."""
    width = max(len(n) for n in REDUCERS)
    lines = []
    for family in ("hardware", "software_cost", "sustained", "duration"):
        lines.append("")
        lines.append(f"== {family.upper().replace('_', ' ')} ==")
        lines.append(f"   {FAMILY_NOTES[family]}")
        lines.append("")
        for name in sorted(REDUCERS):
            if TARGET_FAMILY.get(name) != family:
                continue
            reducer = REDUCERS[name]
            flag = " [needs intervals]" if reducer.needs_intervals else ""
            lines.append(f"   {name:<{width}}  {reducer.description}{flag}")
    return "\n".join(lines)
