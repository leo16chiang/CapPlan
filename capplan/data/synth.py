"""Synthetic prime-time MIPS generator.

Not a toy. The pipeline has to be runnable and testable before the SMF extract
lands, and every property the architecture depends on has to be present in the
synthetic data or the tests prove nothing:

  * apps peak in *different* intervals, so peaks do not sum and the realised
    LPAR peak sits strictly below the sum of app peaks;
  * a common latent driver, so cross-app dependence is real and a bootstrap
    that ignores it will be visibly wrong;
  * within-day shape (ramp, lunch dip, afternoon peak) that day-block
    resampling must preserve;
  * heavy right tails, so mean-based losses are visibly biased for a maximum;
  * growth, day-of-week, month-end and quarter-end structure for the features
    to find;
  * DR / IST / GCC SDF windows landing in prime time, to be labelled out;
  * custodian submissions carrying a per-app bias for the submission-bias score
    to recover.

The generator returns the *true* per-app series and the *measured* LPAR totals
separately, so a test can assert that the simulation recovers a coincidence
structure it was never told.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd

from capplan.data.calendar import PrimeTimeGrid


@dataclass
class SynthSpec:
    n_apps: int = 35
    lpars: Sequence[str] = ("PRDA", "PRDB")
    start: date = date(2022, 11, 1)
    end: date = date(2025, 10, 31)
    seed: int = 7
    # Annual growth, drawn per app: most apps flat-ish, a few growing hard.
    growth_lognormal_sigma: float = 0.12
    # Weight of the shared latent driver. This is the coincidence knob: at 0.0
    # apps are independent and the sum-of-peaks overstates badly; at 1.0 they
    # move together and coincidence approaches 1.
    common_factor_weight: float = 0.35
    heavy_tail_df: float = 4.0     # Student-t tails on the idiosyncratic noise
    noise_cv: float = 0.10         # coefficient of variation of the noise
    n_dr_events: int = 4
    n_ist_events: int = 6
    n_gcc_events: int = 3
    submission_bias_sigma: float = 0.18
    app_prefix: str = "APP"
    extra: dict = field(default_factory=dict)


def _intraday_shape(n_intervals: int, peak_pos: float, width: float) -> np.ndarray:
    """Bimodal working-day profile: morning ramp, lunch dip, afternoon peak."""
    x = np.linspace(0.0, 1.0, n_intervals)
    main = np.exp(-0.5 * ((x - peak_pos) / width) ** 2)
    morning = 0.55 * np.exp(-0.5 * ((x - 0.18) / 0.12) ** 2)
    lunch_dip = 0.25 * np.exp(-0.5 * ((x - 0.45) / 0.07) ** 2)
    shape = 0.35 + main + morning - lunch_dip
    return np.clip(shape, 0.05, None)


def generate(grid: PrimeTimeGrid, spec: SynthSpec = SynthSpec()) -> dict[str, pd.DataFrame]:
    """Return dict with keys: intervals, events, submissions, lpar_totals, truth."""
    rng = np.random.default_rng(spec.seed)
    days = grid.business_days(spec.start, spec.end)
    if not days:
        raise ValueError("synthetic window contains no business days")
    n_days, n_int = len(days), grid.intervals_per_day
    apps = [f"{spec.app_prefix}{i:03d}" for i in range(1, spec.n_apps + 1)]
    lpars = list(spec.lpars)
    app_lpar = {a: lpars[i % len(lpars)] for i, a in enumerate(apps)}

    # Per-app character. Peak positions are spread deliberately across the
    # prime window -- this is what makes peaks fail to sum.
    base_level = np.exp(rng.normal(np.log(120.0), 0.9, spec.n_apps))
    peak_pos = rng.uniform(0.15, 0.92, spec.n_apps)
    peak_width = rng.uniform(0.06, 0.22, spec.n_apps)
    growth = np.exp(rng.normal(0.03, spec.growth_lognormal_sigma, spec.n_apps))
    dow_effect = 1.0 + rng.normal(0.0, 0.05, (spec.n_apps, 5))
    month_end_lift = np.exp(rng.normal(0.12, 0.10, spec.n_apps))
    quarter_end_lift = np.exp(rng.normal(0.22, 0.12, spec.n_apps))
    common_beta = rng.uniform(0.3, 1.6, spec.n_apps)

    shapes = np.stack(
        [_intraday_shape(n_int, peak_pos[a], peak_width[a]) for a in range(spec.n_apps)]
    )  # (apps, intervals)

    day_arr = np.array(days, dtype="object")
    years_elapsed = np.array([(d - days[0]).days / 365.25 for d in days])
    dow = np.array([d.weekday() for d in days])
    is_month_end = np.array([_is_last_n_business_days(days, i, 2) for i in range(n_days)])
    is_quarter_end = is_month_end & np.isin([d.month for d in days], [3, 6, 9, 12])

    # Shared latent driver: an AR(1) day-level factor plus an intra-day factor.
    # Every app loads on it, which is where cross-app coincidence comes from.
    day_factor = _ar1(rng, n_days, rho=0.55, sigma=0.09)
    intraday_factor = rng.normal(0.0, 0.05, (n_days, n_int))

    rows_mips = np.empty((spec.n_apps, n_days, n_int), dtype=np.float64)
    for a in range(spec.n_apps):
        level = base_level[a] * growth[a] ** years_elapsed
        level = level * dow_effect[a][dow]
        level = level * np.where(is_month_end, month_end_lift[a], 1.0)
        level = level * np.where(is_quarter_end, quarter_end_lift[a], 1.0)
        common = 1.0 + spec.common_factor_weight * common_beta[a] * (
            day_factor[:, None] + intraday_factor
        )
        mean = level[:, None] * shapes[a][None, :] * common
        # Student-t multiplicative noise: right tail heavy enough that the
        # sample maximum is genuinely hard, which is the point of the exercise.
        t = rng.standard_t(spec.heavy_tail_df, size=(n_days, n_int))
        noise = 1.0 + spec.noise_cv * t / np.sqrt(spec.heavy_tail_df / (spec.heavy_tail_df - 2))
        rows_mips[a] = np.clip(mean * noise, 0.5, None)

    truth = rows_mips.copy()  # pre-anomaly signal, kept for tests

    events, rows_mips = _inject_events(rng, spec, grid, days, apps, app_lpar, rows_mips)

    intervals = _to_long(grid, days, apps, app_lpar, rows_mips)
    intervals = _apply_event_labels(intervals, events)

    lpar_totals = _lpar_totals(grid, days, apps, app_lpar, rows_mips)
    submissions = _submissions(rng, spec, grid, days, apps, truth)

    return {
        "intervals": intervals,
        "events": events,
        "submissions": submissions,
        "lpar_totals": lpar_totals,
        "truth": truth,
        "apps": apps,
        "days": day_arr,
    }


def _ar1(rng: np.random.Generator, n: int, rho: float, sigma: float) -> np.ndarray:
    out = np.zeros(n)
    innov = rng.normal(0.0, sigma, n)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + innov[i]
    return out


def _is_last_n_business_days(days: Sequence[date], i: int, n: int) -> bool:
    month = days[i].month
    remaining = 0
    for j in range(i + 1, min(i + n + 1, len(days))):
        if days[j].month == month:
            remaining += 1
    return remaining < n


def _inject_events(rng, spec, grid, days, apps, app_lpar, mips):
    """Insert DR / IST / GCC SDF windows that land inside prime time."""
    records = []
    n_days, n_int = mips.shape[1], mips.shape[2]
    plans = (
        [("DR", spec.n_dr_events, 3.0, 1.0)]
        + [("IST", spec.n_ist_events, 1.8, 0.35)]
        + [("GCC_SDF", spec.n_gcc_events, 2.4, 0.6)]
    )
    for event_type, count, multiplier, app_share in plans:
        for k in range(count):
            d = int(rng.integers(0, n_days))
            start_i = int(rng.integers(0, max(1, n_int - 8)))
            span = int(rng.integers(4, 12))
            end_i = min(n_int, start_i + span)
            if app_share >= 1.0:
                touched = list(range(len(apps)))
                app_field = "*"
            else:
                k_apps = max(1, int(app_share * len(apps)))
                touched = list(rng.choice(len(apps), size=k_apps, replace=False))
                app_field = "*" if k_apps > 1 else apps[touched[0]]
            for a in touched:
                mips[a, d, start_i:end_i] *= multiplier
            day_ts = grid.day_timestamps(days[d])
            records.append(
                {
                    "event_id": f"{event_type}-{k:03d}",
                    "event_type": event_type,
                    "lpar": "*",
                    "app_id": app_field,
                    "start_ts": pd.Timestamp(day_ts[start_i]),
                    "end_ts": pd.Timestamp(day_ts[end_i - 1]) + pd.Timedelta(
                        minutes=grid.interval_minutes
                    ),
                    "note": f"synthetic {event_type} exercise",
                }
            )
    return pd.DataFrame(records), mips


def _to_long(grid, days, apps, app_lpar, mips) -> pd.DataFrame:
    n_apps, n_days, n_int = mips.shape
    times = grid.interval_starts
    ts = pd.to_datetime(
        [pd.Timestamp(d.year, d.month, d.day, t.hour, t.minute) for d in days for t in times]
    )
    per_app = n_days * n_int
    frames = []
    for a, app in enumerate(apps):
        frames.append(
            pd.DataFrame(
                {
                    "ts": ts,
                    "business_date": np.repeat(np.array(days, dtype="object"), n_int),
                    "interval_idx": np.tile(np.arange(n_int, dtype="int16"), n_days),
                    "fiscal_year": np.repeat(
                        np.array([grid.fiscal_year(d) for d in days], dtype="int16"), n_int
                    ),
                    "app_id": np.full(per_app, app, dtype="object"),
                    "lpar": np.full(per_app, app_lpar[app], dtype="object"),
                    "environment": np.full(per_app, "PROD", dtype="object"),
                    "mips": mips[a].reshape(-1),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    out["capture_ratio"] = 1.0
    out["mips_per_msu"] = 6.0
    out["msu"] = out["mips"] * out["capture_ratio"] / out["mips_per_msu"]
    out["is_anomaly"] = False
    out["event_label"] = None
    return out


def _apply_event_labels(intervals: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    from capplan.data.event_labels import label_from_windows

    return label_from_windows(intervals, events)


def _lpar_totals(grid, days, apps, app_lpar, mips) -> pd.DataFrame:
    """Realised LPAR peaks -- the SMF 70-1 ground truth.

    Measured at the LPAR, so it is the peak of the sum, carrying the true
    coincidence. Nothing downstream is allowed to reconstruct this by summing
    app peaks; that comparison is exactly what the diagnostics test.
    """
    by_lpar: dict[str, list[int]] = {}
    for i, app in enumerate(apps):
        by_lpar.setdefault(app_lpar[app], []).append(i)
    rows = []
    for lpar, idx in by_lpar.items():
        total = mips[idx].sum(axis=0)          # (days, intervals)
        peak = total.max(axis=1)
        peak_at = total.argmax(axis=1)
        mean = total.mean(axis=1)
        for j, d in enumerate(days):
            rows.append(
                {
                    "business_date": d,
                    "lpar": lpar,
                    "fiscal_year": grid.fiscal_year(d),
                    "peak_mips": float(peak[j]),
                    "peak_interval_idx": int(peak_at[j]),
                    "mean_mips": float(mean[j]),
                }
            )
    return pd.DataFrame(rows)


def _submissions(rng, spec, grid, days, apps, truth) -> pd.DataFrame:
    """Custodian submissions, each carrying a persistent per-app bias.

    The bias is persistent by design: that is what makes a per-app submission
    bias score worth computing at all.
    """
    bias = np.exp(rng.normal(0.10, spec.submission_bias_sigma, len(apps)))
    fys = sorted({grid.fiscal_year(d) for d in days})
    day_fy = np.array([grid.fiscal_year(d) for d in days])
    rows = []
    for a, app in enumerate(apps):
        for fy in fys:
            mask = day_fy == fy
            if not mask.any():
                continue
            realised_peak = float(truth[a][mask].max())
            noise = float(np.exp(rng.normal(0.0, 0.07)))
            rows.append(
                {
                    "app_id": app,
                    "fiscal_year": int(fy),
                    "submitted_on": grid.fiscal_year_bounds(fy)[0],
                    "submitted_peak_mips": realised_peak * bias[a] * noise,
                    "basis": rng.choice(["volume growth", "flat", "project uplift"]),
                }
            )
    return pd.DataFrame(rows)
