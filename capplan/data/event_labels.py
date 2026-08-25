"""Anomaly labelling for DR, IST and GCC SDF activity.

These are explicitly out of scope as forecast *targets* -- nobody wants a
two-year forecast of disaster-recovery load. But they land in prime time, and
if they are left unlabelled two things go wrong:

  1. Stage 1 fits them as ordinary variation and inflates every marginal.
  2. Stage 2 resamples the DR day as if it were a normal Tuesday, so one
     historical DR exercise becomes a recurring feature of the forward
     distribution.

So they are labelled, excluded from training and from the residual pool, and
reported separately -- the count of prime-time anomaly intervals is itself a
number the custodian interview will ask about.

Two labelling routes, used together:
  * `label_from_windows` -- authoritative. The change calendar knows when the
    DR test ran.
  * `label_unexplained_spikes` -- the safety net. Robust z-score on the
    per-(app, interval) residual from a rolling median. Flags what the change
    calendar missed, for a human to confirm; never silently treated as known.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from capplan.logging_utils import get_logger

LOG = get_logger(__name__)

WILDCARD = "*"
UNEXPLAINED = "UNEXPLAINED_SPIKE"


@dataclass(frozen=True)
class SpikeRule:
    """Robust-z rule for unexplained spikes.

    Median absolute deviation rather than standard deviation, because the thing
    being detected is exactly the thing that would inflate a standard deviation.
    """

    window_days: int = 20
    z_threshold: float = 6.0
    min_abs_mips: float = 0.0  # ignore spikes on trivially small apps


def label_from_windows(intervals: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Mark intervals falling inside a known event window.

    `events` rows may wildcard `lpar` and/or `app_id` with '*'. Overlapping
    events resolve to the first match in event order, which keeps the label
    deterministic.
    """
    out = intervals.copy()
    if "is_anomaly" not in out.columns:
        out["is_anomaly"] = False
    if "event_label" not in out.columns:
        out["event_label"] = None

    if events is None or events.empty:
        return out

    ts = out["ts"].to_numpy(dtype="datetime64[ns]")
    lpar = out["lpar"].to_numpy()
    app = out["app_id"].to_numpy()

    for row in events.itertuples(index=False):
        in_window = (ts >= np.datetime64(row.start_ts)) & (ts < np.datetime64(row.end_ts))
        if not in_window.any():
            continue
        if row.lpar != WILDCARD:
            in_window &= lpar == row.lpar
        if row.app_id != WILDCARD:
            in_window &= app == row.app_id
        fresh = in_window & ~out["is_anomaly"].to_numpy()
        if fresh.any():
            out.loc[fresh, "is_anomaly"] = True
            out.loc[fresh, "event_label"] = row.event_type

    LOG.info(
        "labelled %d prime-time intervals from %d event windows",
        int(out["is_anomaly"].sum()),
        len(events),
    )
    return out


def label_unexplained_spikes(
    intervals: pd.DataFrame, rule: SpikeRule = SpikeRule()
) -> pd.DataFrame:
    """Flag spikes the change calendar does not explain.

    Adds `spike_z` and `is_unexplained_spike`. Deliberately does NOT set
    `is_anomaly`: an unexplained spike might be a genuine business peak, which
    is precisely the thing being forecast. It goes on a review list instead.
    """
    out = intervals.copy()
    out["spike_z"] = 0.0
    out["is_unexplained_spike"] = False

    for (_app, _idx), grp in out.groupby(["app_id", "interval_idx"], sort=False):
        series = grp["mips"].astype(float)
        if len(series) < max(5, rule.window_days // 2):
            continue
        med = series.rolling(rule.window_days, min_periods=5, center=True).median()
        med = med.bfill().ffill()
        dev = (series - med).abs()
        mad = dev.rolling(rule.window_days, min_periods=5, center=True).median()
        mad = mad.bfill().ffill()
        # 1.4826 makes MAD a consistent estimator of sigma under normality.
        scale = 1.4826 * mad.replace(0.0, np.nan)
        z = (series - med) / scale
        z = z.fillna(0.0)
        out.loc[grp.index, "spike_z"] = z.to_numpy()

    known = out["is_anomaly"].to_numpy() if "is_anomaly" in out.columns else np.zeros(len(out), bool)
    flagged = (
        (out["spike_z"].to_numpy() > rule.z_threshold)
        & (out["mips"].to_numpy() >= rule.min_abs_mips)
        & ~known
    )
    out["is_unexplained_spike"] = flagged
    LOG.info(
        "flagged %d unexplained prime-time spikes (z > %.1f) for review",
        int(flagged.sum()),
        rule.z_threshold,
    )
    return out


def anomaly_report(intervals: pd.DataFrame) -> pd.DataFrame:
    """Per-(app, label) counts and severity. Goes straight into the pack."""
    frames = []
    if "event_label" in intervals.columns:
        known = intervals[intervals["is_anomaly"]]
        if not known.empty:
            frames.append(
                known.groupby(["app_id", "event_label"])
                .agg(
                    intervals=("mips", "size"),
                    days=("business_date", "nunique"),
                    max_mips=("mips", "max"),
                )
                .reset_index()
                .rename(columns={"event_label": "label"})
            )
    if "is_unexplained_spike" in intervals.columns:
        spikes = intervals[intervals["is_unexplained_spike"]]
        if not spikes.empty:
            frame = (
                spikes.groupby("app_id")
                .agg(
                    intervals=("mips", "size"),
                    days=("business_date", "nunique"),
                    max_mips=("mips", "max"),
                )
                .reset_index()
            )
            frame["label"] = UNEXPLAINED
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["app_id", "label", "intervals", "days", "max_mips"])
    out = pd.concat(frames, ignore_index=True)
    return out[["app_id", "label", "intervals", "days", "max_mips"]].sort_values(
        ["app_id", "label"]
    )


def clean_days(intervals: pd.DataFrame) -> np.ndarray:
    """Business dates with no labelled anomaly on any app.

    Day-block bootstrap resamples whole days across all apps at once, so a day
    is only usable if it is clean *everywhere*. Dropping the whole day is the
    price of preserving cross-app coincidence.
    """
    if "is_anomaly" not in intervals.columns:
        return np.array(sorted(intervals["business_date"].unique()))
    dirty = set(intervals.loc[intervals["is_anomaly"], "business_date"].unique())
    all_days = sorted(intervals["business_date"].unique())
    kept = np.array([d for d in all_days if d not in dirty], dtype="object")
    LOG.info("%d of %d business days are anomaly-free", len(kept), len(all_days))
    return kept
