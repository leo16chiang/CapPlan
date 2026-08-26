"""One test that runs the whole thing, and the scope checks around it."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from capplan.data.event_labels import SpikeRule, clean_days, label_from_windows, label_unexplained_spikes
from capplan.data.ingest import scope_intervals
from capplan.data.mips_normalisation import NormalisationTable


def test_scope_drops_non_prod_and_off_prime(grid, cfg):
    """Everything downstream assumes the lake is already scoped, so the
    'what's in, what's out' argument lives in exactly one function."""
    rows = []
    for env in ("PROD", "TEST"):
        for hour in (7, 9, 18):
            rows.append(
                {
                    "ts": datetime(2026, 1, 5, hour, 0),
                    "app_id": "A",
                    "lpar": "PRDA",
                    "environment": env,
                    "mips": 100.0,
                    "msu": 16.0,
                }
            )
    rows.append(
        {
            "ts": datetime(2026, 1, 3, 9, 0),   # Saturday
            "app_id": "A", "lpar": "PRDA", "environment": "PROD", "mips": 100.0, "msu": 16.0,
        }
    )
    # Stated explicitly rather than inherited: the shipped default is hourly,
    # and an interval index means nothing without the grain that produced it.
    from capplan.data.calendar import grid_from_config

    fine_cfg = cfg.with_overrides({"calendar.interval_minutes": 15})
    fine_grid = grid_from_config(fine_cfg)
    scoped, report = scope_intervals(
        pd.DataFrame(rows), fine_grid, fine_cfg, NormalisationTable.from_config(fine_cfg)
    )
    assert len(scoped) == 1                       # only PROD, 09:00, a weekday
    assert scoped.iloc[0]["interval_idx"] == 4     # 09:00 is the fifth 15-min interval
    assert report.dropped_non_prod == 3
    assert report.dropped_off_prime == 3


def test_top_n_apps_are_kept_by_prime_time_mean(grid, cfg):
    rows = []
    for app, level in (("BIG", 1000.0), ("MID", 100.0), ("SMALL", 1.0)):
        for interval in range(grid.intervals_per_day):
            start = grid.interval_starts[interval]
            rows.append(
                {
                    "ts": datetime(2026, 1, 5, start.hour, start.minute),
                    "app_id": app, "lpar": "PRDA", "environment": "PROD",
                    "mips": level, "msu": level / 6,
                }
            )
    scoped, report = scope_intervals(
        pd.DataFrame(rows), grid, cfg.with_overrides({"scope.top_n_apps": 2}),
        NormalisationTable.from_config(cfg),
    )
    assert set(scoped["app_id"]) == {"BIG", "MID"}
    assert report.apps_dropped == 1


def test_event_windows_label_only_what_they_cover(grid):
    ts = pd.to_datetime([datetime(2026, 1, 5, 8 + h, 0) for h in range(8)])
    intervals = pd.DataFrame(
        {
            "ts": ts,
            "business_date": [t.date() for t in ts],
            "interval_idx": range(8),
            "app_id": "A",
            "lpar": "PRDA",
            "mips": 100.0,
        }
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "DR-1", "event_type": "DR", "lpar": "*", "app_id": "*",
                "start_ts": pd.Timestamp(2026, 1, 5, 10, 0),
                "end_ts": pd.Timestamp(2026, 1, 5, 12, 0),
                "note": "",
            }
        ]
    )
    labelled = label_from_windows(intervals, events)
    assert labelled["is_anomaly"].sum() == 2
    assert set(labelled.loc[labelled["is_anomaly"], "event_label"]) == {"DR"}


def test_unexplained_spikes_are_flagged_for_review_not_silently_excluded(grid):
    """An unexplained spike might be a genuine business peak -- which is exactly
    the thing being forecast. It goes on a review list, not into is_anomaly."""
    n = 120
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-01", periods=n, freq="D"),
            "business_date": pd.date_range("2026-01-01", periods=n, freq="D").date,
            "interval_idx": 0,
            "app_id": "A",
            "lpar": "PRDA",
            "mips": 100.0 + np.random.default_rng(0).normal(0, 1, n),
            "is_anomaly": False,
        }
    )
    frame.loc[60, "mips"] = 400.0
    out = label_unexplained_spikes(frame, SpikeRule(window_days=20, z_threshold=6.0))
    assert out.loc[60, "is_unexplained_spike"]
    assert not out["is_anomaly"].any(), "must not be promoted to a known anomaly"
    assert out["is_unexplained_spike"].sum() == 1


def test_clean_days_drops_a_day_that_is_dirty_for_any_app():
    frame = pd.DataFrame(
        {
            "business_date": [date(2026, 1, 5)] * 2 + [date(2026, 1, 6)] * 2,
            "app_id": ["A", "B", "A", "B"],
            "is_anomaly": [False, True, False, False],
            "mips": 1.0,
        }
    )
    assert list(clean_days(frame)) == [date(2026, 1, 6)]


@pytest.mark.slow
def test_full_pipeline_produces_a_defensible_fiscal_year_number(tmp_path, synth, grid, cfg):
    """Ingest -> diagnostics -> Stage 1 -> residuals -> simulate -> reduce.

    Asserts the properties that would make the output indefensible if violated,
    not the exact numbers.
    """
    from capplan.data.ingest import write_lake
    from capplan.diagnostics.runner import connect, run_coincidence
    from capplan.model.features import anomaly_mask
    from capplan.model.train import fit_stage1, fitted_quantiles, forecast_horizon
    from capplan.serve.forecast_store import build_fy_summary
    from capplan.sim.bootstrap import BlockBootstrap
    from capplan.sim.residuals import compute_residuals
    from capplan.sim.simulate import simulate

    write_lake(
        {
            "intervals": synth["intervals"],
            "events": synth["events"],
            "submissions": synth["submissions"],
            "lpar_totals": synth["lpar_totals"],
        },
        tmp_path,
    )
    con = connect(tmp_path)
    try:
        historical = run_coincidence(con).headline["coincidence_mean"]
    finally:
        con.close()

    art = fit_stage1(synth["intervals"], grid, cfg)
    panel = compute_residuals(
        observed=art.cube,
        predicted_q=fitted_quantiles(art),
        quantiles=art.quantiles,
        days=art.index.days,
        apps=art.index.apps,
        anomaly=anomaly_mask(synth["intervals"], art.index),
        scaling="spread",
    )
    future = grid.horizon_business_days(max(art.index.days), 2)
    forecast = forecast_horizon(art, future, progress_every=0)
    result = simulate(
        forecast, BlockBootstrap.from_panel(panel, seed=42), grid,
        n_paths=200, path_chunk=100,
        reducers=("annual_max", "p95_of_daily_peaks"), progress_every=0,
    )

    # 1. The horizon reaches the end of the second fiscal year.
    fiscal_years = sorted(result.reducer_by_fy["annual_max"])
    assert len(fiscal_years) >= 2

    # 2. The coincidence mechanism is reproduced, not assumed.
    assert result.diagnostics["simulated_coincidence_daily_mean"] == pytest.approx(
        historical, abs=0.08
    )

    # 3. Peak of the sum is well below the sum of app peaks.
    assert result.daily_peaks.max() < result.app_peaks.sum(axis=1).max()

    # 4. The reduction convention materially changes the answer.
    fy = fiscal_years[-1]
    strict = np.quantile(result.reducer_by_fy["annual_max"][fy], 0.5)
    loose = np.quantile(result.reducer_by_fy["p95_of_daily_peaks"][fy], 0.5)
    assert strict > loose

    # 5. The number is in the right order of magnitude for the observed data.
    observed_peak = (
        synth["intervals"]
        .groupby(["business_date", "interval_idx"])["mips"]
        .sum()
        .groupby("business_date")
        .max()
        .quantile(0.95)
    )
    assert 0.5 * observed_peak < strict < 3.0 * observed_peak

    # 6. The serving table is publishable.
    summary = build_fy_summary(result, run_id="test")
    assert not summary.empty
    assert set(summary["reducer"]) == {"annual_max", "p95_of_daily_peaks"}
    assert (summary["peak_mips"] > 0).all()
