"""Peaks do not sum. This is the claim the whole architecture rests on, so it
gets tested from three directions: in the data, in the SQL diagnostic, and in
the simulator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from capplan.diagnostics.runner import connect, run_coincidence, verdict


@pytest.fixture(scope="module")
def lake(tmp_path_factory, synth, cfg, grid):
    """Write the synthetic panel to a temporary lake for DuckDB to read."""
    from capplan.data.ingest import write_lake

    root = tmp_path_factory.mktemp("lake")
    write_lake(
        {
            "intervals": synth["intervals"],
            "events": synth["events"],
            "submissions": synth["submissions"],
            "lpar_totals": synth["lpar_totals"],
        },
        root,
    )
    return root


def test_sum_of_app_peaks_exceeds_the_realised_lpar_peak(lake):
    """The 32% error the architecture exists to remove, measured in SQL."""
    con = connect(lake)
    try:
        result = run_coincidence(con)
    finally:
        con.close()
    daily = result["coincidence_daily"]
    assert not daily.empty
    assert (daily["sum_app_peaks"] >= daily["lpar_peak"]).mean() > 0.95
    assert 0.5 < result.headline["coincidence_mean"] < 0.95
    assert result.headline["mean_overstatement_pct"] > 10


def test_apps_peak_in_different_intervals(lake):
    """The mechanism. If they all peaked together, coincidence would be ~1."""
    con = connect(lake)
    try:
        result = run_coincidence(con)
    finally:
        con.close()
    assert result["coincidence_summary"]["mean_distinct_peak_intervals"].min() > 1.5


def test_unattributed_load_is_flagged_not_hidden(lake):
    """A coincidence above 1 is a scoping finding, never something to clip."""
    con = connect(lake)
    try:
        result = run_coincidence(con, exclude_anomalies=True)
    finally:
        con.close()
    daily = result["coincidence_daily"]
    # Anomaly days are dropped from the app side but remain in SMF 70-1, so
    # some days must show up as unattributed rather than silently absorbed.
    assert "unattributed" in daily.columns
    assert daily["unattributed"].sum() > 0
    assert result.headline["n_unattributed_days"] == int(daily["unattributed"].sum())


def test_verdict_recommends_stopping_when_peaks_effectively_do_sum():
    """The diagnostic has to be able to say 'do not build this'."""
    call = verdict({"coincidence.coincidence_mean": 0.99})
    assert call.startswith("STOP AND RECONSIDER")
    call = verdict({"coincidence.coincidence_mean": 0.75, "coincidence.mean_overstatement_pct": 33})
    assert call.startswith("PROCEED")
    assert verdict({}).startswith("INCONCLUSIVE")


def test_simulator_takes_the_peak_of_the_sum_not_the_sum_of_peaks(synth, grid, cfg):
    """The transposition that would silently produce the wrong answer.

    Constructed so the two differ by a large, known amount: two apps with
    disjoint peak intervals. Peak of the sum must be far below the sum of peaks.
    """
    from capplan.model.forecast import ForecastCube
    from capplan.sim.simulate import simulate

    n_apps, n_days, n_int, quantiles = 2, 10, 36, np.array([0.1, 0.5, 0.9])
    q = np.zeros((n_apps, n_days, n_int, len(quantiles)), dtype=np.float32)
    base = np.full(n_int, 10.0)
    app0, app1 = base.copy(), base.copy()
    app0[5] = 100.0     # app 0 peaks in interval 5
    app1[25] = 100.0    # app 1 peaks in interval 25 -- never together
    for i in range(len(quantiles)):
        q[0, :, :, i] = app0
        q[1, :, :, i] = app1
    cube = ForecastCube(
        q=q, quantiles=quantiles, apps=["A", "B"],
        days=grid.business_days(*grid.fiscal_year_bounds(2026))[:n_days],
        n_intervals=n_int,
    )

    class ZeroSampler:
        """No residuals: isolates the reduction from the dependence model."""

        def stream(self, size, n_days, rng):
            class _S:
                def day(self, d):
                    return np.zeros((size, n_apps, n_int))

                def nbytes(self):
                    return 0

            return _S()

        def diagnostics(self):
            return {}

    result = simulate(
        cube, ZeroSampler(), grid, n_paths=4, path_chunk=2,
        reducers=("annual_max",), progress_every=0,
    )
    # Peak of the sum: 100 + 10 = 110. Sum of peaks would be 200.
    assert result.daily_peaks.max() == pytest.approx(110.0, rel=1e-4)
    assert result.daily_sum_app_peaks.max() == pytest.approx(200.0, rel=1e-4)
    assert result.coincidence()["simulated_coincidence_daily_mean"] == pytest.approx(0.55, rel=1e-3)
