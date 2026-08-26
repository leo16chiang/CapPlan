"""Daily-grain forecasting and the transferred coincidence factor.

The question this answers: if most of the available tables are daily, can the
forecast run on daily data? It can, but only with a coincidence factor measured
on sub-daily data -- and the tests here pin down both halves of that: what goes
wrong without one, and how close it gets with one.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from capplan.data.synth import SynthSpec, generate
from capplan.model.forecast import ForecastCube
from capplan.sim.coincidence_factor import (
    CoincidenceFactor,
    estimate_from_intervals,
    load,
    save,
)
from capplan.sim.simulate import simulate


@pytest.fixture(scope="module")
def hourly(grid):
    synth = generate(
        grid,
        SynthSpec(n_apps=12, start=date(2023, 11, 1), end=date(2025, 10, 31), seed=9),
    )
    frame = synth["intervals"]
    dirty = set(frame.loc[frame["is_anomaly"], "business_date"].unique())
    return frame[~frame["business_date"].isin(dirty)]


class ZeroSampler:
    def __init__(self, n_apps: int, n_int: int = 1) -> None:
        self.n_apps, self.n_int = n_apps, n_int

    def stream(self, size, n_days, rng):
        outer = self

        class _S:
            def day(self, d):
                return np.zeros((size, outer.n_apps, outer.n_int))

            def nbytes(self):
                return 0

        return _S()

    def diagnostics(self):
        return {}


def daily_cube(frame, days, grid):
    """A degenerate one-interval cube: each app's daily peak."""
    apps = sorted(frame["app_id"].unique())
    pivot = (
        frame[frame["business_date"].isin(days)]
        .groupby(["app_id", "business_date"])["mips"]
        .max()
        .unstack("business_date")
        .reindex(index=apps, columns=days)
    )
    q = np.repeat(pivot.to_numpy()[:, :, None, None], 3, axis=3).astype(np.float32)
    return ForecastCube(
        q=q, quantiles=np.array([0.1, 0.5, 0.9]), apps=apps, days=list(days), n_intervals=1
    )


# -- what daily data cannot do on its own -----------------------------------


def test_daily_grain_alone_reduces_to_the_sum_of_app_peaks(hourly, grid):
    """With one interval a day, 'peak of the sum' IS 'sum of the peaks'.

    That is the whole error the architecture exists to remove, so a daily-only
    forecast reproduces it exactly rather than approximately.
    """
    days = sorted(hourly["business_date"].unique())[-60:]
    cube = daily_cube(hourly, days, grid)
    result = simulate(
        cube, ZeroSampler(len(cube.apps)), grid,
        n_paths=20, path_chunk=10, reducers=("mean_of_daily_peaks",), progress_every=0,
    )
    expected = (
        hourly[hourly["business_date"].isin(days)]
        .groupby(["business_date", "app_id"])["mips"].max()
        .groupby("business_date").sum()
    )
    assert result.daily_peaks.mean() == pytest.approx(expected.mean(), rel=1e-3)

    realised = (
        hourly[hourly["business_date"].isin(days)]
        .groupby(["business_date", "interval_idx"])["mips"].sum()
        .groupby("business_date").max()
    )
    # Materially high, and in the direction that buys hardware nobody needs.
    assert result.daily_peaks.mean() / realised.mean() > 1.15


def test_a_transferred_factor_recovers_the_realised_peak(hourly, grid):
    """A factor from a SHORT sub-daily sample, applied to a daily forecast.

    The sample is the first three months; the forecast covers the last year, so
    the two do not overlap and this measures transfer rather than fit.
    """
    all_days = sorted(hourly["business_date"].unique())
    sample_days, holdout = all_days[:63], all_days[-250:]
    assert not set(sample_days) & set(holdout)

    factor = estimate_from_intervals(
        hourly[hourly["business_date"].isin(sample_days)], grid
    )
    cube = daily_cube(hourly, holdout, grid)
    result = simulate(
        cube, ZeroSampler(len(cube.apps)), grid,
        n_paths=200, path_chunk=100, reducers=("mean_of_daily_peaks",),
        coincidence_factor=factor, progress_every=0,
    )
    realised = (
        hourly[hourly["business_date"].isin(holdout)]
        .groupby(["business_date", "interval_idx"])["mips"].sum()
        .groupby("business_date").max()
    )
    error = result.daily_peaks.mean() / realised.mean() - 1.0
    assert abs(error) < 0.05, f"transferred factor left a {100 * error:+.1f}% error"
    assert result.diagnostics["coincidence_factor_applied"]


def test_the_factor_is_resampled_not_applied_as_a_constant(hourly, grid):
    """Collapsing it to its mean would understate the forecast's spread by
    exactly the factor's own day-to-day variation."""
    days = sorted(hourly["business_date"].unique())[-120:]
    factor = estimate_from_intervals(hourly, grid)
    cube = daily_cube(hourly, days, grid)

    varying = simulate(
        cube, ZeroSampler(len(cube.apps)), grid, n_paths=300, path_chunk=100,
        reducers=("mean_of_daily_peaks",), coincidence_factor=factor, progress_every=0,
    )
    constant = CoincidenceFactor(
        samples=np.full(200, factor.samples.mean()),
        source_days=factor.source_days, grain_minutes=factor.grain_minutes,
        n_apps=factor.n_apps,
    )
    flat = simulate(
        cube, ZeroSampler(len(cube.apps)), grid, n_paths=300, path_chunk=100,
        reducers=("mean_of_daily_peaks",), coincidence_factor=constant, progress_every=0,
    )
    # Compare spread ACROSS PATHS within each day. Overall std is dominated by
    # day-to-day level variation, which both runs share, so it would hide the
    # thing being tested.
    # Relative to the level, not absolute: these are MIPS in the thousands, so
    # float32 rounding alone leaves a non-zero absolute standard deviation.
    level = float(varying.daily_peaks.mean())
    varying_spread = float(varying.daily_peaks.std(axis=0).mean()) / level
    flat_spread = float(flat.daily_peaks.std(axis=0).mean()) / level

    assert flat_spread < 1e-5, (
        "a constant factor with a zero-residual sampler leaves no path-to-path spread"
    )
    assert varying_spread > 0.01, "resampling the factor must produce real spread"
    # Four orders of magnitude apart, so this is not a threshold-tuning exercise.
    assert varying_spread > 1000 * flat_spread


def test_applying_a_factor_at_sub_daily_grain_is_refused(hourly, grid):
    """The coincidence is already in the data; applying it again discounts twice."""
    days = sorted(hourly["business_date"].unique())[-10:]
    apps = sorted(hourly["app_id"].unique())
    q = np.ones((len(apps), len(days), grid.intervals_per_day, 3), dtype=np.float32)
    cube = ForecastCube(
        q=q, quantiles=np.array([0.1, 0.5, 0.9]), apps=apps, days=list(days),
        n_intervals=grid.intervals_per_day,
    )
    factor = estimate_from_intervals(hourly, grid)
    with pytest.raises(ValueError, match="already represented"):
        simulate(
            cube, ZeroSampler(len(apps), grid.intervals_per_day), grid,
            n_paths=10, path_chunk=5, reducers=("annual_max",),
            coincidence_factor=factor, progress_every=0,
        )


# -- how much sample is enough ----------------------------------------------


def test_sample_adequacy_distinguishes_thin_from_sufficient(hourly, grid):
    """The number that says how much hourly data to go and get."""
    all_days = sorted(hourly["business_date"].unique())
    thin = estimate_from_intervals(
        hourly[hourly["business_date"].isin(all_days[:12])], grid
    ).sample_adequacy()
    assert thin["verdict"].startswith("TOO THIN")

    ample = estimate_from_intervals(
        hourly[hourly["business_date"].isin(all_days[:250])], grid
    ).sample_adequacy()
    assert ample["mean_ci_width_pct"] < thin["mean_ci_width_pct"]
    assert not ample["verdict"].startswith("TOO THIN")


def test_hourly_sample_is_flagged_as_an_upper_bound(hourly, grid, caplog):
    """Two applications peaking 20 minutes apart look simultaneous at hourly
    grain, so an hourly factor understates the true overstatement."""
    import logging

    with caplog.at_level(logging.WARNING):
        estimate_from_intervals(hourly, grid)
    assert any("UPPER BOUND" in r.message for r in caplog.records)


def test_weekday_conditioning_falls_back_when_thin(hourly, grid):
    factor = estimate_from_intervals(hourly, grid)
    rng = np.random.default_rng(0)
    drawn = factor.draw_for_weekday(0, 500, rng)
    assert drawn.size == 500
    assert np.isfinite(drawn).all()

    sparse = CoincidenceFactor(
        samples=np.array([0.8, 0.82, 0.79]),
        source_days=3, grain_minutes=60, n_apps=5,
        by_weekday={0: np.array([0.5])},          # one observation: must not be used
    )
    assert set(np.unique(sparse.draw_for_weekday(0, 200, rng))) <= {0.8, 0.82, 0.79}


def test_factor_round_trips_through_disk(hourly, grid, tmp_path):
    factor = estimate_from_intervals(hourly, grid)
    path = tmp_path / "factor.npz"
    save(factor, path)
    restored = load(path)
    assert restored.summary() == factor.summary()
    assert set(restored.by_weekday) == set(factor.by_weekday)


def test_an_empty_sample_is_refused_not_defaulted():
    with pytest.raises(ValueError, match="no usable coincidence samples"):
        CoincidenceFactor(samples=np.array([np.nan, np.inf]), source_days=0,
                          grain_minutes=60, n_apps=1)
