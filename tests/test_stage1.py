"""Stage 1: the marginals, and the three bugs that were found by measuring.

Each of the regression tests here corresponds to a defect that produced
plausible-looking output. That is the category worth guarding.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from capplan.data.calendar import grid_from_config
from capplan.model.backends import build_backend, neural_available
from capplan.model.features import (
    DesignMatrix,
    FeatureSpec,
    PanelIndex,
    anomaly_mask,
    build_design,
    calendar_features,
    damp_trend,
    is_month_end,
    to_cube,
)
from capplan.model.forecast import ForecastCube
from capplan.model.train import fit_stage1, forecast_horizon


@pytest.fixture(scope="module")
def fitted(synth, grid, cfg):
    return fit_stage1(synth["intervals"], grid, cfg)


def test_quantile_ridge_achieves_nominal_coverage():
    """Fixed-epsilon IRLS undercovered the tails; annealing fixed it."""
    rng = np.random.default_rng(0)
    n = 20_000
    X = rng.normal(size=(n, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0]) + rng.standard_t(4, n) * 0.5
    design = DesignMatrix(
        X=X, y=y,
        app_idx=np.zeros(n, int), day_idx=np.zeros(n, int), interval_idx=np.zeros(n, int),
        feature_names=list("abcd"), scales=np.array([1.0]),
        index=PanelIndex(["A"], [], 1), valid=np.ones(n, bool),
    )
    quantiles = [0.05, 0.1, 0.5, 0.9, 0.95]
    predicted = build_backend("quantile_ridge", quantiles).fit(design).predict(X)
    for i, tau in enumerate(quantiles):
        assert (y <= predicted[:, i]).mean() == pytest.approx(tau, abs=0.02)


def test_predicted_quantiles_never_cross():
    rng = np.random.default_rng(1)
    n = 5_000
    X = rng.normal(size=(n, 3))
    y = X[:, 0] + rng.normal(size=n)
    design = DesignMatrix(
        X=X, y=y,
        app_idx=np.zeros(n, int), day_idx=np.zeros(n, int), interval_idx=np.zeros(n, int),
        feature_names=list("abc"), scales=np.array([1.0]),
        index=PanelIndex(["A"], [], 1), valid=np.ones(n, bool),
    )
    predicted = build_backend("quantile_ridge", [0.1, 0.25, 0.5, 0.75, 0.9]).fit(design).predict(X)
    assert (np.diff(predicted, axis=1) >= -1e-9).all()


def test_month_end_is_derived_from_the_calendar_not_the_block(grid):
    """REGRESSION: positional derivation flagged every single-day forecast block
    as month-end, and every September day as quarter-end -- a 50% over-forecast
    for the whole of September."""
    # A single-day block must not look like month-end just because it is alone.
    assert is_month_end(grid, date(2026, 9, 15), 2) == 0.0
    assert is_month_end(grid, date(2026, 9, 30), 2) == 1.0

    spec = FeatureSpec()
    one_day, names = calendar_features([date(2026, 9, 15)], 36, grid, spec)
    month_end_col = names.index("month_end")
    quarter_end_col = names.index("quarter_end")
    assert one_day[0, 0, month_end_col] == 0.0
    assert one_day[0, 0, quarter_end_col] == 0.0

    # And a mid-block day must agree with the same day forecast alone.
    days = grid.business_days(date(2026, 9, 1), date(2026, 9, 30))
    block, _ = calendar_features(days, 36, grid, spec)
    pos = days.index(date(2026, 9, 15))
    assert block[pos, 0, month_end_col] == one_day[0, 0, month_end_col]


def test_trend_origin_is_pinned_at_fit_time(fitted):
    """REGRESSION: deriving `years_elapsed` from the block's own first day made
    it identically zero on every forecast row, switching the trend off exactly
    when it was the only thing carrying the forecast."""
    assert fitted.spec.origin == fitted.index.days[0]
    assert fitted.spec.train_years_max > 0.5

    spec = fitted.spec
    _, names = calendar_features(fitted.index.days[:3], 36, fitted.grid, spec)
    col = names.index("years_elapsed")
    future = [date(2027, 6, 1), date(2027, 6, 2)]
    block, _ = calendar_features(future, 36, fitted.grid, spec)
    assert block[0, 0, col] > spec.train_years_max


def test_trend_damping_flattens_extrapolation_only():
    years = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
    damped = damp_trend(years, train_years_max=2.0, phi=0.5)
    assert damped[:3] == pytest.approx([0.0, 1.0, 2.0])   # untouched in-sample
    assert damped[3] == pytest.approx(2.5)                # 1 year out -> 0.5
    assert damped[4] == pytest.approx(3.5)
    assert damp_trend(years, 2.0, 1.0) == pytest.approx(years)


def test_two_year_horizon_neither_explodes_nor_flatlines(fitted, grid):
    """REGRESSION: pure recursion on lag features compounded to 3.2x over two
    fiscal years; anchoring it flat then killed the trend entirely."""
    future = grid.horizon_business_days(max(fitted.index.days), 2)
    cube = forecast_horizon(fitted, future, progress_every=0)
    totals = cube.median.sum(axis=0).max(axis=1)
    ratio = totals[-1] / totals[0]
    assert 0.9 < ratio < 1.5, f"two-year growth of {ratio:.2f}x is not credible"

    # And the intraday shape must survive to the end of the horizon.
    first_day, last_day = cube.median[:, 0].sum(axis=0), cube.median[:, -1].sum(axis=0)
    assert last_day.max() / last_day.mean() == pytest.approx(
        first_day.max() / first_day.mean(), rel=0.25
    )


def test_fitted_growth_recovers_the_synthetic_rate(fitted, synth):
    """Growth is explicit and per-app because it is what a custodian argues with.

    Tested against each app's *actual* trajectory rather than against the
    population the rates were drawn from: with eight apps the sample median of
    the draws is nowhere near the distribution median, and asserting on it tests
    the random seed rather than the estimator.
    """
    truth = synth["truth"]                       # (apps, days, intervals), pre-anomaly
    n_days = truth.shape[1]
    years = np.arange(n_days) / 252.0
    realised = np.array(
        [np.polyfit(years, np.log(np.median(truth[a], axis=1)), 1)[0] for a in range(len(truth))]
    )
    fitted_rates = fitted.growth[: len(realised)]
    assert np.corrcoef(realised, fitted_rates)[0, 1] > 0.9
    assert np.abs(fitted_rates - realised).mean() < 0.03


def test_fitted_quantiles_return_nan_on_burn_in_rows(fitted):
    """REGRESSION: zero-filling burn-in features produced predictions with
    near-degenerate spreads; Stage 2 divides by that spread, and a handful of
    such cells put the simulated peak 23x too high."""
    from capplan.model.train import fitted_quantiles

    predicted = fitted_quantiles(fitted)
    invalid = ~fitted.design.valid.reshape(fitted.cube.shape)
    assert invalid.sum() > 0, "expected some burn-in rows in this fixture"
    assert np.isnan(predicted[invalid][:, 0]).all()
    assert np.isfinite(predicted[~invalid][:, 0]).all()


def test_spread_floor_is_relative_to_the_level():
    """REGRESSION: an absolute 1e-6 floor is no floor at all for a large app."""
    quantiles = np.array([0.1, 0.5, 0.9])
    q = np.zeros((1, 1, 1, 3), dtype=np.float32)
    q[0, 0, 0] = [2000.0, 2000.0, 2000.0]   # degenerate interval on a large app
    cube = ForecastCube(
        q=q, quantiles=quantiles, apps=["BIG"], days=[date(2026, 1, 2)], n_intervals=1
    )
    assert cube.spread(0.1, 0.9)[0, 0, 0] == pytest.approx(0.02 * 2000.0)


def test_neural_backend_reports_a_missing_wheel_clearly():
    """The week-1 proxy question must not surface as a traceback mid-run."""
    ok, message = neural_available()
    if ok:
        pytest.skip("torch is installed in this environment")
    assert "torch" in message and "mirror" in message
    from capplan.model.backends import NeuralUnavailable

    with pytest.raises(NeuralUnavailable):
        build_backend("neuralforecast", [0.5])


def test_anomalies_are_excluded_from_the_fit(synth, grid, cfg):
    cube, index = to_cube(synth["intervals"], grid)
    mask = anomaly_mask(synth["intervals"], index)
    assert mask.sum() > 0
    design = build_design(cube, index, grid, FeatureSpec(), exclude=mask)
    flagged = mask.reshape(-1)
    assert not design.valid[flagged].any()
