"""Stages 2 and 3: dependence, memory discipline, and the reduction order."""

from __future__ import annotations

import numpy as np
import pytest

from capplan.model.features import anomaly_mask
from capplan.model.train import fit_stage1, fitted_quantiles, forecast_horizon
from capplan.sim.bootstrap import BlockBootstrap
from capplan.sim.copula import fit_gaussian_copula, tail_dependence_check
from capplan.sim.residuals import compute_residuals, cross_app_correlation, intraday_autocorrelation
from capplan.sim.simulate import simulate


@pytest.fixture(scope="module")
def fitted(synth, grid, cfg):
    return fit_stage1(synth["intervals"], grid, cfg)


@pytest.fixture(scope="module")
def panel(fitted, synth):
    return compute_residuals(
        observed=fitted.cube,
        predicted_q=fitted_quantiles(fitted),
        quantiles=fitted.quantiles,
        days=fitted.index.days,
        apps=fitted.index.apps,
        anomaly=anomaly_mask(synth["intervals"], fitted.index),
        scaling="spread",
    )


def test_residuals_are_standardised_and_free_of_pathological_cells(panel):
    """REGRESSION: an absolute spread floor gave sd 13.5 against a p99 of 3."""
    stats = panel.summary()
    assert abs(stats["residual_mean"]) < 0.25
    assert 0.4 < stats["residual_sd"] < 2.0
    assert stats["residual_p99"] < 6.0


def test_only_clean_complete_days_enter_the_pool(panel, fitted, synth):
    """A day is resamplable only if it is clean across every app -- that is the
    price of preserving cross-app coincidence."""
    dirty = set(synth["intervals"].loc[synth["intervals"]["is_anomaly"], "business_date"])
    for i, day in enumerate(panel.days):
        if day in dirty:
            assert not panel.usable_days[i], f"{day} carries an anomaly but is in the pool"
    assert panel.n_usable > 100


def test_dependence_is_real_in_both_directions(panel):
    """If either were zero, i.i.d. resampling would be defensible. Neither is."""
    corr = cross_app_correlation(panel)
    off_diagonal = corr[~np.eye(len(corr), dtype=bool)]
    assert np.nanmean(off_diagonal) > 0.1

    # The structural claim, not a magic number: residual autocorrelation is
    # positive at lag 1 and decays with lag. Both must hold for aligned blocks
    # to be earning their keep over i.i.d. resampling.
    acf = intraday_autocorrelation(panel, max_lag=6)
    assert acf[0] == pytest.approx(1.0)
    assert acf[1] > 0.1, "load excursions last longer than one interval"
    assert acf[1] > acf[4], "autocorrelation should decay with lag"


def test_block_bootstrap_preserves_whole_days(panel):
    """The draw must be a real historical day, not a per-app recombination."""
    sampler = BlockBootstrap.from_panel(panel, block_days=1, seed=3)
    rng = np.random.default_rng(0)
    indices = sampler.draw_indices(20, rng)
    drawn = sampler.draw(20, np.random.default_rng(0))
    for step, idx in enumerate(indices):
        assert np.allclose(drawn[step], sampler.pool[idx])


def test_bootstrap_refuses_an_empty_pool(panel):
    with pytest.raises(ValueError, match="empty"):
        BlockBootstrap(pool=np.zeros((0, 3, 36)))


def test_multi_day_blocks_keep_consecutive_days_together(panel):
    sampler = BlockBootstrap.from_panel(panel, block_days=5, seed=3)
    indices = sampler.draw_indices(20, np.random.default_rng(0))
    # Within each block the indices must be consecutive.
    for start in range(0, 20, 5):
        block = indices[start : start + 5]
        assert np.all(np.diff(block) == 1) or block.max() == sampler.n_days - 1


def test_copula_reports_what_its_assumption_costs(panel):
    """Zero tail dependence is the Gaussian copula's known weakness. It has to
    be measured, not asserted away."""
    report = tail_dependence_check(panel, threshold=0.9)
    assert 0.0 <= report["observed_tail_dependence"] <= 1.0
    assert np.isfinite(report["gaussian_implied_tail_dependence"])


def test_both_dependence_models_recover_the_historical_coincidence(fitted, panel, grid, synth):
    """The mechanism check. If this fails, the forward peak is wrong in the
    same direction and nothing downstream matters."""
    import pandas as pd

    frame = synth["intervals"]
    clean = frame[~frame["business_date"].isin(
        set(frame.loc[frame["is_anomaly"], "business_date"])
    )]
    peak_of_sum = (
        clean.groupby(["business_date", "interval_idx"])["mips"].sum().groupby("business_date").max()
    )
    sum_of_peaks = (
        clean.groupby(["business_date", "app_id"])["mips"].max().groupby("business_date").sum()
    )
    historical = float((peak_of_sum / sum_of_peaks).mean())

    future = grid.horizon_business_days(max(fitted.index.days), 1)[:60]
    cube = forecast_horizon(fitted, future, progress_every=0)
    for sampler in (
        BlockBootstrap.from_panel(panel, seed=5),
        fit_gaussian_copula(panel, shrinkage=0.1, seed=5),
    ):
        result = simulate(
            cube, sampler, grid, n_paths=100, path_chunk=50,
            reducers=("annual_max",), progress_every=0,
        )
        simulated = result.coincidence()["simulated_coincidence_daily_mean"]
        assert simulated == pytest.approx(historical, abs=0.06), (
            f"{type(sampler).__name__} gave {simulated:.3f} against {historical:.3f}"
        )


def test_simulation_stays_within_its_memory_budget(fitted, panel, grid):
    """The whole shape of the simulator exists to keep this true."""
    import tracemalloc

    future = grid.horizon_business_days(max(fitted.index.days), 1)[:120]
    cube = forecast_horizon(fitted, future, progress_every=0)
    sampler = BlockBootstrap.from_panel(panel, seed=7)

    tracemalloc.start()
    result = simulate(
        cube, sampler, grid, n_paths=400, path_chunk=100,
        reducers=("annual_max",), progress_every=0,
    )
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_apps, n_days, n_int = cube.q.shape[:3]
    naive_bytes = 400 * n_days * n_int * n_apps * 4
    assert peak < 200e6, f"peak allocation {peak / 1e6:.0f}MB"
    assert peak < naive_bytes / 10, "not meaningfully better than materialising everything"
    assert result.diagnostics["peak_chunk_working_mb"] < 50


def test_reducers_run_over_every_path(fitted, panel, grid):
    future = grid.horizon_business_days(max(fitted.index.days), 1)[:60]
    cube = forecast_horizon(fitted, future, progress_every=0)
    result = simulate(
        cube, BlockBootstrap.from_panel(panel, seed=9), grid,
        n_paths=50, path_chunk=25,
        reducers=("annual_max", "p95_of_daily_peaks"), progress_every=0,
    )
    for name in ("annual_max", "p95_of_daily_peaks"):
        assert result.reducer_values[name].shape == (50,)
        assert np.isfinite(result.reducer_values[name]).all()
    assert (
        result.reducer_values["annual_max"] >= result.reducer_values["p95_of_daily_peaks"]
    ).all()
