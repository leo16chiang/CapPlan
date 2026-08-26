"""Config, normalisation, calibration, scenarios, registry and serving."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from capplan.config import Config, ConfigError, load_config
from capplan.data.mips_normalisation import NormalisationTable, audit_summary, normalise
from capplan.model.calibrate import (
    apply_conformal,
    conformity_scores,
    empirical_coverage,
    fit_conformal,
    split_holdout,
)
from capplan.model.forecast import ForecastCube
from capplan.registry import Registry
from capplan.serve.scenario import Scenario, apply_scenario, compare


# -- config ---------------------------------------------------------------


def test_config_rejects_an_interval_that_does_not_divide_a_day(cfg):
    with pytest.raises(ConfigError, match="1440"):
        from capplan.config import validate

        validate(cfg.with_overrides({"calendar.interval_minutes": 7}))


def test_config_rejects_quantiles_without_a_median(cfg):
    from capplan.config import validate

    with pytest.raises(ConfigError, match="median"):
        validate(cfg.with_overrides({"model.quantiles": [0.1, 0.9]}))


def test_overrides_do_not_mutate_the_original(cfg):
    before = cfg.get("simulation.n_paths")
    other = cfg.with_overrides({"simulation.n_paths": 7})
    assert other.get("simulation.n_paths") == 7
    assert cfg.get("simulation.n_paths") == before


# -- normalisation --------------------------------------------------------


def test_capture_ratio_grosses_up_and_is_recorded_per_row():
    frame = pd.DataFrame({"lpar": ["PRDA", "PRDB"], "msu": [100.0, 100.0]})
    table = NormalisationTable(
        capture_ratio_default=1.0,
        capture_ratio_by_lpar={"PRDA": 0.8},
        mips_per_msu_default=6.0,
    )
    out = normalise(frame, table)
    assert out.loc[0, "mips"] == pytest.approx(100 * 6.0 / 0.8)
    assert out.loc[1, "mips"] == pytest.approx(600.0)
    # Recorded, so a disputed number is a lookup rather than an argument.
    assert list(out["capture_ratio"]) == [0.8, 1.0]
    assert not audit_summary(out).empty


def test_a_zero_capture_ratio_is_rejected_not_divided_by():
    frame = pd.DataFrame({"lpar": ["PRDA"], "msu": [100.0]})
    table = NormalisationTable(capture_ratio_by_lpar={"PRDA": 0.0})
    with pytest.raises(ValueError, match="positive"):
        normalise(frame, table)


def test_a_preconverted_mips_column_is_not_converted_twice():
    frame = pd.DataFrame({"lpar": ["PRDA"], "msu": [100.0], "mips": [1234.0]})
    out = normalise(frame, NormalisationTable(mips_per_msu_default=6.0))
    assert out.loc[0, "mips"] == 1234.0


# -- calibration ----------------------------------------------------------


def test_holdout_split_is_chronological():
    days = [date(2026, 1, 1 + i) for i in range(10)]
    train, holdout = split_holdout(days, 3)
    assert train[-1] < holdout[0]
    assert len(holdout) == 3
    with pytest.raises(ValueError):
        split_holdout(days, 20)


def test_conformity_score_is_signed():
    y = np.array([5.0, 15.0, 25.0])
    lo, hi = np.array([10.0] * 3), np.array([20.0] * 3)
    scores = conformity_scores(y, lo, hi)
    assert scores[0] == 5.0     # below the interval
    assert scores[1] == -5.0    # comfortably inside -> allows narrowing
    assert scores[2] == 5.0     # above the interval


def _shifted_forecast(shift: float, sd: float = 10.0, seed: int = 0, n_days: int = 60):
    """Truth ~ N(100, 10); the forecast has the right shape but the wrong centre.

    Built from a *distribution*, not from the truth itself: predicted quantiles
    constructed as truth-plus-a-constant collapse onto the observation once
    conformal correction is applied, which tests arithmetic rather than
    calibration.
    """
    from scipy import stats

    rng = np.random.default_rng(seed)
    quantiles = np.array([0.1, 0.5, 0.9])
    n_int = 36
    truth = rng.normal(100.0, sd, (1, n_days, n_int))
    levels = shift + sd * stats.norm.ppf(quantiles)
    predicted = np.broadcast_to(levels, truth.shape + (len(quantiles),)).copy()
    return truth, predicted, quantiles, n_days, n_int


def test_per_quantile_conformal_corrects_a_location_bias():
    """Symmetric CQR corrects width but not location. A long-horizon forecast
    almost always carries some location bias, so the default has to fix it."""
    truth, predicted, quantiles, n_days, n_int = _shifted_forecast(shift=120.0)
    before = empirical_coverage(truth, predicted, quantiles)
    assert before[1] > 0.9, "the median should be badly over-covering to start with"

    adjustment = fit_conformal(
        truth, predicted, quantiles, ["A"], per_app=True, min_scores_per_app=50,
        method="per_quantile",
    )
    cube = ForecastCube(
        q=predicted.astype(np.float32), quantiles=quantiles, apps=["A"],
        days=[date(2026, 1, 1)] * n_days, n_intervals=n_int,
    )
    after = empirical_coverage(truth, apply_conformal(cube, adjustment).q.astype(float), quantiles)
    assert np.abs(after - quantiles).max() < 0.03
    assert cube.calibrated


def test_symmetric_cqr_widens_but_cannot_move_the_centre():
    """Documents the limitation that motivates the per-quantile default."""
    truth, predicted, quantiles, n_days, n_int = _shifted_forecast(shift=120.0, seed=1)
    adjustment = fit_conformal(
        truth, predicted, quantiles, ["A"], per_app=True, min_scores_per_app=50, method="cqr"
    )
    # The median column is untouched by construction: no location correction.
    assert adjustment.per_app[0, 1] == 0.0

    cube = ForecastCube(
        q=predicted.astype(np.float32), quantiles=quantiles, apps=["A"],
        days=[date(2026, 1, 1)] * n_days, n_intervals=n_int,
    )
    after = empirical_coverage(truth, apply_conformal(cube, adjustment).q.astype(float), quantiles)
    # Still badly off-centre: widening a mis-centred interval does not centre it.
    assert after[1] > 0.9


def test_conformal_falls_back_to_pooled_for_thin_apps():
    quantiles = np.array([0.1, 0.5, 0.9])
    truth = np.random.default_rng(2).normal(100.0, 10.0, (2, 5, 4))
    predicted = np.stack([truth - 10, truth, truth + 10], axis=-1)
    adjustment = fit_conformal(
        truth, predicted, quantiles, ["A", "B"], per_app=True, min_scores_per_app=10_000
    )
    assert adjustment.used_pooled.all()


# -- scenarios ------------------------------------------------------------


def make_cube(apps=("A", "B"), n_days=100):
    quantiles = np.array([0.1, 0.5, 0.9])
    days = [date(2026, 1, 1) for _ in range(n_days)]
    days = pd.date_range("2026-01-01", periods=n_days, freq="B").date.tolist()
    q = np.ones((len(apps), n_days, 4, 3), dtype=np.float32) * 100.0
    return ForecastCube(q=q, quantiles=quantiles, apps=list(apps), days=days, n_intervals=4)


def test_level_multiplier_scales_only_the_named_app():
    cube = make_cube()
    out = apply_scenario(cube, Scenario(name="s", level_multiplier={"A": 1.3}))
    assert out.q[0].mean() == pytest.approx(130.0)
    assert out.q[1].mean() == pytest.approx(100.0)
    assert cube.q[0].mean() == pytest.approx(100.0), "the original must not be mutated"


def test_step_change_applies_only_from_the_stated_date():
    cube = make_cube(n_days=100)
    when = cube.days[50]
    out = apply_scenario(cube, Scenario(name="s", step_change={"A": (when, 2.0)}))
    assert out.q[0, 49].mean() == pytest.approx(100.0)
    assert out.q[0, 50].mean() == pytest.approx(200.0)


def test_growth_override_rebases_rather_than_compounds():
    cube = make_cube(n_days=260)
    out = apply_scenario(
        cube,
        Scenario(name="s", growth_override={"A": 0.10}),
        fitted_growth={"A": 0.02},
        anchor=cube.days[0],
    )
    years = (cube.days[-1] - cube.days[0]).days / 365.25
    assert out.q[0, -1].mean() == pytest.approx(100.0 * np.exp(0.08 * years), rel=1e-3)


def test_a_typo_in_an_app_id_is_loud():
    """A silently-ignored override is how a scenario gets presented as applied
    when it was not."""
    with pytest.raises(KeyError, match="unknown apps"):
        apply_scenario(make_cube(), Scenario(name="s", level_multiplier={"NOPE": 2.0}))


def test_scenario_describes_itself_with_attribution():
    text = Scenario(
        name="payments migration",
        requested_by="A. Custodian",
        rationale="Q2 cutover",
        level_multiplier={"A": 1.4},
    ).describe()
    assert "A. Custodian" in text and "1.40x" in text


def test_compare_reports_deltas():
    out = compare(np.full(100, 1000.0), np.full(100, 1200.0))
    assert out["delta_p50_pct"] == pytest.approx(20.0)


# -- registry -------------------------------------------------------------


def test_registry_round_trips_and_verifies(tmp_path, cfg):
    registry = Registry(tmp_path)
    run = registry.new_run("train", config=cfg, tags=["unit"])
    run.record_metrics({"pinball": 0.1})
    run.write_json("thing.json", {"a": 1})
    run.finalise()

    resolved = registry.resolve("train")
    assert resolved.run_id == run.run_id
    assert resolved.manifest["metrics"]["pinball"] == 0.1
    assert resolved.manifest["config"]["calendar"]["interval_minutes"] == cfg.get(
        "calendar.interval_minutes"
    ), "the manifest must record the grain the run actually used"
    assert registry.verify("train") == []

    (run.dir / "thing.json").write_text("tampered", encoding="utf-8")
    assert "hash mismatch: thing.json" in registry.verify("train")


def test_registry_verifies_with_a_relative_root(tmp_path, monkeypatch, cfg):
    """REGRESSION: with the default relative root (`artefacts/`), artefact paths
    were stored un-relativised, so `verify` looked under
    artefacts/<kind>/<id>/artefacts/<kind>/<id>/... and reported every artefact
    missing. The original test used tmp_path, which is absolute, and missed it.
    """
    monkeypatch.chdir(tmp_path)
    registry = Registry("artefacts")
    run = registry.new_run("simulate", config=cfg)
    np.savez_compressed(run.path("simulation.npz"), a=np.arange(3))
    run.add_artefact(run.dir / "simulation.npz", role="simulation")
    run.write_json("summary.json", {"ok": True})
    run.finalise()

    entry = registry.resolve("simulate").manifest["artefacts"][0]
    assert entry["path"] == "simulation.npz", "manifest paths must be run-relative"
    assert registry.verify("simulate") == []


def test_artefacts_outside_the_run_directory_are_rejected(tmp_path, cfg):
    registry = Registry(tmp_path / "artefacts")
    run = registry.new_run("train", config=cfg)
    stray = tmp_path / "elsewhere.json"
    stray.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not inside run directory"):
        run.add_artefact(stray, role="stray")


def test_resolving_a_missing_run_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        Registry(tmp_path).resolve("train")


# -- serving --------------------------------------------------------------


def test_published_app_peaks_carry_a_do_not_sum_flag(tmp_path):
    from capplan.serve.forecast_store import build_app_peaks
    from capplan.sim.simulate import SimulationResult

    result = SimulationResult(
        daily_peaks=np.ones((10, 5), dtype=np.float32),
        daily_means=np.ones((10, 5), dtype=np.float32),
        daily_sum_app_peaks=np.ones((10, 5), dtype=np.float32),
        app_peaks=np.ones((10, 2), dtype=np.float32),
        days=[date(2026, 1, 1)] * 5,
        apps=["A", "B"],
        fiscal_years=np.full(5, 2026),
        dependence="block_bootstrap",
        n_paths=10,
    )
    frame = build_app_peaks(result, run_id="r")
    assert frame["do_not_sum"].all()
    assert "different intervals" in frame["note"].iloc[0]
