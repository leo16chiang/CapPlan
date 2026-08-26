"""R4HA, the table catalogue, and dotenv.

The R4HA tests matter more than most: it is the one target with an externally
fixed definition (IBM bills on it), so "close enough" is not a standard that
applies. The ring-buffer implementation is checked against a brute-force
rolling mean.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from capplan.data.tables import (
    CATALOGUE,
    DoubleCountError,
    Grain,
    TableRole,
    spec,
    validate_roles,
)
from capplan.envfile import load, parse
from capplan.model.forecast import ForecastCube
from capplan.sim.reducers import REDUCERS, PathSummary, TARGET_FAMILY, get_reducer, needs_r4ha
from capplan.sim.simulate import simulate


# -- R4HA ------------------------------------------------------------------


class FlatSampler:
    """Zero residuals: isolates the R4HA arithmetic from the dependence model."""

    def __init__(self, n_apps: int, n_int: int) -> None:
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


def hourly_cube(grid, values, apps=("A",), n_days=5):
    """A forecast cube whose median is exactly `values` for each app and day."""
    quantiles = np.array([0.1, 0.5, 0.9])
    n_int = len(values)
    q = np.zeros((len(apps), n_days, n_int, 3), dtype=np.float32)
    for i in range(3):
        q[:, :, :, i] = np.asarray(values, dtype=np.float32)
    days = grid.business_days(*grid.fiscal_year_bounds(2026))[:n_days]
    return ForecastCube(
        q=q, quantiles=quantiles, apps=list(apps), days=days, n_intervals=n_int
    )


def test_r4ha_matches_a_brute_force_rolling_mean(cfg):
    """The ring buffer must reproduce the definition exactly, not approximately."""
    from capplan.data.calendar import grid_from_config

    grid = grid_from_config(cfg.with_overrides({"calendar.interval_minutes": 60}))
    values = [10.0, 20.0, 60.0, 80.0, 30.0, 20.0, 10.0, 10.0, 10.0]
    cube = hourly_cube(grid, values, apps=("A",), n_days=3)

    result = simulate(
        cube, FlatSampler(1, len(values)), grid,
        n_paths=2, path_chunk=1, reducers=("annual_peak_r4ha",), progress_every=0,
    )
    assert result.r4ha_window_intervals == 4

    # Brute force over the concatenated horizon, which is what the ring buffer
    # sees -- the window legitimately spans the day boundary.
    series = np.array(values * 3)
    window = 4
    rolling = np.convolve(series, np.ones(window) / window, mode="valid")
    assert result.daily_r4ha.max() == pytest.approx(rolling.max(), rel=1e-5)


def test_r4ha_window_spans_the_day_boundary(cfg):
    """IBM's window does not stop at 17:00, so neither does this one."""
    from capplan.data.calendar import grid_from_config

    grid = grid_from_config(cfg.with_overrides({"calendar.interval_minutes": 60}))
    # Two high intervals at the end of the day and two at the start of the
    # next. Within either day alone the best 4-window contains two lows; only
    # a window crossing the boundary sees four highs together.
    values = [100.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 100.0]
    cube = hourly_cube(grid, values, apps=("A",), n_days=3)
    result = simulate(
        cube, FlatSampler(1, len(values)), grid,
        n_paths=1, path_chunk=1, reducers=("annual_peak_r4ha",), progress_every=0,
    )
    assert result.daily_r4ha.max() == pytest.approx(100.0, rel=1e-5)
    # Sanity: within a single day the best window is strictly worse, which is
    # what makes this test about the boundary rather than about arithmetic.
    within_day = np.convolve(np.array(values), np.ones(4) / 4, mode="valid").max()
    assert within_day < 100.0


def test_r4ha_is_never_above_the_interval_peak(cfg):
    """An average over four intervals cannot exceed the largest of them."""
    from capplan.data.calendar import grid_from_config

    grid = grid_from_config(cfg.with_overrides({"calendar.interval_minutes": 60}))
    rng = np.random.default_rng(0)
    values = rng.uniform(10, 100, 9).tolist()
    cube = hourly_cube(grid, values, apps=("A", "B"), n_days=6)
    result = simulate(
        cube, FlatSampler(2, 9), grid,
        n_paths=4, path_chunk=2, reducers=("annual_max", "annual_peak_r4ha"), progress_every=0,
    )
    assert (result.daily_r4ha <= result.daily_peaks + 1e-4).all()
    assert result.diagnostics["r4ha_to_interval_peak_ratio"] <= 1.0


def test_partial_windows_are_not_counted_as_four_hour_averages(cfg):
    """A 2-interval mean is not a 4-hour average and must not be reported as one."""
    from capplan.data.calendar import grid_from_config

    grid = grid_from_config(cfg.with_overrides({"calendar.interval_minutes": 60}))
    # A huge first interval then zeros: a partial window would report it high.
    values = [1000.0] + [0.0] * 8
    cube = hourly_cube(grid, values, apps=("A",), n_days=2)
    result = simulate(
        cube, FlatSampler(1, 9), grid,
        n_paths=1, path_chunk=1, reducers=("annual_peak_r4ha",), progress_every=0,
    )
    # Best genuine 4-window contains one 1000 and three 0s.
    assert result.daily_r4ha.max() == pytest.approx(250.0, rel=1e-5)


def test_r4ha_coincidence_is_measured_like_for_like(cfg):
    """REGRESSION: dividing R4HA-of-sum by the sum of app *interval* peaks folds
    smoothing into the coincidence figure and reads far worse than reality.
    The denominator must be the sum of each application's own R4HA."""
    from capplan.data.calendar import grid_from_config

    grid = grid_from_config(cfg.with_overrides({"calendar.interval_minutes": 60}))
    quantiles = np.array([0.1, 0.5, 0.9])
    n_apps, n_int, n_days = 2, 9, 6
    q = np.zeros((n_apps, n_days, n_int, 3), dtype=np.float32)
    a = np.zeros(n_int); a[1:5] = 100.0        # app A busy early
    b = np.zeros(n_int); b[5:9] = 100.0        # app B busy late -- disjoint
    for i in range(3):
        q[0, :, :, i] = a
        q[1, :, :, i] = b
    days = grid.business_days(*grid.fiscal_year_bounds(2026))[:n_days]
    cube = ForecastCube(q=q, quantiles=quantiles, apps=["A", "B"], days=days, n_intervals=n_int)

    result = simulate(
        cube, FlatSampler(n_apps, n_int), grid,
        n_paths=2, path_chunk=1,
        reducers=("annual_max", "annual_peak_r4ha"), progress_every=0,
    )
    stats = result.coincidence_by_target()
    assert result.daily_sum_app_r4ha is not None
    # Both factors are ratios of comparable quantities, so both are <= 1.
    assert 0.0 < stats["coincidence_interval_peak"] <= 1.0
    assert 0.0 < stats["coincidence_r4ha"] <= 1.0
    # Four-hour averaging smooths timing differences, so the R4HA factor is at
    # least as favourable. This is the finding that decides how much the
    # Stage 2 machinery is worth for a cost deliverable.
    assert stats["coincidence_r4ha"] >= stats["coincidence_interval_peak"] - 1e-6
    assert stats["coincidence_gain_from_r4ha"] >= -1e-6


def test_r4ha_reducers_refuse_to_guess_when_not_accumulated():
    path = PathSummary(
        daily_peaks=np.ones(10), daily_means=np.ones(10),
        days=[date(2026, 11, 2) + timedelta(days=i) for i in range(10)],
    )
    with pytest.raises(ValueError, match="rolling 4-hour average"):
        get_reducer("monthly_peak_r4ha")(path)


def test_needs_r4ha_gates_the_ring_buffer():
    assert not needs_r4ha(["annual_max", "p95_of_daily_peaks"])
    assert needs_r4ha(["annual_max", "monthly_peak_r4ha"])


def test_every_reducer_is_assigned_a_decision_family():
    """The pack prints the family, because 'which number' is a question about
    the decision being made, not about statistics."""
    for name in REDUCERS:
        assert name in TARGET_FAMILY, f"{name} has no target family"
    assert TARGET_FAMILY["monthly_peak_r4ha"] == "software_cost"
    assert TARGET_FAMILY["annual_max"] == "hardware"


def test_r4ha_accumulators_stay_within_budget(cfg):
    """Per-app R4HA adds (chunk, apps, window) floats, not (paths, days, apps)."""
    import tracemalloc

    from capplan.data.calendar import grid_from_config

    grid = grid_from_config(cfg.with_overrides({"calendar.interval_minutes": 60}))
    cube = hourly_cube(grid, list(np.linspace(10, 90, 9)), apps=tuple("ABCDEFGH"), n_days=120)
    tracemalloc.start()
    simulate(
        cube, FlatSampler(8, 9), grid, n_paths=400, path_chunk=100,
        reducers=("annual_max", "monthly_peak_r4ha"), progress_every=0,
    )
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 100e6, f"peak allocation {peak / 1e6:.0f}MB"


# -- table catalogue -------------------------------------------------------


def test_daily_tables_cannot_measure_coincidence():
    """Coincidence is defined by intra-day timing, which daily grain lacks.

    Not 'hard to measure' -- absent. Two daily maxima cannot be distinguished
    from two simultaneous ones.
    """
    assert not Grain.DAY.has_intraday_timing
    assert not Grain.MONTH.has_intraday_timing
    assert Grain.HOUR.has_intraday_timing
    assert Grain.INTERVAL.has_intraday_timing

    assert not CATALOGUE["MVS_ADDRSPACE_D"].can_measure_coincidence
    assert CATALOGUE["MVSPM_WORKLOAD2_HV"].can_measure_coincidence


def test_only_the_spine_may_feed_intervals():
    assert validate_roles({"intervals": "MVSPM_WORKLOAD2_HV"}) == []

    with pytest.raises(DoubleCountError, match="already counted"):
        validate_roles({"intervals": "CICS_TRANSACTIO_DP"})
    with pytest.raises(DoubleCountError, match="already counted"):
        validate_roles({"intervals": "MVS_ADDRSPACE_D"})


def test_a_validation_table_used_as_input_is_refused():
    """Feeding the site's own MIPS table in makes any later agreement circular."""
    with pytest.raises(DoubleCountError, match="circular"):
        validate_roles({"intervals": "CAP_GRP_MIPS_D"})


def test_two_spines_are_refused():
    from capplan.data.tables import TableSpec, _add

    _add(TableSpec(
        name="SECOND_SPINE_H", grain=Grain.HOUR, role=TableRole.SPINE,
        smf="test", measures="test",
    ))
    try:
        with pytest.raises(DoubleCountError, match="more than one spine"):
            validate_roles({"intervals": "MVSPM_WORKLOAD2_HV", "extra": "SECOND_SPINE_H"})
    finally:
        CATALOGUE.pop("SECOND_SPINE_H")


def test_a_daily_spine_warns_about_the_lost_diagnostic():
    from capplan.data.tables import TableSpec, _add

    _add(TableSpec(
        name="DAILY_SPINE_D", grain=Grain.DAY, role=TableRole.SPINE,
        smf="test", measures="test",
    ))
    try:
        warnings = validate_roles({"intervals": "DAILY_SPINE_D"})
        assert any("coincidence factor cannot be measured" in w for w in warnings)
    finally:
        CATALOGUE.pop("DAILY_SPINE_D")


def test_unknown_tables_warn_rather_than_pass_silently():
    warnings = validate_roles({"intervals": "SOMEONES_CUSTOM_VIEW"})
    assert any("not in the catalogue" in w for w in warnings)


def test_spec_lookup_tolerates_schema_prefixes():
    assert spec("SMFDB.MVSPM_WORKLOAD2_HV") is CATALOGUE["MVSPM_WORKLOAD2_HV"]
    assert spec("mvspm_workload2_hv") is CATALOGUE["MVSPM_WORKLOAD2_HV"]


# -- dotenv ----------------------------------------------------------------


def test_dotenv_parses_the_awkward_cases():
    parsed = parse(
        "\n".join([
            "PLAIN=value",
            'export QUOTED="a b"  # trailing comment',
            "INLINE=v1 # note",
            "HASH_IN_QUOTES='p#ss w0rd'",
            "BRACES={IBM DB2 ODBC DRIVER}",
            "EMPTY=",
            "# whole-line comment",
            "",
            "NO_EQUALS_SIGN",
        ])
    )
    assert parsed["PLAIN"] == "value"
    assert parsed["QUOTED"] == "a b", "a quoted value ends at its closing quote"
    assert parsed["INLINE"] == "v1"
    assert parsed["HASH_IN_QUOTES"] == "p#ss w0rd", "# inside quotes is not a comment"
    assert parsed["BRACES"] == "{IBM DB2 ODBC DRIVER}"
    assert parsed["EMPTY"] == ""
    assert "NO_EQUALS_SIGN" not in parsed


def test_real_environment_beats_the_dotenv_file(tmp_path, monkeypatch):
    """So a scheduled job or container injects credentials the usual way and
    a checked-out file never silently overrides production."""
    (tmp_path / ".env").write_text("CAPPLAN_TEST_VAR=from-file\n", encoding="utf-8")
    monkeypatch.setenv("CAPPLAN_TEST_VAR", "from-environment")
    load((".env",), root=tmp_path)
    import os

    assert os.environ["CAPPLAN_TEST_VAR"] == "from-environment"

    load((".env",), root=tmp_path, override=True)
    assert os.environ["CAPPLAN_TEST_VAR"] == "from-file"


def test_dotenv_local_wins_over_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("CAPPLAN_TEST_ORDER=base\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text("CAPPLAN_TEST_ORDER=local\n", encoding="utf-8")
    monkeypatch.delenv("CAPPLAN_TEST_ORDER", raising=False)
    load((".env", ".env.local"), root=tmp_path, override=True)
    import os

    assert os.environ["CAPPLAN_TEST_ORDER"] == "local"


def test_env_example_is_committed_and_holds_no_secret():
    from pathlib import Path

    example = Path(".env.example")
    assert example.exists(), ".env.example must be committed as the template"
    text = example.read_text(encoding="utf-8")
    # The password line must be present but empty.
    assert "CAPPLAN_DB2_PASSWORD=" in text
    for line in text.splitlines():
        if line.startswith("CAPPLAN_DB2_PASSWORD="):
            assert line.strip() == "CAPPLAN_DB2_PASSWORD=", "template must ship empty"


def test_dotenv_is_gitignored():
    from pathlib import Path

    ignored = Path(".gitignore").read_text(encoding="utf-8")
    assert "\n.env\n" in ignored
    assert "!.env.example" in ignored
