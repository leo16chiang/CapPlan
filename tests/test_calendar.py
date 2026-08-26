"""The calendar carries the whole architecture. If prime time is wrong,
everything downstream is wrong in a way no other test would catch.
"""

from datetime import date

import pytest

from capplan.data.calendar import PrimeTimeGrid, prime_time_frame


@pytest.mark.parametrize(
    "minutes, per_day, last",
    [(15, 36, "16:45"), (30, 18, "16:30"), (60, 9, "16:00")],
)
def test_prime_day_divides_correctly_at_every_grain(cfg, minutes, per_day, last):
    """Grain is a configuration question, not a constant.

    IZPCA's aggregation interval (MVSPM_TIME_RES) defaults to an hour and
    cannot be finer than the SMF interval, so 15 is an aspiration and 60 is the
    common case. The row count that justifies a neural Stage 1 moves with it:
    36 intervals/day gives ~945k rows over three years, 9 gives ~236k.
    """
    from capplan.data.calendar import grid_from_config

    g = grid_from_config(cfg.with_overrides({"calendar.interval_minutes": minutes}))
    assert g.intervals_per_day == per_day
    assert g.interval_starts[0].hour == 8
    assert g.interval_starts[-1].strftime("%H:%M") == last


def test_shipped_default_matches_the_izpca_hourly_grain(grid):
    """The default tracks what MVSPM_WORKLOAD2_HV actually delivers."""
    assert grid.interval_minutes == 60
    assert grid.intervals_per_day == 9


def test_fiscal_year_runs_november_to_october(grid):
    assert grid.fiscal_year(date(2025, 10, 31)) == 2025
    assert grid.fiscal_year(date(2025, 11, 1)) == 2026
    assert grid.fiscal_year(date(2026, 10, 31)) == 2026
    assert grid.fiscal_year_bounds(2026) == (date(2025, 11, 1), date(2026, 10, 31))


def test_fiscal_year_has_roughly_250_business_days(grid):
    assert 245 <= len(grid.fiscal_year_business_days(2026)) <= 262


def test_two_fiscal_year_horizon_reaches_the_end_of_the_second(grid):
    """Two fiscal years out is not 2 x 250 days -- it runs to the FY boundary."""
    horizon = grid.horizon_business_days(date(2025, 10, 31), 2)
    assert horizon[0] > date(2025, 10, 31)
    assert horizon[-1].month == 10 and horizon[-1].year == 2027
    assert 490 <= len(horizon) <= 525


def test_weekends_and_holidays_are_excluded(grid):
    assert not grid.is_business_day(date(2026, 1, 3))   # Saturday
    assert not grid.is_business_day(date(2026, 1, 1))   # configured holiday
    assert grid.is_business_day(date(2026, 1, 2))


def test_off_prime_timestamps_are_rejected(cfg):
    from datetime import datetime

    from capplan.data.calendar import grid_from_config

    grid = grid_from_config(cfg.with_overrides({"calendar.interval_minutes": 15}))
    assert grid.interval_index(datetime(2026, 1, 2, 7, 45)) == -1
    assert grid.interval_index(datetime(2026, 1, 2, 17, 0)) == -1   # end is exclusive
    assert grid.interval_index(datetime(2026, 1, 2, 8, 0)) == 0
    assert grid.interval_index(datetime(2026, 1, 2, 16, 45)) == 35
    assert grid.interval_index(datetime(2026, 1, 2, 8, 7)) == -1    # unaligned

    hourly = grid_from_config(cfg.with_overrides({"calendar.interval_minutes": 60}))
    assert hourly.interval_index(datetime(2026, 1, 2, 16, 0)) == 8
    # Aligned for a 15-minute grid, off-grid for an hourly one.
    assert hourly.interval_index(datetime(2026, 1, 2, 16, 45)) == -1


def test_prime_time_frame_is_dense(grid):
    frame = prime_time_frame(grid, date(2026, 1, 1), date(2026, 1, 31))
    n_days = len(grid.business_days(date(2026, 1, 1), date(2026, 1, 31)))
    assert len(frame) == n_days * grid.intervals_per_day
    assert frame["interval_idx"].max() == grid.intervals_per_day - 1


def test_misaligned_prime_window_is_rejected():
    bad = PrimeTimeGrid(
        interval_minutes=25, start=__import__("datetime").time(8), end=__import__("datetime").time(17),
        fiscal_year_start_month=11, holidays=frozenset(),
    )
    with pytest.raises(Exception):
        _ = bad.intervals_per_day
