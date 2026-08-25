"""The calendar carries the whole architecture. If prime time is wrong,
everything downstream is wrong in a way no other test would catch.
"""

from datetime import date

import pytest

from capplan.data.calendar import PrimeTimeGrid, prime_time_frame


def test_36_intervals_per_prime_day(grid):
    """The number the row-count argument -- and the neural net -- rests on."""
    assert grid.interval_minutes == 15
    assert grid.intervals_per_day == 36
    assert grid.interval_starts[0].hour == 8
    assert grid.interval_starts[-1].strftime("%H:%M") == "16:45"


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


def test_off_prime_timestamps_are_rejected(grid):
    from datetime import datetime

    assert grid.interval_index(datetime(2026, 1, 2, 7, 45)) == -1
    assert grid.interval_index(datetime(2026, 1, 2, 17, 0)) == -1   # end is exclusive
    assert grid.interval_index(datetime(2026, 1, 2, 8, 0)) == 0
    assert grid.interval_index(datetime(2026, 1, 2, 16, 45)) == 35
    assert grid.interval_index(datetime(2026, 1, 2, 8, 7)) == -1    # unaligned


def test_prime_time_frame_is_dense(grid):
    frame = prime_time_frame(grid, date(2026, 1, 1), date(2026, 1, 31))
    n_days = len(grid.business_days(date(2026, 1, 1), date(2026, 1, 31)))
    assert len(frame) == n_days * 36
    assert frame["interval_idx"].max() == 35


def test_misaligned_prime_window_is_rejected():
    bad = PrimeTimeGrid(
        interval_minutes=25, start=__import__("datetime").time(8), end=__import__("datetime").time(17),
        fiscal_year_start_month=11, holidays=frozenset(),
    )
    with pytest.raises(Exception):
        _ = bad.intervals_per_day
