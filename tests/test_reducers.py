"""Reducers are the seam the unsettled fiscal-year convention hides behind.

The tests that matter are: the contract is honoured, the conventions genuinely
differ, and a new one can be added without touching anything else.
"""

from datetime import date, timedelta

import numpy as np
import pytest

from capplan.sim.reducers import (
    REDUCERS,
    PathSummary,
    get_reducer,
    reduce_by_fiscal_year,
    register_reducer,
)


def make_path(n_days: int = 250, seed: int = 0) -> PathSummary:
    rng = np.random.default_rng(seed)
    start = date(2026, 11, 2)
    days, day = [], start
    while len(days) < n_days:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    peaks = 4000 * np.exp(rng.standard_t(4, n_days) * 0.06)
    return PathSummary(daily_peaks=peaks, daily_means=peaks * 0.7, days=days)


def test_every_reducer_returns_a_scalar():
    """Including the R4HA family, which needs the rolling series supplied."""
    rng = np.random.default_rng(1)
    path = make_path()
    path.daily_r4ha = path.daily_peaks * rng.uniform(0.7, 0.95, len(path.daily_peaks))
    for name, reducer in REDUCERS.items():
        if reducer.needs_intervals:
            continue
        value = reducer(path)
        assert isinstance(value, float) and np.isfinite(value), name


def test_conventions_are_ordered_as_expected():
    """These are not interchangeable, and the ordering is the reason."""
    path = make_path()
    assert (
        get_reducer("mean_of_daily_peaks")(path)
        < get_reducer("p95_of_daily_peaks")(path)
        < get_reducer("p99_of_daily_peaks")(path)
        <= get_reducer("annual_max")(path)
    )
    assert get_reducer("mean_of_monthly_peaks")(path) < get_reducer("annual_max")(path)


def test_the_choice_of_convention_moves_the_number_materially():
    """If these agreed, the pluggable seam would be over-engineering."""
    path = make_path()
    low = get_reducer("p95_of_daily_peaks")(path)
    high = get_reducer("annual_max")(path)
    assert (high - low) / low > 0.10


def test_max_of_monthly_peaks_equals_annual_max():
    path = make_path()
    assert get_reducer("max_of_monthly_peaks")(path) == pytest.approx(
        get_reducer("annual_max")(path)
    )


def test_reduce_by_fiscal_year_splits_the_horizon():
    path = make_path(n_days=500)
    fys = np.array([2027 if d.year >= 2027 else 2026 for d in path.days])
    path.fiscal_years = fys
    out = reduce_by_fiscal_year(path, get_reducer("annual_max"), fys)
    assert set(out) == {2026, 2027}
    assert max(out.values()) == pytest.approx(get_reducer("annual_max")(path))


def test_a_new_convention_needs_only_a_registration():
    """The whole point: settle the definition later, change one function."""

    @register_reducer("custom_top10_mean", "Mean of the ten highest daily peaks.")
    def _top10(path: PathSummary) -> float:
        return float(np.mean(np.sort(path.daily_peaks)[-10:]))

    try:
        path = make_path()
        value = get_reducer("custom_top10_mean")(path)
        assert get_reducer("p95_of_daily_peaks")(path) < value < get_reducer("annual_max")(path)
    finally:
        REDUCERS.pop("custom_top10_mean")


def test_interval_reducer_refuses_to_guess():
    path = make_path()
    with pytest.raises(ValueError):
        get_reducer("intervals_above_p99")(path)


def test_unknown_reducer_is_a_clear_error():
    with pytest.raises(KeyError, match="unknown reducer"):
        get_reducer("peak_of_vibes")
