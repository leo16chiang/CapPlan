"""Fiscal and prime-time calendar.

The whole architecture rests on one assumption confirmed in week 1: SMF
intervals are 15 minutes. That gives 36 prime-time intervals per business day
(08:00 through 16:45). If the extract turns out to be coarser, this module is
the only place that has to change -- but so does the row count that justifies a
neural Stage 1 at all, so it is worth confirming before anything else.

Fiscal year runs November through October. FY2026 = 2025-11-01 .. 2026-10-31.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from capplan.config import Config, ConfigError


@dataclass(frozen=True)
class PrimeTimeGrid:
    """The (business day x intra-day interval) lattice everything is indexed on."""

    interval_minutes: int
    start: time
    end: time  # exclusive
    fiscal_year_start_month: int
    holidays: frozenset[date]
    timezone: str = "America/New_York"

    @property
    def intervals_per_day(self) -> int:
        span = _minutes(self.end) - _minutes(self.start)
        if span <= 0:
            raise ConfigError("prime_end must be later than prime_start")
        if span % self.interval_minutes != 0:
            raise ConfigError(
                f"prime window of {span} min is not a whole number of "
                f"{self.interval_minutes}-min intervals"
            )
        return span // self.interval_minutes

    @property
    def interval_starts(self) -> list[time]:
        """Clock times of each prime-time interval start, e.g. 08:00 .. 16:45."""
        base = _minutes(self.start)
        out = []
        for k in range(self.intervals_per_day):
            total = base + k * self.interval_minutes
            out.append(time(hour=total // 60, minute=total % 60))
        return out

    # -- business days ---------------------------------------------------

    def is_business_day(self, day: date) -> bool:
        return day.weekday() < 5 and day not in self.holidays

    def business_days(self, start: date, end: date) -> list[date]:
        """Business days in [start, end] inclusive."""
        if end < start:
            return []
        n = (end - start).days + 1
        return [d for d in (start + timedelta(days=i) for i in range(n)) if self.is_business_day(d)]

    # -- prime-time timestamps -------------------------------------------

    def day_timestamps(self, day: date) -> list[datetime]:
        return [datetime.combine(day, t) for t in self.interval_starts]

    def timestamps(self, start: date, end: date) -> list[datetime]:
        """Every prime-time interval start in [start, end], business days only."""
        out: list[datetime] = []
        for day in self.business_days(start, end):
            out.extend(self.day_timestamps(day))
        return out

    def interval_index(self, ts: datetime) -> int:
        """Position of `ts` within the prime-time day, or -1 if off-prime."""
        minutes = ts.hour * 60 + ts.minute
        base = _minutes(self.start)
        if minutes < base or minutes >= _minutes(self.end):
            return -1
        offset = minutes - base
        if offset % self.interval_minutes != 0:
            return -1
        return offset // self.interval_minutes

    def is_prime_time(self, ts: datetime) -> bool:
        return self.is_business_day(ts.date()) and self.interval_index(ts) >= 0

    # -- fiscal years -----------------------------------------------------

    def fiscal_year(self, day: date) -> int:
        """FY label for a date. Nov-Dec belong to the next calendar year's FY."""
        return day.year + 1 if day.month >= self.fiscal_year_start_month else day.year

    def fiscal_year_bounds(self, fy: int) -> tuple[date, date]:
        start = date(fy - 1, self.fiscal_year_start_month, 1)
        end = date(fy, self.fiscal_year_start_month, 1) - timedelta(days=1)
        return start, end

    def fiscal_year_business_days(self, fy: int) -> list[date]:
        start, end = self.fiscal_year_bounds(fy)
        return self.business_days(start, end)

    def fiscal_years_forward(self, anchor: date, n_years: int) -> list[int]:
        """The next `n_years` fiscal years strictly after the one containing `anchor`."""
        current = self.fiscal_year(anchor)
        return [current + i for i in range(1, n_years + 1)]

    def horizon_business_days(self, anchor: date, n_years: int) -> list[date]:
        """Every business day from just after `anchor` to the end of the last FY.

        This is the forecast horizon: the model has to reach the end of the
        second fiscal year out, not just n_years x 250 days.
        """
        fys = self.fiscal_years_forward(anchor, n_years)
        _, end = self.fiscal_year_bounds(fys[-1])
        return self.business_days(anchor + timedelta(days=1), end)


def _minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def _parse_time(raw: str | time) -> time:
    if isinstance(raw, time):
        return raw
    hh, _, mm = str(raw).partition(":")
    return time(hour=int(hh), minute=int(mm or 0))


@lru_cache(maxsize=8)
def _load_holidays(path_str: str) -> frozenset[date]:
    path = Path(path_str)
    if not path.exists():
        return frozenset()
    days: set[date] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            days.add(date.fromisoformat(line))
    return frozenset(days)


def grid_from_config(cfg: Config) -> PrimeTimeGrid:
    cal = cfg.section("calendar")
    return PrimeTimeGrid(
        interval_minutes=int(cal["interval_minutes"]),
        start=_parse_time(cal["prime_start"]),
        end=_parse_time(cal["prime_end"]),
        fiscal_year_start_month=int(cal["fiscal_year_start_month"]),
        holidays=_load_holidays(str(cal.get("holiday_calendar", ""))),
        timezone=str(cal.get("timezone", "America/New_York")),
    )


def prime_time_frame(grid: PrimeTimeGrid, start: date, end: date) -> pd.DataFrame:
    """Skeleton frame of every prime-time interval in the window.

    Columns: ts, business_date, interval_idx, fiscal_year, dow, month.
    Joining raw data onto this makes missing intervals explicit instead of
    silently absent -- a missing SMF interval and a zero-MIPS interval mean
    very different things.
    """
    days = grid.business_days(start, end)
    per_day = grid.intervals_per_day
    if not days:
        return pd.DataFrame(
            {
                "ts": pd.Series(dtype="datetime64[ns]"),
                "business_date": pd.Series(dtype="object"),
                "interval_idx": pd.Series(dtype="int16"),
                "fiscal_year": pd.Series(dtype="int16"),
                "dow": pd.Series(dtype="int8"),
                "month": pd.Series(dtype="int8"),
            }
        )
    times = grid.interval_starts
    ts = [datetime.combine(d, t) for d in days for t in times]
    return pd.DataFrame(
        {
            "ts": pd.to_datetime(ts),
            "business_date": np.repeat(np.array(days, dtype="object"), per_day),
            "interval_idx": np.tile(np.arange(per_day, dtype="int16"), len(days)),
            "fiscal_year": np.repeat(
                np.array([grid.fiscal_year(d) for d in days], dtype="int16"), per_day
            ),
            "dow": np.repeat(np.array([d.weekday() for d in days], dtype="int8"), per_day),
            "month": np.repeat(np.array([d.month for d in days], dtype="int8"), per_day),
        }
    )


def day_index(days: Sequence[date]) -> dict[date, int]:
    """Stable positional index used to align residual day-blocks."""
    return {d: i for i, d in enumerate(days)}


def iter_month_groups(days: Sequence[date]) -> Iterator[tuple[tuple[int, int], list[int]]]:
    """Yield ((year, month), positions) for reducers that work on monthly peaks."""
    groups: dict[tuple[int, int], list[int]] = {}
    for i, d in enumerate(days):
        groups.setdefault((d.year, d.month), []).append(i)
    for key in sorted(groups):
        yield key, groups[key]


def fiscal_year_groups(
    grid: PrimeTimeGrid, days: Sequence[date]
) -> dict[int, list[int]]:
    """Positions of each day within its fiscal year."""
    groups: dict[int, list[int]] = {}
    for i, d in enumerate(days):
        groups.setdefault(grid.fiscal_year(d), []).append(i)
    return groups


def describe(grid: PrimeTimeGrid, days: Iterable[date]) -> str:
    days = list(days)
    if not days:
        return "empty calendar"
    return (
        f"{len(days)} business days {days[0]} .. {days[-1]}, "
        f"{grid.intervals_per_day} x {grid.interval_minutes}-min prime intervals/day "
        f"({grid.start:%H:%M}-{grid.end:%H:%M})"
    )
