"""Source protocol and registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Iterator, Protocol, runtime_checkable

import pandas as pd

TABLES = ("intervals", "lpar_totals", "events", "submissions")


@dataclass
class ExtractReport:
    """What an extract actually pulled. Goes into the run manifest verbatim."""

    source: str
    rows_by_table: dict[str, int] = field(default_factory=dict)
    batches: int = 0
    seconds: float = 0.0
    date_min: date | None = None
    date_max: date | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "rows_by_table": dict(self.rows_by_table),
            "batches": self.batches,
            "seconds": round(self.seconds, 1),
            "date_min": str(self.date_min) if self.date_min else None,
            "date_max": str(self.date_max) if self.date_max else None,
            "warnings": list(self.warnings),
        }


@runtime_checkable
class Source(Protocol):
    """Produce lake tables. Nothing here knows about MIPS or prime time."""

    name: str

    def fetch(
        self, table: str, start: date, end: date
    ) -> Iterator[pd.DataFrame]:
        """Yield batches for `table` over [start, end]. May yield nothing."""

    def probe(self, start: date, end: date) -> dict:
        """Cheap reconnaissance: does the data look like what we assume?"""


_REGISTRY: dict[str, Callable[..., Source]] = {}


def register_source(name: str) -> Callable:
    def wrap(factory: Callable[..., Source]) -> Callable[..., Source]:
        _REGISTRY[name] = factory
        return factory

    return wrap


def get_source(name: str, **kwargs) -> Source:
    if name not in _REGISTRY:
        raise KeyError(f"unknown source {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_sources() -> list[str]:
    return sorted(_REGISTRY)
