"""Synthetic source, so `--source synthetic` sits alongside the real ones.

The generator itself lives in `capplan/data/synth.py`; this is the adapter that
makes it satisfy the same protocol as Db2, which keeps `capplan ingest` free of
special cases.
"""

from __future__ import annotations

from datetime import date
from typing import Iterator

import pandas as pd

from capplan.data.sources.base import register_source


class SyntheticSource:
    name = "synthetic"

    def __init__(self, grid, n_apps: int = 35, seed: int = 7) -> None:
        self.grid = grid
        self.n_apps = n_apps
        self.seed = seed
        self._cache: dict | None = None

    def _generate(self, start: date, end: date) -> dict:
        if self._cache is None:
            from capplan.data.synth import SynthSpec, generate

            self._cache = generate(
                self.grid,
                SynthSpec(n_apps=self.n_apps, start=start, end=end, seed=self.seed),
            )
        return self._cache

    def fetch(self, table: str, start: date, end: date) -> Iterator[pd.DataFrame]:
        data = self._generate(start, end)
        if table in data:
            yield data[table]

    def probe(self, start: date, end: date, sample_rows: int = 500) -> dict:
        from capplan.data.sources.sql import _profile

        data = self._generate(start, end)
        report: dict = {"source": self.name}
        for table in ("intervals", "lpar_totals", "events", "submissions"):
            if table in data:
                report[table] = _profile(data[table].head(sample_rows), table)
        return report


@register_source("synthetic")
def build(grid=None, n_apps: int = 35, seed: int = 7, **_ignored) -> SyntheticSource:
    if grid is None:
        raise ValueError("the synthetic source needs a PrimeTimeGrid")
    return SyntheticSource(grid=grid, n_apps=n_apps, seed=seed)
