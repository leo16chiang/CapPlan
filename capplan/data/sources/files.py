"""CSV / parquet source.

For the common case where the Db2 extract is done by someone else, on a
schedule, into a landing directory -- and for the equally common case where the
first pass is a one-off pull to a laptop.

Expects one file (or a glob of files) per table, with column names matching the
lake contract:

    data_drop/intervals*.csv
    data_drop/lpar_totals*.csv
    data_drop/events*.csv
    data_drop/submissions*.csv

Any of the four may be absent; the pipeline degrades in documented ways
(no `lpar_totals` means no simulation backtest).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterator

import pandas as pd

from capplan.data.sources.base import register_source
from capplan.data.sources.sql import REQUIRED, SchemaDriftError, _profile
from capplan.logging_utils import get_logger

LOG = get_logger(__name__)


class FileSource:
    name = "files"

    def __init__(self, root: str | Path, column_map: dict | None = None) -> None:
        self.root = Path(root)
        self.column_map = column_map or {}
        if not self.root.exists():
            raise FileNotFoundError(f"data drop directory not found: {self.root}")

    def _paths(self, table: str) -> list[Path]:
        found: list[Path] = []
        for suffix in ("parquet", "csv", "csv.gz", "tsv"):
            found.extend(sorted(self.root.glob(f"{table}*.{suffix}")))
        return found

    def fetch(self, table: str, start: date, end: date) -> Iterator[pd.DataFrame]:
        paths = self._paths(table)
        if not paths:
            LOG.info("no files matching %s* in %s", table, self.root)
            return
        for path in paths:
            frame = _read(path)
            frame = self._prepare(frame, table)
            frame = _clip_to_window(frame, start, end)
            if frame.empty:
                continue
            LOG.info("%s: %d rows from %s", table, len(frame), path.name)
            yield frame

    def _prepare(self, frame: pd.DataFrame, table: str) -> pd.DataFrame:
        frame = frame.rename(columns={c: str(c).strip().lower() for c in frame.columns})
        mapping = {k.lower(): v for k, v in (self.column_map.get(table) or {}).items()}
        if mapping:
            frame = frame.rename(columns=mapping)
        missing = [c for c in REQUIRED.get(table, ()) if c not in frame.columns]
        if missing:
            raise SchemaDriftError(
                f"{table} files are missing {missing}. Got: {sorted(frame.columns)}"
            )
        for col in ("ts", "start_ts", "end_ts"):
            if col in frame.columns:
                frame[col] = pd.to_datetime(frame[col])
        for col in ("business_date", "submitted_on"):
            if col in frame.columns:
                frame[col] = pd.to_datetime(frame[col]).dt.date
        for col in ("app_id", "lpar", "environment", "event_type"):
            if col in frame.columns:
                frame[col] = frame[col].astype("string").str.strip()
        return frame

    def probe(self, start: date, end: date, sample_rows: int = 500) -> dict:
        report: dict = {"source": self.name, "root": str(self.root)}
        for table in REQUIRED:
            paths = self._paths(table)
            if not paths:
                report[table] = {"rows_sampled": 0, "files": []}
                continue
            frame = self._prepare(_read(paths[0]).head(sample_rows), table)
            report[table] = {**_profile(frame, table), "files": [p.name for p in paths]}
        return report


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    separator = "\t" if path.name.endswith(".tsv") else ","
    return pd.read_csv(path, sep=separator)


def _clip_to_window(frame: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    if "ts" in frame.columns:
        days = frame["ts"].dt.date
    elif "business_date" in frame.columns:
        days = pd.Series(frame["business_date"])
    else:
        # Reference tables (events, submissions) are not date-partitioned.
        return frame
    return frame[(days >= start) & (days <= end)]


@register_source("files")
def build(root: str | Path = "data_drop", column_map: dict | None = None, **_ignored):
    return FileSource(root=root, column_map=column_map)
