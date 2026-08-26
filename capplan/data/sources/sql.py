"""Generic SQL source over any DB-API 2.0 connection.

Db2 is one instance of this; the tests drive the same code through SQLite. The
database-specific part is only the connection and a couple of dialect details,
which is why `db2.py` is thin.

Three things this does that a `SELECT *` / `fetchall()` loop does not.

**It never holds the whole result set.** The interval table is roughly 950k
prime-time rows over three years, and that is *after* scoping -- the raw SMF 72
extract before the prime-time filter is 10-20x larger, because it contains
every interval of every day including nights and weekends. `fetchall()` on that
returns a list of tuples with per-row Python object overhead measured in
hundreds of bytes. Extraction runs in date chunks (a month at a time by
default) and each chunk is converted to a DataFrame and written before the next
is fetched.

**It pushes the filter to the database.** Prime time is ~21% of the wall-clock
week (9 hours x 5 days out of 168). Filtering in SQL rather than in pandas
means the network carries a fifth of the rows. The default query templates do
this with an hour-of-day and day-of-week predicate.

**It fails loudly on schema drift.** A query that stops returning `app_id`
because someone renamed a column should stop the run, not produce a silently
empty forecast.

## The one distinction that is not pedantic

`intervals` and `lpar_totals` must come from **different SMF records**:

    intervals    SMF 72 subtype 3 (Workload Activity) -- per service class /
                 report class, which is what maps to an application
    lpar_totals  SMF 70 subtype 1 (RMF CPU Activity)  -- per LPAR partition

It is tempting to derive `lpar_totals` by summing the `intervals` rows. Do not.
The entire evaluation strategy is: simulate the LPAR peak from application-level
data, then check it against what the LPAR actually did. If the "actual" is
itself computed by summing application rows, the backtest compares the
simulation against its own assumption and will pass no matter how wrong the
coincidence model is. That failure is invisible until a hardware configuration
has been signed.

A useful side effect of keeping them separate: SMF 70-1 includes work that SMF
72-3 does not attribute to any of your top-N applications, so the gap between
them is a real measurement -- `capplan diagnostics` reports it as
`n_unattributed_days`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Iterator, Mapping, Sequence

import pandas as pd

from capplan.data.sources.base import ExtractReport
from capplan.logging_utils import get_logger

LOG = get_logger(__name__)

# Columns each table must carry by the time it leaves the source. Anything
# else the query returns is passed through untouched (extra columns are
# harmless; `cpu_model` for instance is picked up by normalisation).
REQUIRED = {
    "intervals": ("ts", "app_id", "lpar"),
    "lpar_totals": ("business_date", "lpar", "peak_mips"),
    "events": ("event_type", "start_ts", "end_ts"),
    "submissions": ("app_id", "fiscal_year", "submitted_peak_mips"),
}

# Either of these satisfies the "how much work" requirement on intervals; the
# normalisation step derives whichever is missing.
VALUE_COLUMNS = ("mips", "msu")


class SchemaDriftError(RuntimeError):
    """A query returned columns that do not match the lake contract."""


@dataclass
class SqlSource:
    """Chunked extraction from a DB-API connection.

    `connect` is a zero-argument callable returning a fresh connection. It is
    a callable rather than a live connection so a long extract can reconnect
    between chunks -- mainframe-adjacent gateways drop idle sessions, and
    losing hour three of a four-hour extract to an idle timeout is a bad
    afternoon.
    """

    connect: Callable[[], Any]
    queries: Mapping[str, str]
    name: str = "sql"
    chunk_days: int = 30
    fetch_batch_rows: int = 50_000
    column_map: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    reconnect_each_chunk: bool = True
    paramstyle: str = "qmark"

    # -- extraction -------------------------------------------------------

    def fetch(self, table: str, start: date, end: date) -> Iterator[pd.DataFrame]:
        """Yield one DataFrame per date chunk."""
        query = self.queries.get(table)
        if not query:
            LOG.info("no query configured for %s; skipping", table)
            return

        windows = list(self._windows(table, start, end))
        connection = None if self.reconnect_each_chunk else self.connect()
        try:
            for lo, hi in windows:
                conn = self.connect() if self.reconnect_each_chunk else connection
                try:
                    frame = self._run(conn, query, table, lo, hi)
                finally:
                    if self.reconnect_each_chunk:
                        _close(conn)
                if frame is None or frame.empty:
                    LOG.info("%s %s..%s: no rows", table, lo, hi)
                    continue
                LOG.info("%s %s..%s: %d rows", table, lo, hi, len(frame))
                yield frame
        finally:
            if connection is not None:
                _close(connection)

    def _windows(self, table: str, start: date, end: date) -> Iterator[tuple[date, date]]:
        """Date chunks, half-open at the top so no interval is double-counted."""
        # Reference data is small and unpartitioned -- one shot, no chunking.
        if table in ("events", "submissions"):
            yield start, end + timedelta(days=1)
            return
        cursor = start
        step = timedelta(days=max(self.chunk_days, 1))
        stop = end + timedelta(days=1)
        while cursor < stop:
            yield cursor, min(cursor + step, stop)
            cursor += step

    def _run(self, conn, query: str, table: str, lo: date, hi: date) -> pd.DataFrame | None:
        cursor = conn.cursor()
        try:
            cursor.execute(query, self._params(lo, hi))
            columns = [d[0].lower() for d in cursor.description]
            batches = []
            while True:
                rows = cursor.fetchmany(self.fetch_batch_rows)
                if not rows:
                    break
                # pyodbc returns Row objects; tuple() makes the DataFrame
                # constructor take the fast path instead of treating each row
                # as a mapping.
                batches.append(pd.DataFrame([tuple(r) for r in rows], columns=columns))
            if not batches:
                return None
            frame = pd.concat(batches, ignore_index=True) if len(batches) > 1 else batches[0]
        finally:
            cursor.close()
        return self._normalise_columns(frame, table)

    def _params(self, lo: date, hi: date) -> Sequence[Any] | Mapping[str, Any]:
        if self.paramstyle == "named":
            return {"start": lo, "end": hi}
        return (lo, hi)

    def _normalise_columns(self, frame: pd.DataFrame, table: str) -> pd.DataFrame:
        frame = frame.rename(columns={c: c.lower() for c in frame.columns})
        mapping = {k.lower(): v for k, v in (self.column_map.get(table) or {}).items()}
        if mapping:
            frame = frame.rename(columns=mapping)

        missing = [c for c in REQUIRED.get(table, ()) if c not in frame.columns]
        if missing:
            raise SchemaDriftError(
                f"query for {table!r} did not return {missing}. Got: "
                f"{sorted(frame.columns)}. Alias the columns in "
                f"config/sources.yaml (e.g. `SELECT SMF_TS AS ts`) or add a "
                f"column_map entry."
            )
        if table == "intervals" and not any(c in frame.columns for c in VALUE_COLUMNS):
            raise SchemaDriftError(
                f"query for 'intervals' returned neither {' nor '.join(VALUE_COLUMNS)}. "
                "One of them is the workload measure; normalisation derives the other."
            )

        for col in ("ts", "start_ts", "end_ts"):
            if col in frame.columns:
                frame[col] = pd.to_datetime(frame[col])
        for col in ("business_date", "submitted_on"):
            if col in frame.columns:
                frame[col] = pd.to_datetime(frame[col]).dt.date
        for col in ("app_id", "lpar", "environment", "event_type"):
            if col in frame.columns:
                # Trailing blanks are the default in fixed-width CHAR columns,
                # and 'PAYMENTS  ' != 'PAYMENTS' will silently split one
                # application into two.
                frame[col] = frame[col].astype("string").str.strip()
        return frame

    # -- reconnaissance ---------------------------------------------------

    def probe(self, start: date, end: date, sample_rows: int = 500) -> dict:
        """Answer the week-1 questions without pulling the full history.

        Cheap by construction: it reads one short window, not the archive.
        """
        report: dict[str, Any] = {"source": self.name, "window": [str(start), str(end)]}
        conn = self.connect()
        try:
            for table, query in self.queries.items():
                cursor = conn.cursor()
                try:
                    cursor.execute(query, self._params(start, end + timedelta(days=1)))
                    columns = [d[0].lower() for d in cursor.description]
                    rows = cursor.fetchmany(sample_rows)
                finally:
                    cursor.close()
                if not rows:
                    report[table] = {"rows_sampled": 0, "columns": columns}
                    continue
                frame = self._normalise_columns(
                    pd.DataFrame([tuple(r) for r in rows], columns=columns), table
                )
                report[table] = _profile(frame, table)
        finally:
            _close(conn)
        return report


def _profile(frame: pd.DataFrame, table: str) -> dict:
    out: dict[str, Any] = {
        "rows_sampled": len(frame),
        "columns": sorted(frame.columns),
    }
    if "ts" in frame.columns and len(frame) > 1:
        deltas = (
            frame.sort_values(["app_id", "ts"] if "app_id" in frame else "ts")["ts"]
            .diff()
            .dt.total_seconds()
            .div(60)
            .dropna()
        )
        positive = deltas[deltas > 0]
        if not positive.empty:
            # THE week-1 question. 36 prime-time intervals a day, and the ~950k
            # rows that justify a neural Stage 1, both follow from this being 15.
            out["interval_minutes_mode"] = float(positive.mode().iloc[0])
            out["interval_minutes_distinct"] = sorted(positive.unique().tolist())[:10]
        out["ts_min"] = str(frame["ts"].min())
        out["ts_max"] = str(frame["ts"].max())
    for col in ("app_id", "lpar", "environment", "event_type"):
        if col in frame.columns:
            values = frame[col].dropna().unique().tolist()
            out[f"distinct_{col}"] = len(values)
            out[f"sample_{col}"] = sorted(map(str, values))[:12]
    for col in ("mips", "msu", "peak_mips"):
        if col in frame.columns:
            series = pd.to_numeric(frame[col], errors="coerce")
            out[f"{col}_min"] = float(series.min())
            out[f"{col}_max"] = float(series.max())
            out[f"{col}_nulls"] = int(series.isna().sum())
    return out


def extract_to_frames(
    source, tables: Sequence[str], start: date, end: date
) -> tuple[dict[str, pd.DataFrame], ExtractReport]:
    """Run a source over every table and concatenate the batches.

    Concatenation is fine at this scale: post-scoping the whole lake is ~4M
    rows. If a site's raw extract is large enough that even the pre-scope
    intervals frame does not fit, scope inside the SQL (the query templates
    already filter to prime time) rather than reaching for a bigger machine.
    """
    started = time.time()
    report = ExtractReport(source=getattr(source, "name", "unknown"))
    out: dict[str, pd.DataFrame] = {}
    for table in tables:
        batches = list(source.fetch(table, start, end))
        report.batches += len(batches)
        if not batches:
            report.warnings.append(f"{table}: no rows returned for {start}..{end}")
            continue
        frame = pd.concat(batches, ignore_index=True)
        out[table] = frame
        report.rows_by_table[table] = len(frame)
    if "intervals" in out and "ts" in out["intervals"]:
        report.date_min = out["intervals"]["ts"].min().date()
        report.date_max = out["intervals"]["ts"].max().date()
    report.seconds = time.time() - started

    if "lpar_totals" not in out:
        report.warnings.append(
            "no lpar_totals: without realised SMF 70-1 LPAR peaks there is no "
            "simulation backtest, and without that there is no evidence the "
            "coincidence model is right. `capplan backtest` will not run."
        )
    for warning in report.warnings:
        LOG.warning("%s", warning)
    return out, report


def _close(conn) -> None:
    try:
        conn.close()
    except Exception:  # pragma: no cover - best effort
        LOG.debug("connection close failed", exc_info=True)
