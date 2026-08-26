"""Table schemas for the lake.

Column names are the contract between the SQL diagnostics, the feature builder
and the simulator. Changing one means changing all three, so they are declared
in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

# -- intervals: the fact table ------------------------------------------------
# One row per (app, lpar, prime-time interval).
INTERVAL_COLUMNS: dict[str, str] = {
    "ts": "datetime64[ns]",       # interval start, local mainframe time
    "business_date": "object",    # datetime.date, for day-block alignment
    "interval_idx": "int16",      # 0..35 within the prime window
    "fiscal_year": "int16",
    "app_id": "object",
    "lpar": "object",
    "environment": "object",      # PROD only, after scoping
    "msu": "float64",             # as reported by SMF/RMF
    "mips": "float64",            # normalised: msu * mips_per_msu / capture_ratio
    "capture_ratio": "float64",   # applied as given, recorded for audit
    "mips_per_msu": "float64",
    "is_anomaly": "bool",         # DR / IST / GCC SDF landing in prime time
    "event_label": "object",      # None, or the event type
}

# -- lpar_totals: realised LPAR peaks from SMF 70-1 ---------------------------
# The ground truth the simulation is backtested against. NOT the sum of app
# rows: it is measured at the LPAR, so it already contains the true coincidence.
LPAR_TOTAL_COLUMNS: dict[str, str] = {
    "business_date": "object",
    "lpar": "object",
    "fiscal_year": "int16",
    "peak_mips": "float64",        # max over prime-time intervals that day
    "peak_interval_idx": "int16",  # which interval the peak landed in
    "mean_mips": "float64",
}

# -- submissions: what custodians told capacity planning last cycle -----------
SUBMISSION_COLUMNS: dict[str, str] = {
    "app_id": "object",
    "fiscal_year": "int16",         # FY the submission is *about*
    "submitted_on": "object",       # date the custodian submitted
    "submitted_peak_mips": "float64",
    "basis": "object",              # free text: "volume growth", "flat", ...
}

# -- events: DR / IST / GCC SDF windows ---------------------------------------
EVENT_COLUMNS: dict[str, str] = {
    "event_id": "object",
    "event_type": "object",         # DR | IST | GCC_SDF
    "lpar": "object",               # '*' = all LPARs
    "app_id": "object",             # '*' = all apps
    "start_ts": "datetime64[ns]",
    "end_ts": "datetime64[ns]",
    "note": "object",
}


# Columns CapPlan can derive or default when a site's extract does not carry
# them. A first extract almost never has all of these -- `peak_interval_idx`
# in particular requires an ARG_MAX the warehouse may not make easy -- and
# failing the run over a column that only feeds one diagnostic chart would be
# the wrong trade. They are filled with an explicit "unknown" marker rather
# than a plausible-looking value.
OPTIONAL_DEFAULTS: dict[str, dict[str, object]] = {
    "intervals": {
        "environment": "PROD",
        "is_anomaly": False,
        "event_label": None,
        "capture_ratio": 1.0,
    },
    "lpar_totals": {
        "peak_interval_idx": -1,   # -1 = not supplied, never a real interval
        "mean_mips": float("nan"),
    },
    "events": {"event_id": None, "lpar": "*", "app_id": "*", "note": None},
    "submissions": {"submitted_on": None, "basis": None},
}


def conform(frame, table: str, grid=None):
    """Fill derivable and defaultable columns so downstream SQL can rely on them.

    Two derivations rather than defaults, because a wrong value here is worse
    than a missing one:
      * `business_date` from `ts`
      * `fiscal_year` from `business_date`, using the configured calendar
    """
    import pandas as pd

    out = frame.copy()
    if "ts" in out.columns and "business_date" not in out.columns:
        out["business_date"] = pd.to_datetime(out["ts"]).dt.date
    if grid is not None and "business_date" in out.columns and "fiscal_year" not in out.columns:
        out["fiscal_year"] = [grid.fiscal_year(d) for d in out["business_date"]]
    for column, default in OPTIONAL_DEFAULTS.get(table, {}).items():
        if column not in out.columns:
            out[column] = default
    return out


@dataclass(frozen=True)
class SchemaError(Exception):
    """Raised when an ingested frame does not match the declared schema."""

    table: str
    missing: tuple[str, ...]

    def __str__(self) -> str:  # pragma: no cover - message formatting
        return f"{self.table}: missing columns {', '.join(self.missing)}"


def require_columns(frame, table: str, columns: dict[str, str]) -> None:
    missing = tuple(c for c in columns if c not in frame.columns)
    if missing:
        raise SchemaError(table=table, missing=missing)
