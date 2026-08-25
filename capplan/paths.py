"""Filesystem layout for the data lake.

Parquet on disk, queried through DuckDB. ~4M rows total -- this is small data
and deliberately does not have a warehouse behind it.
"""

from __future__ import annotations

from pathlib import Path

LAKE_ROOT = Path("data_lake")

RAW = LAKE_ROOT / "raw"                  # landed SMF extracts, untouched
INTERVALS = LAKE_ROOT / "intervals"      # normalised (app, lpar, ts) MIPS
SUBMISSIONS = LAKE_ROOT / "submissions"  # custodian-submitted forecasts
EVENTS = LAKE_ROOT / "events"            # DR / IST / GCC SDF windows
LPAR_TOTALS = LAKE_ROOT / "lpar_totals"  # realised LPAR-level peaks (SMF 70-1)
SERVING = LAKE_ROOT / "serving"          # what the Dash app reads

INTERVALS_FILE = INTERVALS / "intervals.parquet"
SUBMISSIONS_FILE = SUBMISSIONS / "submissions.parquet"
EVENTS_FILE = EVENTS / "events.parquet"
LPAR_TOTALS_FILE = LPAR_TOTALS / "lpar_totals.parquet"


def ensure_lake(root: Path = LAKE_ROOT) -> None:
    """Create the lake directory skeleton. Idempotent."""
    for sub in ("raw", "intervals", "submissions", "events", "lpar_totals", "serving"):
        (root / sub).mkdir(parents=True, exist_ok=True)
