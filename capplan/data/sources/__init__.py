"""Data sources.

CapPlan reads four tables. Where they come from is a plug: a Db2 warehouse, a
directory of CSV extracts, or the synthetic generator. The source's only job is
to produce frames matching `capplan.data.schema`; everything after that --
scoping, normalisation, anomaly labelling -- is shared.

    intervals     one row per (app, LPAR, prime-time interval)   <- SMF 72-3
    lpar_totals   one row per (LPAR, business day)               <- SMF 70-1
    events        DR / IST / GCC SDF windows                     <- change calendar
    submissions   what custodians forecast last cycle            <- capacity planning

`lpar_totals` must come from SMF 70-1, not from summing the `intervals` rows.
See `capplan/data/sources/sql.py` for why that distinction is load-bearing
rather than pedantic.
"""

from capplan.data.sources.base import ExtractReport, Source, get_source, register_source

# Import for the side effect of registering. Each module guards its own
# optional dependency, so importing this package never requires pyodbc.
from capplan.data.sources import files as _files  # noqa: F401
from capplan.data.sources import db2 as _db2  # noqa: F401
from capplan.data.sources import synthetic as _synthetic  # noqa: F401

__all__ = ["ExtractReport", "Source", "get_source", "register_source"]
