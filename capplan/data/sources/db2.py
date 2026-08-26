"""Db2 source over pyodbc.

Thin on purpose -- the extraction machinery is in `sql.py`, which the tests
drive through SQLite. What lives here is the connection and the handful of Db2
dialect details.

## Credentials

Never in `config/capplan.yaml`, never in `config/sources.yaml`. Both are in
version control, and a password in git history outlives every attempt to remove
it. Read from the environment:

    export CAPPLAN_DB2_HOST=your_host
    export CAPPLAN_DB2_PORT=50000
    export CAPPLAN_DB2_DATABASE=your_db
    export CAPPLAN_DB2_USER=your_user
    export CAPPLAN_DB2_PASSWORD=...        # or CAPPLAN_DB2_PASSWORD_CMD
    export CAPPLAN_DB2_DRIVER='{IBM DB2 ODBC DRIVER}'

`CAPPLAN_DB2_PASSWORD_CMD` is run and its stdout used as the password, so a
site secret store (`vault read ...`, `security find-generic-password ...`) can
supply it without the value ever sitting in a shell variable.

`CAPPLAN_DB2_DSN` overrides all of the above with a complete ODBC connection
string, for sites where the DSN is managed centrally.

## Dialect notes

* Parameter markers are `?` (qmark), which is what the templates use.
* Row limiting is `FETCH FIRST n ROWS ONLY`, not `LIMIT`.
* `CURRENT TIMESTAMP`, not `NOW()`.
* Timestamp comparisons: pass `datetime.date` objects as parameters and let the
  driver bind them. String formatting into the SQL is both an injection risk
  and a portability problem across Db2 for z/OS and Db2 LUW.
* Db2 folds unquoted identifiers to upper case, so results come back as `TS`,
  `APP_ID`. `sql.py` lower-cases every column name on the way in.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from typing import Any

from capplan.data.sources.base import register_source
from capplan.data.sources.sql import SqlSource
from capplan.logging_utils import get_logger

LOG = get_logger(__name__)

ENV_PREFIX = "CAPPLAN_DB2_"


class Db2Unavailable(RuntimeError):
    """pyodbc is not installed, or the connection details are incomplete."""


def pyodbc_available() -> tuple[bool, str]:
    import importlib.util

    if importlib.util.find_spec("pyodbc") is None:
        return False, (
            "pyodbc is not installed (pip install 'capplan[db2]'). It also needs the "
            "IBM Db2 ODBC driver present on the machine -- the Python package alone "
            "is not enough."
        )
    return True, "pyodbc importable"


def connection_string(redact: bool = False) -> str:
    """Build the ODBC connection string from the environment."""
    dsn = os.environ.get(f"{ENV_PREFIX}DSN")
    if dsn:
        return "<CAPPLAN_DB2_DSN>" if redact else dsn

    required = ("HOST", "DATABASE", "USER")
    missing = [f"{ENV_PREFIX}{k}" for k in required if not os.environ.get(f"{ENV_PREFIX}{k}")]
    if missing:
        raise Db2Unavailable(
            f"missing environment variable(s): {', '.join(missing)}. "
            f"Set them, or set {ENV_PREFIX}DSN to a full ODBC connection string. "
            "Credentials never belong in config/."
        )

    password = _password()
    driver = os.environ.get(f"{ENV_PREFIX}DRIVER", "{IBM DB2 ODBC DRIVER}")
    parts = [
        f"DRIVER={driver}",
        f"HOSTNAME={os.environ[f'{ENV_PREFIX}HOST']}",
        f"PORT={os.environ.get(f'{ENV_PREFIX}PORT', '50000')}",
        f"DATABASE={os.environ[f'{ENV_PREFIX}DATABASE']}",
        f"UID={os.environ[f'{ENV_PREFIX}USER']}",
        f"PWD={'***' if redact else password}",
        f"PROTOCOL={os.environ.get(f'{ENV_PREFIX}PROTOCOL', 'TCPIP')}",
    ]
    if os.environ.get(f"{ENV_PREFIX}SECURITY"):
        parts.append(f"SECURITY={os.environ[f'{ENV_PREFIX}SECURITY']}")
    return ";".join(parts) + ";"


def _password() -> str:
    command = os.environ.get(f"{ENV_PREFIX}PASSWORD_CMD")
    if command:
        result = subprocess.run(
            shlex.split(command), capture_output=True, text=True, check=False, timeout=30
        )
        if result.returncode != 0:
            raise Db2Unavailable(
                f"{ENV_PREFIX}PASSWORD_CMD failed ({result.returncode}): "
                f"{result.stderr.strip()[:200]}"
            )
        return result.stdout.strip()
    password = os.environ.get(f"{ENV_PREFIX}PASSWORD")
    if password is None:
        raise Db2Unavailable(
            f"set {ENV_PREFIX}PASSWORD or {ENV_PREFIX}PASSWORD_CMD "
            "(the latter runs a command and reads the password from its stdout)"
        )
    return password


def connect() -> Any:
    """Open one Db2 connection. Read-only by intent -- CapPlan never writes."""
    ok, message = pyodbc_available()
    if not ok:
        raise Db2Unavailable(message)
    import pyodbc

    conn = pyodbc.connect(connection_string(), autocommit=True, timeout=60)
    # Extraction is read-only, so uncommitted reads avoid taking locks on
    # tables that an overnight SMF load may still be writing to. If the site
    # forbids UR, drop this -- the extract still works, it just contends.
    try:
        conn.execute("SET CURRENT ISOLATION = UR")
    except Exception:  # pragma: no cover - not every Db2 build allows it
        LOG.debug("could not set isolation UR; continuing at the default", exc_info=True)
    return conn


@register_source("db2")
def build(
    queries: dict[str, str],
    chunk_days: int = 30,
    fetch_batch_rows: int = 50_000,
    column_map: dict | None = None,
    **_ignored,
) -> SqlSource:
    return SqlSource(
        connect=connect,
        queries=queries,
        name="db2",
        chunk_days=chunk_days,
        fetch_batch_rows=fetch_batch_rows,
        column_map=column_map or {},
        reconnect_each_chunk=True,
        paramstyle="qmark",
    )
