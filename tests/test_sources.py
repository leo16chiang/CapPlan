"""Data sources.

The SQL path is tested against a real database -- SQLite standing in for Db2 --
so chunking, parameter binding, batched fetch and schema-drift detection are all
genuinely exercised. What SQLite cannot test is the ODBC driver and the Db2
dialect, and those are exactly what `capplan probe` is for.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from capplan.data.sources.base import available_sources, get_source
from capplan.data.sources.sql import SchemaDriftError, SqlSource, extract_to_frames


# -- a stand-in warehouse -------------------------------------------------


@pytest.fixture()
def warehouse(tmp_path):
    """Two applications, three business days, 15-minute prime-time intervals.

    Column names deliberately do NOT match CapPlan's -- that is the normal
    case, and the aliasing in the query is what bridges it.
    """
    path = tmp_path / "smf.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE WORKLOAD_72_3 ("
        " SMF_INTERVAL_START TEXT, LPAR_NAME TEXT, REPORT_CLASS TEXT,"
        " ENVIRONMENT TEXT, MSU_CONSUMED REAL)"
    )
    conn.execute(
        "CREATE TABLE PARTITION_70_1 ("
        " BUSINESS_DATE TEXT, LPAR_NAME TEXT, PEAK REAL, MEAN REAL)"
    )
    rows, totals = [], []
    for day_offset in range(3):
        day = date(2025, 6, 2) + timedelta(days=day_offset)
        peak = 0.0
        for slot in range(36):
            stamp = datetime(day.year, day.month, day.day, 8) + timedelta(minutes=15 * slot)
            for app, level in (("PAYMENTS ", 100.0), ("CARDS", 40.0)):
                # Trailing blank on PAYMENTS is not a typo: fixed-width CHAR
                # columns pad, and 'PAYMENTS ' != 'PAYMENTS' silently splits
                # one application into two.
                value = level + slot
                rows.append((stamp.isoformat(sep=" "), "PRDA", app, "PROD", value))
                peak = max(peak, value)
        totals.append((day.isoformat(), "PRDA", peak * 1.8, peak * 1.2))
    conn.executemany("INSERT INTO WORKLOAD_72_3 VALUES (?,?,?,?,?)", rows)
    conn.executemany("INSERT INTO PARTITION_70_1 VALUES (?,?,?,?)", totals)
    conn.commit()
    conn.close()
    return path


def make_source(path, **kwargs) -> SqlSource:
    queries = {
        "intervals": (
            "SELECT SMF_INTERVAL_START AS ts, LPAR_NAME AS lpar,"
            " REPORT_CLASS AS app_id, ENVIRONMENT AS environment,"
            " MSU_CONSUMED AS msu FROM WORKLOAD_72_3"
            " WHERE SMF_INTERVAL_START >= ? AND SMF_INTERVAL_START < ?"
        ),
        "lpar_totals": (
            "SELECT BUSINESS_DATE AS business_date, LPAR_NAME AS lpar,"
            " PEAK AS peak_mips, MEAN AS mean_mips FROM PARTITION_70_1"
            " WHERE BUSINESS_DATE >= ? AND BUSINESS_DATE < ?"
        ),
    }
    defaults = dict(
        connect=lambda: sqlite3.connect(path),
        queries=queries,
        name="sqlite",
        chunk_days=1,
        fetch_batch_rows=100,
    )
    defaults.update(kwargs)
    return SqlSource(**defaults)


# -- extraction -----------------------------------------------------------


def test_extract_aliases_columns_to_the_lake_contract(warehouse):
    source = make_source(warehouse)
    frames, report = extract_to_frames(
        source, ("intervals", "lpar_totals"), date(2025, 6, 2), date(2025, 6, 4)
    )
    intervals = frames["intervals"]
    assert {"ts", "lpar", "app_id", "environment", "msu"} <= set(intervals.columns)
    assert len(intervals) == 3 * 36 * 2
    assert report.rows_by_table["intervals"] == len(intervals)
    assert report.date_min == date(2025, 6, 2)


def test_extraction_is_chunked_not_one_big_fetch(warehouse):
    """A month per round trip, so the whole result set is never resident."""
    source = make_source(warehouse, chunk_days=1)
    batches = list(source.fetch("intervals", date(2025, 6, 2), date(2025, 6, 4)))
    assert len(batches) == 3, "one batch per day at chunk_days=1"
    assert sum(len(b) for b in batches) == 3 * 36 * 2

    single = make_source(warehouse, chunk_days=90)
    assert len(list(single.fetch("intervals", date(2025, 6, 2), date(2025, 6, 4)))) == 1


def test_chunk_windows_do_not_double_count(warehouse):
    """Half-open windows: an interval on a chunk boundary appears exactly once."""
    chunked = make_source(warehouse, chunk_days=1)
    whole = make_source(warehouse, chunk_days=365)
    a = pd.concat(list(chunked.fetch("intervals", date(2025, 6, 2), date(2025, 6, 4))))
    b = pd.concat(list(whole.fetch("intervals", date(2025, 6, 2), date(2025, 6, 4))))
    assert len(a) == len(b)
    assert a["msu"].sum() == pytest.approx(b["msu"].sum())


def test_fixed_width_padding_is_stripped(warehouse):
    """'PAYMENTS ' and 'PAYMENTS' must not become two applications."""
    frames, _ = extract_to_frames(
        make_source(warehouse), ("intervals",), date(2025, 6, 2), date(2025, 6, 4)
    )
    assert set(frames["intervals"]["app_id"]) == {"PAYMENTS", "CARDS"}


def test_dates_are_bound_as_parameters_not_formatted_in(warehouse):
    """Narrowing the window must actually narrow the result."""
    source = make_source(warehouse, chunk_days=90)
    one_day = pd.concat(list(source.fetch("intervals", date(2025, 6, 2), date(2025, 6, 2))))
    assert len(one_day) == 36 * 2
    assert set(pd.to_datetime(one_day["ts"]).dt.date) == {date(2025, 6, 2)}


def test_reconnects_per_chunk_by_default(warehouse):
    """Long extracts outlive idle-session timeouts on mainframe gateways."""
    opened = []

    def counting_connect():
        conn = sqlite3.connect(warehouse)
        opened.append(conn)
        return conn

    source = make_source(warehouse, connect=counting_connect, chunk_days=1)
    list(source.fetch("intervals", date(2025, 6, 2), date(2025, 6, 4)))
    assert len(opened) == 3


def test_missing_column_fails_loudly(warehouse):
    """Schema drift stops the run; it does not produce an empty forecast."""
    source = make_source(
        warehouse,
        queries={
            "intervals": (
                "SELECT SMF_INTERVAL_START AS ts, LPAR_NAME AS lpar, MSU_CONSUMED AS msu"
                " FROM WORKLOAD_72_3 WHERE SMF_INTERVAL_START >= ? AND SMF_INTERVAL_START < ?"
            )
        },
    )
    with pytest.raises(SchemaDriftError, match="app_id"):
        list(source.fetch("intervals", date(2025, 6, 2), date(2025, 6, 4)))


def test_intervals_without_a_workload_measure_fail(warehouse):
    source = make_source(
        warehouse,
        queries={
            "intervals": (
                "SELECT SMF_INTERVAL_START AS ts, LPAR_NAME AS lpar,"
                " REPORT_CLASS AS app_id FROM WORKLOAD_72_3"
                " WHERE SMF_INTERVAL_START >= ? AND SMF_INTERVAL_START < ?"
            )
        },
    )
    with pytest.raises(SchemaDriftError, match="msu"):
        list(source.fetch("intervals", date(2025, 6, 2), date(2025, 6, 4)))


def test_column_map_renames_when_the_sql_cannot_be_changed(warehouse):
    source = make_source(
        warehouse,
        queries={
            "intervals": (
                "SELECT SMF_INTERVAL_START, LPAR_NAME, REPORT_CLASS, MSU_CONSUMED"
                " FROM WORKLOAD_72_3 WHERE SMF_INTERVAL_START >= ? AND SMF_INTERVAL_START < ?"
            )
        },
        column_map={
            "intervals": {
                "smf_interval_start": "ts",
                "lpar_name": "lpar",
                "report_class": "app_id",
                "msu_consumed": "msu",
            }
        },
    )
    frame = next(source.fetch("intervals", date(2025, 6, 2), date(2025, 6, 4)))
    assert {"ts", "lpar", "app_id", "msu"} <= set(frame.columns)


def test_missing_lpar_totals_is_warned_about_explicitly(warehouse):
    """The most important dependency in the project fails loudly or not at all."""
    source = make_source(warehouse, queries={"intervals": make_source(warehouse).queries["intervals"]})
    _frames, report = extract_to_frames(
        source, ("intervals", "lpar_totals"), date(2025, 6, 2), date(2025, 6, 4)
    )
    assert any("simulation backtest" in w for w in report.warnings)


# -- probe ----------------------------------------------------------------


def test_probe_infers_the_interval_length(warehouse):
    """The week-1 question, answered from a few hundred sampled rows."""
    report = make_source(warehouse).probe(date(2025, 6, 2), date(2025, 6, 4))
    assert report["intervals"]["interval_minutes_mode"] == pytest.approx(15.0)
    assert report["intervals"]["distinct_app_id"] == 2


def test_probe_findings_flag_an_interval_mismatch(warehouse, cfg):
    from capplan.pipeline import _probe_findings

    # The warehouse fixture emits 15-minute intervals.
    report = make_source(warehouse).probe(date(2025, 6, 2), date(2025, 6, 4))

    mismatched = _probe_findings(report, cfg.with_overrides({"calendar.interval_minutes": 60}))
    assert any(f.startswith("FAIL intervals") and "60" in f for f in mismatched)

    matched = _probe_findings(report, cfg.with_overrides({"calendar.interval_minutes": 15}))
    assert any(f.startswith("OK intervals") for f in matched)


def test_probe_findings_flag_missing_lpar_totals(cfg):
    from capplan.pipeline import _probe_findings

    findings = _probe_findings(
        {"intervals": {"rows_sampled": 100,
                       "interval_minutes_mode": float(cfg.get("calendar.interval_minutes")),
                       "distinct_app_id": 35, "msu_min": 1.0, "msu_max": 2.0},
         "lpar_totals": {"rows_sampled": 0}},
        cfg,
    )
    assert any("FAIL lpar_totals" in f for f in findings)


def test_probe_findings_flag_a_broken_app_mapping(cfg):
    """One distinct app_id means the service-class join is not joining."""
    from capplan.pipeline import _probe_findings

    findings = _probe_findings(
        {"intervals": {"rows_sampled": 100,
                       "interval_minutes_mode": float(cfg.get("calendar.interval_minutes")),
                       "distinct_app_id": 1, "msu_min": 1.0, "msu_max": 2.0},
         "lpar_totals": {"rows_sampled": 10}},
        cfg,
    )
    assert any("APP_MAPPING" in f for f in findings)


# -- files source ---------------------------------------------------------


def test_files_source_reads_a_landing_directory(tmp_path):
    root = tmp_path / "drop"
    root.mkdir()
    stamps = pd.date_range("2025-06-02 08:00", periods=8, freq="15min")
    pd.DataFrame(
        {"ts": stamps, "app_id": "PAYMENTS", "lpar": "PRDA", "mips": 100.0}
    ).to_csv(root / "intervals_2025-06.csv", index=False)
    pd.DataFrame(
        {"business_date": ["2025-06-02"], "lpar": ["PRDA"], "peak_mips": [500.0]}
    ).to_csv(root / "lpar_totals.csv", index=False)

    source = get_source("files", root=root)
    frames, report = extract_to_frames(
        source, ("intervals", "lpar_totals"), date(2025, 6, 1), date(2025, 6, 30)
    )
    assert len(frames["intervals"]) == 8
    assert report.rows_by_table["lpar_totals"] == 1
    assert source.probe(date(2025, 6, 1), date(2025, 6, 30))["intervals"]["rows_sampled"] == 8


def test_files_source_clips_to_the_requested_window(tmp_path):
    root = tmp_path / "drop"
    root.mkdir()
    stamps = pd.date_range("2025-06-02 08:00", periods=200, freq="1D")
    pd.DataFrame(
        {"ts": stamps, "app_id": "A", "lpar": "PRDA", "mips": 1.0}
    ).to_csv(root / "intervals.csv", index=False)
    frames, _ = extract_to_frames(
        get_source("files", root=root), ("intervals",), date(2025, 6, 2), date(2025, 6, 11)
    )
    assert len(frames["intervals"]) == 10


# -- db2 wiring -----------------------------------------------------------


def test_db2_is_registered_and_reports_a_missing_driver_clearly():
    assert "db2" in available_sources()
    from capplan.data.sources.db2 import Db2Unavailable, connect, pyodbc_available

    ok, message = pyodbc_available()
    if ok:
        pytest.skip("pyodbc is installed in this environment")
    assert "pyodbc" in message and "ODBC driver" in message
    with pytest.raises(Db2Unavailable):
        connect()


def test_db2_connection_string_never_leaks_the_password(monkeypatch):
    from capplan.data.sources.db2 import connection_string

    monkeypatch.setenv("CAPPLAN_DB2_HOST", "host")
    monkeypatch.setenv("CAPPLAN_DB2_DATABASE", "db")
    monkeypatch.setenv("CAPPLAN_DB2_USER", "user")
    monkeypatch.setenv("CAPPLAN_DB2_PASSWORD", "hunter2")
    monkeypatch.delenv("CAPPLAN_DB2_DSN", raising=False)

    assert "hunter2" in connection_string()
    assert "hunter2" not in connection_string(redact=True)
    assert "PWD=***" in connection_string(redact=True)


def test_db2_names_the_missing_environment_variables(monkeypatch):
    from capplan.data.sources.db2 import Db2Unavailable, connection_string

    for name in ("DSN", "HOST", "DATABASE", "USER", "PASSWORD"):
        monkeypatch.delenv(f"CAPPLAN_DB2_{name}", raising=False)
    with pytest.raises(Db2Unavailable, match="CAPPLAN_DB2_HOST"):
        connection_string()


def test_db2_password_can_come_from_a_command(monkeypatch):
    """So a site secret store can supply it without it sitting in an env var."""
    from capplan.data.sources.db2 import connection_string

    monkeypatch.setenv("CAPPLAN_DB2_HOST", "host")
    monkeypatch.setenv("CAPPLAN_DB2_DATABASE", "db")
    monkeypatch.setenv("CAPPLAN_DB2_USER", "user")
    monkeypatch.delenv("CAPPLAN_DB2_DSN", raising=False)
    monkeypatch.delenv("CAPPLAN_DB2_PASSWORD", raising=False)
    monkeypatch.setenv("CAPPLAN_DB2_PASSWORD_CMD", "printf from-vault")
    assert "PWD=from-vault" in connection_string()


def test_shipped_query_templates_bind_exactly_two_parameters():
    """Every template takes (window start, window end) and nothing else."""
    import yaml

    config = yaml.safe_load(open("config/sources.yaml", encoding="utf-8"))
    for table, query in config["db2"]["queries"].items():
        assert query.count("?") == 2, f"{table} has {query.count('?')} parameter markers"


def test_sources_config_contains_no_credentials():
    """It is in version control. A password in git history outlives its removal."""
    text = open("config/sources.yaml", encoding="utf-8").read().lower()
    for forbidden in ("pwd=", "password:", "uid=", "passwd"):
        assert forbidden not in text, f"config/sources.yaml appears to contain {forbidden!r}"


# -- degrading gracefully on a first real extract -------------------------


def test_optional_columns_are_defaulted_not_fabricated(grid):
    """REGRESSION: a first real extract rarely carries every column.

    `peak_interval_idx` in particular needs an ARG_MAX the warehouse may not
    make easy, and failing the run over a column that feeds one diagnostic
    chart is the wrong trade. It is filled with -1 -- never a real interval --
    rather than a plausible-looking 0.
    """
    from capplan.data.schema import conform

    frame = pd.DataFrame(
        {"business_date": [date(2025, 6, 2)], "lpar": ["PRDA"], "peak_mips": [500.0]}
    )
    out = conform(frame, "lpar_totals", grid)
    assert out["peak_interval_idx"].iloc[0] == -1
    assert out["fiscal_year"].iloc[0] == grid.fiscal_year(date(2025, 6, 2))
    assert pd.isna(out["mean_mips"].iloc[0])


def test_conform_derives_business_date_from_ts(grid):
    from capplan.data.schema import conform

    out = conform(
        pd.DataFrame({"ts": pd.to_datetime(["2025-06-02 09:15"]), "app_id": ["A"], "lpar": ["P"]}),
        "intervals",
        grid,
    )
    assert out["business_date"].iloc[0] == date(2025, 6, 2)
    assert out["environment"].iloc[0] == "PROD"
    assert not out["is_anomaly"].iloc[0]


def test_diagnostics_degrade_when_optional_tables_are_absent(tmp_path, grid):
    """REGRESSION: with only intervals and lpar_totals -- the normal first
    extract -- diagnostics raised a DuckDB binder error from the middle of a
    SQL file instead of reporting what it could not compute."""
    from capplan.data.ingest import write_lake
    from capplan.data.schema import conform
    from capplan.diagnostics.runner import connect, registered_tables, run_all

    stamps = pd.date_range("2025-06-02 08:00", periods=36, freq="15min")
    intervals = pd.concat(
        [
            pd.DataFrame(
                {
                    "ts": stamps,
                    "app_id": app,
                    "lpar": "PRDA",
                    "interval_idx": range(36),
                    "mips": [100.0 + i for i in range(36)],
                }
            )
            for app in ("A", "B")
        ],
        ignore_index=True,
    )
    totals = pd.DataFrame(
        {"business_date": [date(2025, 6, 2)], "lpar": ["PRDA"], "peak_mips": [200.0]}
    )
    write_lake(
        {
            "intervals": conform(intervals, "intervals", grid),
            "lpar_totals": conform(totals, "lpar_totals", grid),
        },
        tmp_path,
    )

    con = connect(tmp_path)
    try:
        assert registered_tables(con) >= {"intervals", "lpar_totals"}
        assert "submissions" not in registered_tables(con)
    finally:
        con.close()

    result = run_all(tmp_path)   # must not raise
    assert not result.frames["coincidence_daily"].empty
    assert result.frames["submission_bias_detail"].empty


def test_coincidence_reports_rather_than_raises_without_lpar_totals(tmp_path, grid):
    from capplan.data.ingest import write_lake
    from capplan.data.schema import conform
    from capplan.diagnostics.runner import connect, run_coincidence

    stamps = pd.date_range("2025-06-02 08:00", periods=4, freq="15min")
    write_lake(
        {
            "intervals": conform(
                pd.DataFrame(
                    {"ts": stamps, "app_id": "A", "lpar": "PRDA",
                     "interval_idx": range(4), "mips": 100.0}
                ),
                "intervals",
                grid,
            )
        },
        tmp_path,
    )
    con = connect(tmp_path)
    try:
        result = run_coincidence(con)
    finally:
        con.close()
    assert result.frames["coincidence_daily"].empty
    assert not result.headline


# -- CLI error handling ----------------------------------------------------


def test_expected_failures_become_messages_not_tracebacks(capsys, tmp_path, monkeypatch):
    """Configuration and environment problems are not bugs. A traceback tells
    the user nothing they can act on and buries the one line that does."""
    from capplan.cli import main

    monkeypatch.chdir(tmp_path)
    code = main(["--config", "does-not-exist.yaml", "diagnostics"])
    assert code == 2
    err = capsys.readouterr().err
    assert "ConfigError" in err
    assert "Traceback" not in err


def test_missing_torch_is_explained_with_the_way_forward(capsys, monkeypatch):
    import importlib.util

    from capplan.cli import main

    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed in this environment")
    code = main(["--set", "model.backend=neuralforecast", "evaluate", "--folds", "1"])
    assert code == 2
    err = capsys.readouterr().err
    assert "NeuralUnavailable" in err
    assert "quantile_ridge" in err, "the message must name the working alternative"
