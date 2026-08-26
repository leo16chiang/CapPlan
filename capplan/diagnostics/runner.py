"""DuckDB execution of the pre-model diagnostics.

DuckDB reads the parquet files in place. No load step, no server, no schema
migration -- which is the entire reason the storage layer is parquet + DuckDB
rather than something with an operations manual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from capplan import paths
from capplan.logging_utils import get_logger

LOG = get_logger(__name__)

SQL_DIR = Path(__file__).parent / "sql"


def read_sql(name: str) -> str:
    path = SQL_DIR / f"{name}.sql"
    if not path.exists():
        raise FileNotFoundError(f"no such diagnostic query: {name}")
    return path.read_text(encoding="utf-8")


def connect(root: Path = paths.LAKE_ROOT) -> duckdb.DuckDBPyConnection:
    """In-process connection with the lake registered as views.

    Records which tables are actually present. A first real extract typically
    has `intervals` and little else, and the diagnostics that need the missing
    tables should say so plainly rather than raising a binder error out of the
    middle of a SQL file.
    """
    con = duckdb.connect(":memory:")
    registrations = {
        "intervals": root / "intervals" / "intervals.parquet",
        "events": root / "events" / "events.parquet",
        "submissions": root / "submissions" / "submissions.parquet",
        "lpar_totals": root / "lpar_totals" / "lpar_totals.parquet",
    }
    for name, path in registrations.items():
        if path.exists():
            con.execute(
                f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{path.as_posix()}')"
            )
        else:
            LOG.warning("lake table %s not present at %s", name, path)
    return con


def registered_tables(con: duckdb.DuckDBPyConnection) -> set[str]:
    """Which lake views this connection actually has.

    Read from the catalog rather than tracked alongside it: a DuckDB connection
    does not accept attribute assignment, and asking the database what it holds
    cannot drift from what it holds.
    """
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    return {r[0] for r in rows}


def _has(con: duckdb.DuckDBPyConnection, *tables: str) -> bool:
    present = registered_tables(con)
    return all(t in present for t in tables)


@dataclass
class DiagnosticsResult:
    """Frames plus the handful of headline numbers worth putting in a manifest."""

    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    headline: dict[str, float] = field(default_factory=dict)

    def __getitem__(self, key: str) -> pd.DataFrame:
        return self.frames[key]

    def write(self, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for name, frame in self.frames.items():
            target = out_dir / f"{name}.parquet"
            frame.to_parquet(target, index=False)
            written.append(target)
        return written


def run_coincidence(
    con: duckdb.DuckDBPyConnection, exclude_anomalies: bool = True
) -> DiagnosticsResult:
    """Sum-of-app-peaks vs realised LPAR peak.

    The headline `coincidence_mean` is the number that decides whether Stage 2
    is load-bearing. Anything below ~0.95 means app peaks genuinely do not
    coincide and adding them up overstates the LPAR.
    """
    if not _has(con, "intervals", "lpar_totals"):
        LOG.warning(
            "coincidence needs both intervals and lpar_totals. Without realised "
            "SMF 70-1 LPAR peaks the coincidence factor cannot be measured, and "
            "measuring it is the go/no-go for this whole approach."
        )
        return DiagnosticsResult(frames={"coincidence_daily": pd.DataFrame()})
    daily = con.execute(
        read_sql("coincidence"), {"exclude_anomalies": exclude_anomalies}
    ).df()
    con.register("coincidence_daily", daily)
    summary = con.execute(read_sql("coincidence_summary")).df()
    spread = con.execute(read_sql("peak_interval_spread")).df()

    headline: dict[str, float] = {}
    if not daily.empty:
        headline = {
            "coincidence_mean": float(daily["coincidence"].mean()),
            "coincidence_p05": float(daily["coincidence"].quantile(0.05)),
            "coincidence_p95": float(daily["coincidence"].quantile(0.95)),
            "mean_overstatement_mips": float(daily["overstatement_mips"].mean()),
            "mean_overstatement_pct": float(
                100.0 * (daily["sum_app_peaks"] / daily["lpar_peak"] - 1.0).mean()
            ),
            "n_days": int(daily["business_date"].nunique()),
            # Days where the LPAR peak exceeds the sum of scoped app peaks.
            # Not noise -- unattributed prime-time load, and a scoping question
            # for the custodian interview.
            "n_unattributed_days": int(daily["unattributed"].sum()),
        }
        LOG.info(
            "coincidence factor: mean %.3f (p05 %.3f, p95 %.3f) -- "
            "summing app peaks overstates the LPAR by %.1f%% on average",
            headline["coincidence_mean"],
            headline["coincidence_p05"],
            headline["coincidence_p95"],
            headline["mean_overstatement_pct"],
        )
        if headline["n_unattributed_days"]:
            LOG.warning(
                "%d day(s) show LPAR peak above the sum of scoped app peaks -- "
                "unattributed prime-time load, check scope before trusting the model",
                headline["n_unattributed_days"],
            )
    return DiagnosticsResult(
        frames={
            "coincidence_daily": daily,
            "coincidence_summary": summary,
            "peak_interval_spread": spread,
        },
        headline=headline,
    )


def run_submission_bias(con: duckdb.DuckDBPyConnection) -> DiagnosticsResult:
    """Per-app custodian forecast bias across cycles."""
    if not _has(con, "intervals", "submissions"):
        LOG.warning(
            "no submissions table: skipping the per-app bias score. That also "
            "removes the benchmark -- without it there is nothing for the model "
            "to be better than."
        )
        return DiagnosticsResult(frames={"submission_bias_detail": pd.DataFrame()})
    detail = con.execute(read_sql("submission_bias")).df()
    if detail.empty:
        LOG.warning("no submissions joined to realised peaks; skipping bias score")
        return DiagnosticsResult(frames={"submission_bias_detail": detail})
    con.register("submission_bias_daily", detail)
    summary = con.execute(read_sql("submission_bias_summary")).df()
    headline = {
        "bias_ratio_median": float(summary["bias_ratio"].median()),
        "apps_over_forecasting": int((summary["bias_ratio"] > 1.0).sum()),
        "apps_under_forecasting": int((summary["bias_ratio"] <= 1.0).sum()),
        "worst_app": str(summary.iloc[0]["app_id"]) if len(summary) else "",
        "worst_bias_ratio": float(summary.iloc[0]["bias_ratio"]) if len(summary) else float("nan"),
        "mean_sign_consistency": float(summary["sign_consistency"].mean()),
    }
    LOG.info(
        "submission bias: median ratio %.3f, %d/%d apps over-forecasting, "
        "worst is %s at %.2fx",
        headline["bias_ratio_median"],
        headline["apps_over_forecasting"],
        len(summary),
        headline["worst_app"],
        headline["worst_bias_ratio"],
    )
    return DiagnosticsResult(
        frames={"submission_bias_detail": detail, "submission_bias_summary": summary},
        headline=headline,
    )


def run_data_profile(con: duckdb.DuckDBPyConnection) -> DiagnosticsResult:
    if not _has(con, "intervals"):
        LOG.error("no intervals table in the lake; nothing to profile")
        return DiagnosticsResult(frames={"data_profile": pd.DataFrame()})
    profile = con.execute(read_sql("data_profile")).df()
    headline: dict[str, float] = {}
    if not profile.empty:
        headline = {
            "n_apps": int(len(profile)),
            "n_days": int(profile["n_days"].max()),
            "total_intervals": int(profile["n_intervals"].sum()),
            "anomaly_share_pct": float(
                100.0 * profile["anomaly_intervals"].sum() / profile["n_intervals"].sum()
            ),
            "median_peak_to_mean": float(profile["peak_to_mean"].median()),
        }
        LOG.info(
            "profile: %d apps x %d days = %d prime-time rows, %.2f%% anomalous, "
            "median peak/mean %.2f",
            headline["n_apps"],
            headline["n_days"],
            headline["total_intervals"],
            headline["anomaly_share_pct"],
            headline["median_peak_to_mean"],
        )
    return DiagnosticsResult(frames={"data_profile": profile}, headline=headline)


def run_all(root: Path = paths.LAKE_ROOT, exclude_anomalies: bool = True) -> DiagnosticsResult:
    """Everything a go/no-go review needs, before a line of modelling code runs."""
    con = connect(root)
    try:
        merged = DiagnosticsResult()
        for part, prefix in (
            (run_data_profile(con), "profile"),
            (run_coincidence(con, exclude_anomalies), "coincidence"),
            (run_submission_bias(con), "submission"),
        ):
            merged.frames.update(part.frames)
            merged.headline.update({f"{prefix}.{k}": v for k, v in part.headline.items()})
        return merged
    finally:
        con.close()


def verdict(headline: dict[str, float], additive_threshold: float = 0.95) -> str:
    """Plain-language read on whether the three-stage architecture is justified.

    Deliberately blunt. The point of running these first is to be told to stop.
    """
    coincidence = headline.get("coincidence.coincidence_mean")
    if coincidence is None:
        return "INCONCLUSIVE: no LPAR totals to compare against; get SMF 70-1 first."
    if coincidence >= additive_threshold:
        return (
            f"STOP AND RECONSIDER: mean coincidence {coincidence:.3f} >= {additive_threshold}. "
            "App peaks effectively do coincide here, so summing them is nearly right and a "
            "joint simulation buys little. Spend the effort on the marginals instead."
        )
    overstatement = headline.get("coincidence.mean_overstatement_pct", float("nan"))
    return (
        f"PROCEED: mean coincidence {coincidence:.3f}. Summing app peaks overstates the "
        f"realised LPAR peak by {overstatement:.1f}% on average, which is exactly the error "
        "Stage 2 exists to remove."
    )
