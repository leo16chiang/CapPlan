"""Landing raw SMF/RMF extracts into the parquet lake.

Parquet + DuckDB, no warehouse. Roughly 4M rows across every table; a
`SELECT ... GROUP BY` over the whole lake finishes before you have finished
reading the query.

Scoping happens here, once, and is recorded in the manifest:
  * production LPARs only,
  * prime time only (business days, 08:00-17:00),
  * top N apps by prime-time mean MIPS.

Everything downstream assumes the lake is already scoped. That keeps the
"what's in / what's out" argument in one function instead of scattered through
the feature builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from capplan import paths
from capplan.config import Config
from capplan.data import schema
from capplan.data.calendar import PrimeTimeGrid, prime_time_frame
from capplan.data.mips_normalisation import NormalisationTable, normalise
from capplan.logging_utils import get_logger

LOG = get_logger(__name__)


@dataclass
class IngestReport:
    """What the scoping actually threw away. Goes into the run manifest."""

    rows_in: int = 0
    rows_out: int = 0
    dropped_non_prod: int = 0
    dropped_off_prime: int = 0
    dropped_not_top_app: int = 0
    missing_intervals: int = 0
    apps_kept: tuple[str, ...] = ()
    apps_dropped: int = 0
    date_min: date | None = None
    date_max: date | None = None

    def to_dict(self) -> dict:
        out = self.__dict__.copy()
        out["apps_kept"] = list(self.apps_kept)
        out["date_min"] = str(self.date_min) if self.date_min else None
        out["date_max"] = str(self.date_max) if self.date_max else None
        return out


def scope_intervals(
    raw: pd.DataFrame,
    grid: PrimeTimeGrid,
    cfg: Config,
    normalisation: NormalisationTable | None = None,
) -> tuple[pd.DataFrame, IngestReport]:
    """Apply scope rules and normalisation to a raw interval extract."""
    report = IngestReport(rows_in=len(raw))
    frame = raw.copy()
    frame["ts"] = pd.to_datetime(frame["ts"])

    environments = set(cfg.get("scope.environments", ["PROD"]))
    if "environment" in frame.columns:
        keep = frame["environment"].isin(environments)
        report.dropped_non_prod = int((~keep).sum())
        frame = frame[keep]

    lpar_include = list(cfg.get("scope.lpar_include", []) or [])
    if lpar_include:
        frame = frame[frame["lpar"].isin(lpar_include)]

    # Prime time. Vectorised rather than per-row: 4M rows is small, but not so
    # small that a Python loop over it is acceptable.
    ts = frame["ts"]
    minutes = ts.dt.hour * 60 + ts.dt.minute
    base = grid.start.hour * 60 + grid.start.minute
    end = grid.end.hour * 60 + grid.end.minute
    in_window = (minutes >= base) & (minutes < end)
    aligned = ((minutes - base) % grid.interval_minutes) == 0
    business = ts.dt.date.map(grid.is_business_day)
    keep = in_window & aligned & business
    report.dropped_off_prime = int((~keep).sum())
    frame = frame[keep].copy()

    frame["business_date"] = frame["ts"].dt.date
    frame["interval_idx"] = ((minutes[keep] - base) // grid.interval_minutes).astype("int16")
    frame["fiscal_year"] = frame["business_date"].map(grid.fiscal_year).astype("int16")

    if normalisation is not None and "msu" in frame.columns:
        frame = normalise(frame, normalisation)
    for col, default in (("capture_ratio", 1.0), ("mips_per_msu", np.nan)):
        if col not in frame.columns:
            frame[col] = default

    top_n = int(cfg.get("scope.top_n_apps", 35))
    ranked = (
        frame.groupby("app_id")["mips"].mean().sort_values(ascending=False)
    )
    kept_apps = list(ranked.index[:top_n])
    dropped_mask = ~frame["app_id"].isin(kept_apps)
    report.dropped_not_top_app = int(dropped_mask.sum())
    report.apps_dropped = int(len(ranked) - len(kept_apps))
    frame = frame[~dropped_mask].copy()
    report.apps_kept = tuple(sorted(kept_apps))

    for col, default in (("is_anomaly", False), ("event_label", None)):
        if col not in frame.columns:
            frame[col] = default

    frame = frame.sort_values(["app_id", "ts"]).reset_index(drop=True)
    report.rows_out = len(frame)
    if len(frame):
        report.date_min = frame["business_date"].min()
        report.date_max = frame["business_date"].max()
        report.missing_intervals = _count_missing(frame, grid, kept_apps)
    schema.require_columns(frame, "intervals", schema.INTERVAL_COLUMNS)
    LOG.info(
        "scoped %d -> %d rows, %d apps, %s .. %s (%d missing intervals)",
        report.rows_in,
        report.rows_out,
        len(kept_apps),
        report.date_min,
        report.date_max,
        report.missing_intervals,
    )
    return frame, report


def _count_missing(frame: pd.DataFrame, grid: PrimeTimeGrid, apps: list[str]) -> int:
    """How many (app, interval) cells the extract simply does not contain.

    A missing SMF interval and a zero-MIPS interval are different facts. The
    feature builder needs to know which it is looking at, so the count is
    surfaced rather than silently filled.
    """
    skeleton = prime_time_frame(grid, frame["business_date"].min(), frame["business_date"].max())
    expected = len(skeleton) * len(apps)
    return int(expected - len(frame))


def fill_missing_intervals(
    frame: pd.DataFrame, grid: PrimeTimeGrid, method: str = "interpolate"
) -> pd.DataFrame:
    """Complete the (app x interval) lattice, marking what was filled.

    `method='interpolate'` fills short gaps by linear interpolation within an
    app's own interval-of-day series; `method='leave'` marks them and leaves
    NaN for the feature builder to handle. Filled rows carry `is_filled=True`
    and are excluded from the residual pool -- resampling an interpolated
    interval as if it were an observation would understate variance.
    """
    if frame.empty:
        return frame.assign(is_filled=pd.Series(dtype=bool))
    skeleton = prime_time_frame(grid, frame["business_date"].min(), frame["business_date"].max())
    apps = sorted(frame["app_id"].unique())
    app_lpar = frame.drop_duplicates("app_id").set_index("app_id")["lpar"].to_dict()

    lattice = skeleton.merge(pd.DataFrame({"app_id": apps}), how="cross")
    lattice["lpar"] = lattice["app_id"].map(app_lpar)
    merged = lattice.merge(
        frame.drop(columns=["lpar", "fiscal_year", "dow", "month"], errors="ignore"),
        on=["ts", "business_date", "interval_idx", "app_id"],
        how="left",
    )
    merged["is_filled"] = merged["mips"].isna()
    if method == "interpolate":
        merged = merged.sort_values(["app_id", "interval_idx", "business_date"])
        merged["mips"] = merged.groupby(["app_id", "interval_idx"])["mips"].transform(
            lambda s: s.interpolate(limit_direction="both")
        )
    merged["environment"] = merged["environment"].fillna("PROD")
    merged["is_anomaly"] = merged["is_anomaly"].fillna(False).astype(bool)
    n_filled = int(merged["is_filled"].sum())
    if n_filled:
        LOG.warning("filled %d missing (app, interval) cells via %s", n_filled, method)
    return merged.sort_values(["app_id", "ts"]).reset_index(drop=True)


def write_lake(
    tables: dict[str, pd.DataFrame], root: Path = paths.LAKE_ROOT
) -> dict[str, Path]:
    """Write the scoped tables to parquet. Returns the paths written."""
    paths.ensure_lake(root)
    targets = {
        "intervals": root / "intervals" / "intervals.parquet",
        "events": root / "events" / "events.parquet",
        "submissions": root / "submissions" / "submissions.parquet",
        "lpar_totals": root / "lpar_totals" / "lpar_totals.parquet",
    }
    written: dict[str, Path] = {}
    for name, frame in tables.items():
        if name not in targets or frame is None:
            continue
        target = targets[name]
        out = frame.copy()
        # date objects round-trip badly through parquet; store as date32.
        for col in ("business_date", "submitted_on"):
            if col in out.columns:
                out[col] = pd.to_datetime(out[col]).dt.date
        out.to_parquet(target, index=False)
        written[name] = target
        LOG.info("wrote %s rows=%d -> %s", name, len(out), target)
    return written


def read_intervals(root: Path = paths.LAKE_ROOT) -> pd.DataFrame:
    frame = pd.read_parquet(root / "intervals" / "intervals.parquet")
    frame["ts"] = pd.to_datetime(frame["ts"])
    frame["business_date"] = pd.to_datetime(frame["business_date"]).dt.date
    return frame


def read_table(name: str, root: Path = paths.LAKE_ROOT) -> pd.DataFrame:
    mapping = {
        "intervals": ("intervals", "intervals.parquet"),
        "events": ("events", "events.parquet"),
        "submissions": ("submissions", "submissions.parquet"),
        "lpar_totals": ("lpar_totals", "lpar_totals.parquet"),
    }
    sub, filename = mapping[name]
    frame = pd.read_parquet(root / sub / filename)
    for col in ("business_date", "submitted_on"):
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col]).dt.date
    return frame


def bootstrap_synthetic_lake(
    cfg: Config, grid: PrimeTimeGrid, root: Path = paths.LAKE_ROOT, seed: int = 7
) -> tuple[dict[str, Path], IngestReport]:
    """Generate synthetic data and land it as if it had come from SMF.

    The route the pipeline takes on day one, before the real extract exists.
    """
    from capplan.data.synth import SynthSpec, generate

    spec = SynthSpec(n_apps=int(cfg.get("scope.top_n_apps", 35)), seed=seed)
    synth = generate(grid, spec)
    scoped, report = scope_intervals(
        synth["intervals"], grid, cfg, NormalisationTable.from_config(cfg)
    )
    written = write_lake(
        {
            "intervals": scoped,
            "events": synth["events"],
            "submissions": synth["submissions"],
            "lpar_totals": synth["lpar_totals"],
        },
        root,
    )
    return written, report
