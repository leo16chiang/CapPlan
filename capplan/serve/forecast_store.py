"""Parquet forecast store for the existing Dash app.

Deliberately denormalised and deliberately small. Dash gets three tables:

  `fy_summary`     one row per (fiscal_year, reducer, quantile). Tens of rows.
                   This is the headline the pack is built from.
  `app_peaks`      one row per (app, fiscal_year, quantile). ~35 x 2 x 4.
                   The per-app numbers a custodian asks about.
  `daily_profile`  one row per (business_date, quantile) of the LPAR total's
                   daily peak. ~512 x 4 rows -- enough for a time-series chart
                   without shipping paths.

What is *not* published: the paths. Ten thousand paths x 512 days is 20MB of
float that no dashboard can usefully render, and publishing it invites someone
to re-derive a peak from it with the wrong reduction. The reduction happens
once, on the simulation side, where the reducer registry makes the convention
explicit.

Every table carries `run_id`, so a number on a slide can be traced to the run
that produced it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from capplan import paths
from capplan.logging_utils import get_logger
from capplan.sim.simulate import SimulationResult

LOG = get_logger(__name__)

DEFAULT_LEVELS = (0.5, 0.9, 0.95, 0.99)


def build_fy_summary(
    result: SimulationResult, levels: Sequence[float] = DEFAULT_LEVELS, run_id: str = ""
) -> pd.DataFrame:
    rows = []
    for reducer, by_fy in result.reducer_by_fy.items():
        for fy, values in sorted(by_fy.items()):
            for level in levels:
                rows.append(
                    {
                        "run_id": run_id,
                        "fiscal_year": int(fy),
                        "reducer": reducer,
                        "quantile": float(level),
                        "peak_mips": float(np.quantile(values, level)),
                        "mean_mips": float(values.mean()),
                        "n_paths": result.n_paths,
                        "dependence": result.dependence,
                    }
                )
    return pd.DataFrame(rows)


def build_app_peaks(
    result: SimulationResult,
    levels: Sequence[float] = DEFAULT_LEVELS,
    run_id: str = "",
) -> pd.DataFrame:
    """Per-app peak distribution over the whole horizon.

    Carries a loud caveat column. These are *marginal* app peaks: each is the
    maximum that app reaches somewhere in the horizon, and they land in
    different intervals. Adding them up gives the sum-of-peaks number the whole
    architecture exists to avoid, and someone will try, so the constraint is
    published next to the data rather than left in a document.
    """
    rows = []
    for a, app in enumerate(result.apps):
        values = result.app_peaks[:, a]
        for level in levels:
            rows.append(
                {
                    "run_id": run_id,
                    "app_id": app,
                    "quantile": float(level),
                    "app_peak_mips": float(np.quantile(values, level)),
                    "do_not_sum": True,
                    "note": "marginal app peak; apps peak in different intervals",
                }
            )
    return pd.DataFrame(rows)


def build_daily_profile(
    result: SimulationResult, levels: Sequence[float] = DEFAULT_LEVELS, run_id: str = ""
) -> pd.DataFrame:
    quantile_matrix = np.quantile(result.daily_peaks, levels, axis=0)  # (levels, days)
    frames = []
    for i, level in enumerate(levels):
        frames.append(
            pd.DataFrame(
                {
                    "run_id": run_id,
                    "business_date": result.days,
                    "fiscal_year": result.fiscal_years,
                    "quantile": float(level),
                    "daily_peak_mips": quantile_matrix[i],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def publish(
    result: SimulationResult,
    run_id: str,
    out_dir: Path | None = None,
    levels: Sequence[float] = DEFAULT_LEVELS,
) -> dict[str, Path]:
    """Write the three serving tables. Overwrites in place -- Dash reads latest."""
    out_dir = Path(out_dir or paths.SERVING)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "fy_summary": build_fy_summary(result, levels, run_id),
        "app_peaks": build_app_peaks(result, levels, run_id),
        "daily_profile": build_daily_profile(result, levels, run_id),
    }
    written = {}
    for name, frame in tables.items():
        target = out_dir / f"{name}.parquet"
        frame.to_parquet(target, index=False)
        written[name] = target
        LOG.info("published %s rows=%d -> %s", name, len(frame), target)

    manifest = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "published_at": datetime.now(timezone.utc).isoformat(),
                "n_paths": result.n_paths,
                "dependence": result.dependence,
                "horizon_start": result.days[0],
                "horizon_end": result.days[-1],
                "reducers": ",".join(sorted(result.reducer_values)),
            }
        ]
    )
    manifest_path = out_dir / "published_manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    written["manifest"] = manifest_path
    return written


def read_published(name: str, out_dir: Path | None = None) -> pd.DataFrame:
    out_dir = Path(out_dir or paths.SERVING)
    frame = pd.read_parquet(out_dir / f"{name}.parquet")
    if "business_date" in frame.columns:
        frame["business_date"] = pd.to_datetime(frame["business_date"]).dt.date
    return frame
