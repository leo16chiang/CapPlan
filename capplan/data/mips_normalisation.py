"""MSU -> MIPS normalisation and capture ratio.

Both conversions are *applied as given*. CapPlan does not estimate the capture
ratio and does not try to improve on the systems programmers' MIPS-per-MSU
table -- that is explicitly out of scope. What it does do is record which value
was used on every row, so that when a custodian disputes a number the answer is
a lookup rather than an argument.

Convention
----------
    mips = msu * mips_per_msu / capture_ratio

`capture_ratio` is the fraction of the workload SMF attributes to the app. A
ratio below 1.0 therefore *grosses up* the measured figure. If your site's
convention is the reciprocal, flip it in config -- not here.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from capplan.config import Config
from capplan.logging_utils import get_logger

LOG = get_logger(__name__)


class NormalisationTable:
    """Per-LPAR capture ratios and per-CPU-model MIPS/MSU, with defaults."""

    def __init__(
        self,
        capture_ratio_default: float = 1.0,
        capture_ratio_by_lpar: Mapping[str, float] | None = None,
        mips_per_msu_default: float = 6.0,
        mips_per_msu_by_cpu_model: Mapping[str, float] | None = None,
    ) -> None:
        if capture_ratio_default <= 0:
            raise ValueError("capture_ratio_default must be positive")
        if mips_per_msu_default <= 0:
            raise ValueError("mips_per_msu_default must be positive")
        self.capture_ratio_default = float(capture_ratio_default)
        self.capture_ratio_by_lpar = dict(capture_ratio_by_lpar or {})
        self.mips_per_msu_default = float(mips_per_msu_default)
        self.mips_per_msu_by_cpu_model = dict(mips_per_msu_by_cpu_model or {})

    @classmethod
    def from_config(cls, cfg: Config) -> "NormalisationTable":
        sec = cfg.section("normalisation")
        return cls(
            capture_ratio_default=sec.get("capture_ratio_default", 1.0),
            capture_ratio_by_lpar=sec.get("capture_ratio_by_lpar") or {},
            mips_per_msu_default=sec.get("mips_per_msu_default", 6.0),
            mips_per_msu_by_cpu_model=sec.get("mips_per_msu_by_cpu_model") or {},
        )

    def capture_ratio_for(self, lpars: pd.Series) -> np.ndarray:
        return lpars.map(self.capture_ratio_by_lpar).fillna(self.capture_ratio_default).to_numpy(
            dtype=float
        )

    def mips_per_msu_for(self, cpu_models: pd.Series | None, n: int) -> np.ndarray:
        if cpu_models is None:
            return np.full(n, self.mips_per_msu_default, dtype=float)
        return cpu_models.map(self.mips_per_msu_by_cpu_model).fillna(
            self.mips_per_msu_default
        ).to_numpy(dtype=float)


def normalise(frame: pd.DataFrame, table: NormalisationTable) -> pd.DataFrame:
    """Add `mips`, `capture_ratio` and `mips_per_msu` columns to an MSU frame.

    An existing `mips` column is left alone -- some feeds arrive pre-converted,
    and silently re-converting them would double-count.
    """
    out = frame.copy()
    if "msu" not in out.columns:
        raise KeyError("normalise() requires an 'msu' column")

    ratios = table.capture_ratio_for(out["lpar"])
    per_msu = table.mips_per_msu_for(out.get("cpu_model"), len(out))

    if (ratios <= 0).any():
        raise ValueError("capture ratio must be strictly positive on every row")

    out["capture_ratio"] = ratios
    out["mips_per_msu"] = per_msu
    if "mips" in out.columns and out["mips"].notna().any():
        LOG.info("mips column already present; leaving %d values untouched", out["mips"].notna().sum())
        computed = out["msu"].to_numpy(dtype=float) * per_msu / ratios
        out["mips"] = out["mips"].fillna(pd.Series(computed, index=out.index))
    else:
        out["mips"] = out["msu"].to_numpy(dtype=float) * per_msu / ratios
    return out


def denormalise_to_msu(mips: np.ndarray, capture_ratio: float, mips_per_msu: float) -> np.ndarray:
    """Inverse conversion, for handing figures back in the units of the CEC report."""
    return mips * capture_ratio / mips_per_msu


def audit_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (lpar, capture_ratio, mips_per_msu) actually applied.

    Goes into the custodian pack verbatim. Every disputed number traces back to
    a row here.
    """
    cols = ["lpar", "capture_ratio", "mips_per_msu"]
    grouped = (
        frame.groupby(cols, dropna=False)
        .agg(rows=("mips", "size"), mean_mips=("mips", "mean"), max_mips=("mips", "max"))
        .reset_index()
        .sort_values("lpar")
    )
    return grouped
