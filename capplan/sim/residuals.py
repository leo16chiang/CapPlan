"""Residual construction for the dependence model.

The residual is what is left after Stage 1 has explained the predictable part.
Everything the coincidence structure lives in is in here, so how the residual
is defined matters more than it usually would.

Two scalings:

`raw`
    r = y - median. Simple, and correct if variance is roughly constant.
    It is not: a 2000-MIPS app and a 20-MIPS app have very different absolute
    spreads, so raw residuals pooled across apps are dominated by the big ones.

`spread` (default)
    r = (y - median) / half the (q10, q90) width predicted for that cell. Now
    residuals are comparable across apps, across intervals-of-day, and across
    levels -- which is what makes it legitimate to resample a residual from
    last March and add it to a forecast for two Novembers hence. The forecast
    spread carries the units back.

Anomaly cells are excluded outright. A DR exercise in the residual pool would
be resampled as ordinary variation, and one historical DR test would become a
recurring feature of the forward distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np

from capplan.logging_utils import get_logger

LOG = get_logger(__name__)


@dataclass
class ResidualPanel:
    """Residuals aligned as (apps, days, intervals), plus which days are usable.

    The alignment is the whole point. Stage 2 resamples a *day* -- all apps, all
    36 intervals, simultaneously -- so the array has to be indexed such that
    `r[:, d, :]` is a coherent slice of real history. Any per-app reordering or
    per-app gap filling would destroy exactly the structure being modelled.
    """

    values: np.ndarray               # (apps, days, intervals)
    days: list[date]
    apps: list[str]
    scaling: str
    usable_days: np.ndarray          # bool (days,), fully observed and clean
    scale_used: np.ndarray | None = None

    @property
    def n_usable(self) -> int:
        return int(self.usable_days.sum())

    def usable_block(self) -> np.ndarray:
        """(usable_days, apps, intervals) -- the pool the bootstrap draws from."""
        return np.moveaxis(self.values[:, self.usable_days, :], 0, 1)

    def summary(self) -> dict[str, float]:
        block = self.usable_block()
        return {
            "n_days_total": len(self.days),
            "n_days_usable": self.n_usable,
            "scaling": self.scaling,
            "residual_mean": float(np.nanmean(block)),
            "residual_sd": float(np.nanstd(block)),
            "residual_p01": float(np.nanquantile(block, 0.01)),
            "residual_p99": float(np.nanquantile(block, 0.99)),
        }


def compute_residuals(
    observed: np.ndarray,
    predicted_q: np.ndarray,
    quantiles: np.ndarray,
    days: Sequence[date],
    apps: Sequence[str],
    anomaly: np.ndarray | None = None,
    scaling: str = "spread",
    spread_bounds: tuple[float, float] = (0.1, 0.9),
    min_spread_fraction: float = 0.02,
    min_spread_abs: float = 0.1,
    winsorise: float | None = 8.0,
) -> ResidualPanel:
    """Residuals of observed MIPS against the Stage 1 predictive distribution.

    `observed`     (apps, days, intervals)
    `predicted_q`  (apps, days, intervals, quantiles)
    """
    median_col = int(np.argmin(np.abs(quantiles - 0.5)))
    median = predicted_q[..., median_col]
    resid = observed - median

    scale_used = None
    if scaling == "spread":
        lo = int(np.argmin(np.abs(quantiles - spread_bounds[0])))
        hi = int(np.argmin(np.abs(quantiles - spread_bounds[1])))
        half_width = 0.5 * (predicted_q[..., hi] - predicted_q[..., lo])
        # Floor the divisor *relative to the level*, not absolutely. A
        # degenerate predicted interval otherwise turns a small residual into
        # an enormous standardised one, and because the bootstrap resamples
        # whole days, that single cell then dominates every path it appears in.
        # Measured: an absolute 1e-3 floor gave a residual sd of 13.5 against a
        # p01/p99 of -2.0/+3.0, and a simulated peak 23x the historical one.
        floor = np.maximum(min_spread_fraction * np.abs(median), min_spread_abs)
        scale_used = np.maximum(half_width, floor)
        resid = resid / scale_used
    elif scaling != "raw":
        raise ValueError(f"unknown residual scaling {scaling!r}")

    if anomaly is not None:
        resid = np.where(anomaly, np.nan, resid)

    if winsorise is not None and scaling == "spread":
        # A standardised residual beyond +/-8 half-widths is not a heavy tail,
        # it is a data or fit problem. Clipping bounds the damage a single
        # pathological cell can do to every path that resamples its day; the
        # count is reported because a large one means something upstream is
        # wrong and should be fixed rather than clipped.
        extreme = int(np.nansum(np.abs(resid) > winsorise))
        if extreme:
            LOG.warning(
                "winsorised %d residual cells beyond +/-%.0f half-widths (%.4f%% of "
                "the panel) -- investigate before trusting the tail",
                extreme,
                winsorise,
                100.0 * extreme / resid.size,
            )
        resid = np.clip(resid, -winsorise, winsorise)

    # A day is usable only if it is clean and complete across *every* app.
    # Dropping the whole day is the price of preserving cross-app coincidence:
    # a day with one app missing cannot be resampled as a coherent block.
    usable = np.isfinite(resid).all(axis=(0, 2))
    LOG.info(
        "residuals (%s): %d of %d days usable as aligned blocks, sd %.3f",
        scaling,
        int(usable.sum()),
        len(days),
        float(np.nanstd(resid[:, usable, :])) if usable.any() else float("nan"),
    )
    if usable.sum() < 30:
        LOG.warning(
            "only %d usable residual days -- the block bootstrap will resample a "
            "very small pool and the forward distribution will be lumpy",
            int(usable.sum()),
        )
    return ResidualPanel(
        values=resid,
        days=list(days),
        apps=list(apps),
        scaling=scaling,
        usable_days=usable,
        scale_used=scale_used,
    )


def cross_app_correlation(panel: ResidualPanel) -> np.ndarray:
    """Contemporaneous cross-app residual correlation, (apps, apps).

    Computed at the interval level: two apps are correlated if they are high in
    the *same interval*, which is the only sense in which coincidence matters
    for a peak. Daily-aggregate correlation would look reassuringly high and
    mean nothing.
    """
    block = panel.usable_block()                       # (days, apps, intervals)
    flat = np.moveaxis(block, 1, 0).reshape(block.shape[1], -1)
    return np.corrcoef(flat)


def intraday_autocorrelation(panel: ResidualPanel, max_lag: int = 8) -> np.ndarray:
    """Mean within-day residual autocorrelation by interval lag, (max_lag+1,).

    If this decays to zero within one interval, i.i.d. resampling would be
    defensible and the block structure is unnecessary. It never does -- load
    excursions last tens of minutes -- which is the empirical case for blocks.
    """
    block = panel.usable_block()                       # (days, apps, intervals)
    n_int = block.shape[2]
    out = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag >= n_int:
            break
        a = block[:, :, : n_int - lag].reshape(-1)
        b = block[:, :, lag:].reshape(-1)
        ok = np.isfinite(a) & np.isfinite(b)
        out[lag] = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else np.nan
    return out


def dependence_report(panel: ResidualPanel) -> dict[str, float]:
    """Numbers that justify the block bootstrap, for the run manifest."""
    corr = cross_app_correlation(panel)
    off_diagonal = corr[~np.eye(len(corr), dtype=bool)]
    acf = intraday_autocorrelation(panel)
    report = {
        "cross_app_corr_mean": float(np.nanmean(off_diagonal)),
        "cross_app_corr_max": float(np.nanmax(off_diagonal)),
        "cross_app_corr_p95": float(np.nanquantile(off_diagonal, 0.95)),
        "intraday_acf_lag1": float(acf[1]) if len(acf) > 1 else float("nan"),
        "intraday_acf_lag4": float(acf[4]) if len(acf) > 4 else float("nan"),
        **panel.summary(),
    }
    LOG.info(
        "dependence: mean cross-app residual correlation %.3f, intraday ACF(1) %.3f "
        "-- both non-zero, so i.i.d. resampling would understate the joint peak",
        report["cross_app_corr_mean"],
        report["intraday_acf_lag1"],
    )
    return report
