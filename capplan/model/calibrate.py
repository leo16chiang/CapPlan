"""Split-conformal calibration of the Stage 1 quantiles.

Raw quantile forecasts -- neural especially, but linear ones too once they are
extrapolating -- are miscalibrated. This matters more here than in most
forecasting problems, because Stage 3 turns the marginals into a distribution
over a *maximum*, and a maximum is exactly where miscalibration accumulates: an
80% interval that really covers 70% gives a peak distribution that is wrong in
the tail, which is the only part of it anyone reads.

Method: conformalised quantile regression (CQR). On a holdout that the model
never saw, score each observation by how far outside its own predicted interval
it fell:

    E = max(q_lo - y,  y - q_hi)

and widen the interval by the (1-alpha) empirical quantile of E. Negative
scores (observations comfortably inside) let the interval *narrow*, which is
the property that makes CQR sharper than naive widening.

Guarantee: finite-sample marginal coverage, distribution-free, no assumption
about the model. The cost is one holdout window and the honesty to admit the
guarantee is marginal over the holdout distribution, not conditional on an app
or an interval-of-day. Per-app calibration recovers some of that conditioning
where an app has enough scores to support it; below `min_scores_per_app` it
falls back to the pooled adjustment rather than fitting noise.

Two methods, both split-conformal:

`per_quantile` (default)
    One-sided score per quantile, s = y - q_tau, adjusted by the tau-th
    conformal quantile of s. Corrects *location as well as width*, which
    symmetric CQR structurally cannot: an interval that is correctly sized but
    centred 2% high stays 2% high after symmetric widening. On a
    long-horizon forecast a residual location bias is the norm, not the
    exception, so this is the default.

`cqr`
    The classic symmetric interval score, max(q_lo - y, y - q_hi), applied to
    quantile pairs. Fewer parameters, retained for comparison and because it is
    the version in the literature a reviewer will look up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

import numpy as np

from capplan.config import Config
from capplan.logging_utils import get_logger
from capplan.model.forecast import ForecastCube

LOG = get_logger(__name__)


@dataclass
class ConformalAdjustment:
    """Additive per-(app, quantile) widths learned on the holdout.

    Stored in MIPS, applied by addition for upper quantiles and subtraction for
    lower ones. Kept as an explicit object rather than folded into the cube so
    that the pack generator can show exactly how much widening each app needed
    -- a large adjustment is itself a finding about that app's predictability.
    """

    quantiles: np.ndarray
    per_app: np.ndarray             # (apps, quantiles), MIPS
    pooled: np.ndarray              # (quantiles,), MIPS
    apps: list[str]
    n_scores: np.ndarray            # (apps,) holdout scores available per app
    used_pooled: np.ndarray         # (apps,) bool, True where pooled was used
    holdout_days: tuple[date, ...] = ()
    method: str = "per_quantile"

    def summary(self) -> dict[str, float]:
        return {
            "mean_abs_adjustment_mips": float(np.mean(np.abs(self.per_app))),
            "max_abs_adjustment_mips": float(np.max(np.abs(self.per_app))),
            "apps_falling_back_to_pooled": int(self.used_pooled.sum()),
            "n_holdout_days": len(self.holdout_days),
        }


def split_holdout(
    days: Sequence[date], holdout_days: int
) -> tuple[list[date], list[date]]:
    """Chronological split. Never random: a random split leaks the future.

    The holdout is the *most recent* window, so the conformal scores describe
    the model's behaviour on the regime it will actually be extrapolating from.
    """
    days = list(days)
    if holdout_days >= len(days):
        raise ValueError(f"holdout of {holdout_days} days leaves nothing to train on")
    return days[:-holdout_days], days[-holdout_days:]


def conformity_scores(
    y: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray
) -> np.ndarray:
    """CQR score: how far outside its own interval each observation fell.

    Positive means the interval was too narrow; negative means it had room to
    spare. Both directions are used.
    """
    return np.maximum(q_lo - y, y - q_hi)


def fit_conformal(
    y_holdout: np.ndarray,
    q_holdout: np.ndarray,
    quantiles: np.ndarray,
    apps: list[str],
    holdout_days: Sequence[date] = (),
    per_app: bool = True,
    min_scores_per_app: int = 200,
    method: str = "per_quantile",
) -> ConformalAdjustment:
    """Learn conformal adjustments from holdout observations and predictions.

    `y_holdout`  (apps, days, intervals) observed MIPS, NaN where unusable
    `q_holdout`  (apps, days, intervals, quantiles) predicted MIPS
    """
    if method == "per_quantile":
        return _fit_per_quantile(
            y_holdout, q_holdout, quantiles, apps, holdout_days, per_app, min_scores_per_app
        )
    if method != "cqr":
        raise ValueError(f"unknown calibration method {method!r}")
    n_apps, n_q = len(apps), len(quantiles)
    median_col = int(np.argmin(np.abs(quantiles - 0.5)))

    per_app_adj = np.zeros((n_apps, n_q))
    n_scores = np.zeros(n_apps, dtype=int)
    used_pooled = np.zeros(n_apps, dtype=bool)

    # Symmetric quantile pairs around the median: (0.05, 0.95) share one
    # interval and therefore one score set.
    pairs = _symmetric_pairs(quantiles, median_col)

    pooled = np.zeros(n_q)
    for lo_i, hi_i, alpha in pairs:
        scores = conformity_scores(
            y_holdout, q_holdout[..., lo_i], q_holdout[..., hi_i]
        )
        flat = scores[np.isfinite(scores)]
        if flat.size == 0:
            continue
        width = _conformal_quantile(flat, alpha)
        pooled[lo_i], pooled[hi_i] = -width, width

        for a in range(n_apps):
            app_scores = scores[a][np.isfinite(scores[a])]
            n_scores[a] = app_scores.size
            if per_app and app_scores.size >= min_scores_per_app:
                w = _conformal_quantile(app_scores, alpha)
            else:
                w = width
                used_pooled[a] = True
            per_app_adj[a, lo_i], per_app_adj[a, hi_i] = -w, w

    adjustment = ConformalAdjustment(
        quantiles=np.asarray(quantiles, dtype=float),
        per_app=per_app_adj,
        pooled=pooled,
        apps=list(apps),
        n_scores=n_scores,
        used_pooled=used_pooled,
        holdout_days=tuple(holdout_days),
    )
    adjustment.method = "cqr"
    LOG.info(
        "conformal (cqr): mean |adjustment| %.2f MIPS, %d/%d apps fell back to pooled",
        adjustment.summary()["mean_abs_adjustment_mips"],
        int(used_pooled.sum()),
        n_apps,
    )
    return adjustment


def _fit_per_quantile(
    y_holdout: np.ndarray,
    q_holdout: np.ndarray,
    quantiles: np.ndarray,
    apps: list[str],
    holdout_days: Sequence[date],
    per_app: bool,
    min_scores_per_app: int,
) -> ConformalAdjustment:
    """One-sided conformal correction per quantile.

    For quantile tau the score is s = y - q_tau and the correction is the
    tau-th conformal quantile of s. A systematic location error shows up as a
    correction of the same sign at every tau, which is exactly the failure mode
    symmetric CQR leaves in place.
    """
    n_apps, n_q = len(apps), len(quantiles)
    per_app_adj = np.zeros((n_apps, n_q))
    pooled = np.zeros(n_q)
    n_scores = np.zeros(n_apps, dtype=int)
    used_pooled = np.zeros(n_apps, dtype=bool)

    for i, tau in enumerate(quantiles):
        scores = y_holdout - q_holdout[..., i]
        flat = scores[np.isfinite(scores)]
        if flat.size == 0:
            continue
        pooled[i] = _conformal_quantile(flat, 1.0 - float(tau))
        for a in range(n_apps):
            app_scores = scores[a][np.isfinite(scores[a])]
            n_scores[a] = app_scores.size
            if per_app and app_scores.size >= min_scores_per_app:
                per_app_adj[a, i] = _conformal_quantile(app_scores, 1.0 - float(tau))
            else:
                per_app_adj[a, i] = pooled[i]
                used_pooled[a] = True

    adjustment = ConformalAdjustment(
        quantiles=np.asarray(quantiles, dtype=float),
        per_app=per_app_adj,
        pooled=pooled,
        apps=list(apps),
        n_scores=n_scores,
        used_pooled=used_pooled,
        holdout_days=tuple(holdout_days),
        method="per_quantile",
    )
    LOG.info(
        "conformal (per_quantile): mean |adjustment| %.2f MIPS, median location shift "
        "%.2f MIPS, %d/%d apps fell back to pooled",
        adjustment.summary()["mean_abs_adjustment_mips"],
        float(np.median(per_app_adj[:, int(np.argmin(np.abs(quantiles - 0.5)))])),
        int(used_pooled.sum()),
        n_apps,
    )
    return adjustment


def _symmetric_pairs(
    quantiles: np.ndarray, median_col: int
) -> list[tuple[int, int, float]]:
    """Pair quantiles symmetrically about the median, with their miscoverage."""
    pairs = []
    n = len(quantiles)
    for lo_i in range(median_col):
        hi_i = n - 1 - lo_i
        if hi_i <= lo_i:
            continue
        nominal = float(quantiles[hi_i] - quantiles[lo_i])
        pairs.append((lo_i, hi_i, 1.0 - nominal))
    return pairs


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The (1-alpha) conformal quantile with the finite-sample correction.

    The ceil((n+1)(1-alpha))/n rank is what buys the finite-sample guarantee;
    using the plain empirical quantile instead undercovers slightly, and at the
    99th percentile of a heavy tail "slightly" is not slight.
    """
    n = scores.size
    if n == 0:
        return 0.0
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def apply_conformal(cube: ForecastCube, adjustment: ConformalAdjustment) -> ForecastCube:
    """Widen (or narrow) a forecast cube by the learned conformal amounts."""
    if list(cube.apps) != list(adjustment.apps):
        raise ValueError("conformal adjustment was fitted on a different app set")
    if not np.allclose(cube.quantiles, adjustment.quantiles):
        raise ValueError("conformal adjustment was fitted on a different quantile grid")
    cube.q = cube.q + adjustment.per_app[:, None, None, :].astype(np.float32)
    cube.calibrated = True
    # Widening can reorder the tails and can push a lower quantile negative.
    return cube.enforce_monotone().clip_nonnegative()


def calibrate_stage1(
    art,
    intervals,
    cfg: Config,
    forecast_fn=None,
) -> tuple[ConformalAdjustment, dict[str, float]]:
    """End-to-end split-conformal calibration.

    Refits Stage 1 on the pre-holdout prefix, forecasts the holdout window,
    and scores it. Refitting rather than reusing the full-data fit is the whole
    point: scoring a model on data it was trained on measures nothing.
    """
    from capplan.model.train import fit_stage1, forecast_horizon

    holdout_n = int(cfg.get("calibration.holdout_days"))
    train_days, holdout_days = split_holdout(art.index.days, holdout_n)
    LOG.info(
        "conformal split: train %s .. %s, holdout %s .. %s",
        train_days[0], train_days[-1], holdout_days[0], holdout_days[-1],
    )

    inner = fit_stage1(intervals, art.grid, cfg, train_days=train_days)
    forecaster = forecast_fn or forecast_horizon
    predicted = forecaster(inner, holdout_days, progress_every=0)

    observed = _observed_cube(intervals, holdout_days, art.index.apps, art.grid)
    adjustment = fit_conformal(
        y_holdout=observed,
        q_holdout=predicted.q.astype(float),
        quantiles=art.quantiles,
        apps=list(art.index.apps),
        holdout_days=holdout_days,
        per_app=bool(cfg.get("calibration.per_app", True)),
        min_scores_per_app=int(cfg.get("calibration.min_scores_per_app", 200)),
        method=str(cfg.get("calibration.method", "per_quantile")),
    )
    before = empirical_coverage(observed, predicted.q.astype(float), art.quantiles)
    calibrated = apply_conformal(predicted, adjustment)
    after = empirical_coverage(observed, calibrated.q.astype(float), art.quantiles)

    metrics = {
        "coverage_before": before.tolist(),
        "coverage_after": after.tolist(),
        "coverage_abs_error_before": float(np.mean(np.abs(before - art.quantiles))),
        # NOTE: this is measured on the same holdout the adjustment was fitted
        # on, so it is a sanity check, not evidence. The honest number comes
        # from eval/rolling_origin.py on a fold the adjustment never saw.
        "coverage_abs_error_after_insample": float(np.mean(np.abs(after - art.quantiles))),
        **adjustment.summary(),
    }
    LOG.info(
        "coverage abs error %.4f -> %.4f (in-sample on the calibration holdout)",
        metrics["coverage_abs_error_before"],
        metrics["coverage_abs_error_after_insample"],
    )
    return adjustment, metrics


def empirical_coverage(
    observed: np.ndarray, predicted: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    """Share of observations at or below each predicted quantile."""
    ok = np.isfinite(observed)
    out = np.empty(len(quantiles))
    for i in range(len(quantiles)):
        out[i] = float((observed[ok] <= predicted[..., i][ok]).mean())
    return out


def _observed_cube(intervals, days: Sequence[date], apps: Sequence[str], grid) -> np.ndarray:
    """Observed MIPS for a day window, NaN on anomaly cells."""
    import pandas as pd

    day_pos = {d: i for i, d in enumerate(days)}
    app_pos = {a: i for i, a in enumerate(apps)}
    out = np.full((len(apps), len(days), grid.intervals_per_day), np.nan)
    frame = intervals[intervals["business_date"].isin(day_pos)]
    if "is_anomaly" in frame.columns:
        frame = frame[~frame["is_anomaly"]]
    ai = frame["app_id"].map(app_pos).to_numpy()
    di = frame["business_date"].map(day_pos).to_numpy()
    ii = frame["interval_idx"].to_numpy(dtype=int)
    keep = ~pd.isna(ai) & ~pd.isna(di)
    out[ai[keep].astype(int), di[keep].astype(int), ii[keep]] = frame["mips"].to_numpy()[keep]
    return out
