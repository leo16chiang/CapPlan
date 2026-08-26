"""Stage 1 training and horizon generation.

Two fiscal years out is a long way for a lag-driven model, and pretending
otherwise is the fastest route to a forecast that cannot be defended.

The calendar block (interval-of-day, day-of-week, month-end, quarter-end,
fiscal-year position, damped elapsed years) is deterministic at any horizon, so
it carries the forecast. The lag and rolling block is the hard part, and the
obvious approach does not work:

    Pure recursion -- feed the model's own median forward for 512 business days
    -- is explosive. The lag features are near-collinear, ridge leaves their
    coefficients summing above one, and the result compounds: measured at 3.2x
    the starting level by the end of the second fiscal year on data whose true
    growth is 3% a year. That is not a forecast anyone can take to a custodian.

So the lag buffer is *anchored*, not recursed:

    horizon day < recursive_days   real recursion; the recent past is genuinely
                                   informative and the buffer still contains
                                   observed values
    horizon day > recursive_days   climatology -- each app's robust profile for
                                   that interval-of-day, grown by the fitted
                                   trend. No feedback, so nothing compounds.
    in between                     linear blend, so there is no visible seam at
                                   the switchover

At 500 days out "what does this app typically do at 10:15" is not a fallback,
it is the correct lag input; the recursive alternative is just the model's own
error fed back into itself 500 times.

Anchoring the buffer fixes the explosion but introduces the mirror-image
failure: a flat anchor holds the forecast flat, and the trend cannot come back
in through `years_elapsed` because the lag features explain most of the
variance and swamp its coefficient. Measured on data growing 14% a year, the
anchored forecast came back at 1.00x over two fiscal years.

So growth is made explicit instead of implicit: a per-app annual growth rate is
estimated from the training history and the climatology anchor is grown by it.
That is also the right place for it politically -- "we have your application at
6.2% a year, here is the fit" is a sentence a custodian can agree or disagree
with, which is more than can be said for a ridge coefficient on a lag feature.

One more correction is needed, and it is easy to miss. The lag and rolling
features saw *realised* values during training, whose conditional mean sits
above their median because prime-time MIPS is right-skewed. Feeding the
predicted median forward therefore under-feeds every rolling window, and
because the fitted coefficients on `rollmean_20d` and `rollmean_60d` have
opposite signs the error does not even cancel -- it showed up as a 50%
over-forecast between horizon days 15 and 35, appearing and disappearing as the
20-day and then the 60-day window turned over. The fix is `feed_correction`: a
per-(app, interval) mean/median ratio measured on the training data, applied to
whatever goes into the buffer.

Feeding a corrected *central* value rather than a sampled draw is still
deliberate -- Stage 1 owns the marginal only, and injecting noise here would
double-count the variance Stage 2 adds.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from capplan.config import Config
from capplan.data.calendar import PrimeTimeGrid
from capplan.logging_utils import get_logger
from capplan.model.backends import QuantileBackend, build_backend
from capplan.model.features import (
    DesignMatrix,
    FeatureSpec,
    PanelIndex,
    anomaly_mask,
    app_scales,
    assemble_features,
    build_design,
    to_cube,
)
from capplan.model.forecast import ForecastCube

LOG = get_logger(__name__)


@dataclass
class Stage1Artifacts:
    """Everything Stage 2 and the evaluators need from a fit."""

    model: QuantileBackend
    design: DesignMatrix
    cube: np.ndarray
    index: PanelIndex
    scales: np.ndarray
    quantiles: np.ndarray
    spec: FeatureSpec
    grid: PrimeTimeGrid
    backend_name: str
    growth: np.ndarray = field(default=None, repr=False)
    feed_correction: np.ndarray = field(default=None, repr=False)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def max_lag(self) -> int:
        return max(max(self.spec.lags_days), max(self.spec.rolling_windows_days))


def fit_stage1(
    intervals: pd.DataFrame,
    grid: PrimeTimeGrid,
    cfg: Config,
    train_days: Sequence[date] | None = None,
    backend: str | None = None,
) -> Stage1Artifacts:
    """Fit the global interval forecaster.

    `train_days` restricts the fit to a prefix of history, which is what
    rolling-origin evaluation uses to avoid leaking the future into a fold.
    """
    spec = FeatureSpec.from_config(cfg)
    quantiles = np.asarray(cfg.get("model.quantiles"), dtype=float)
    backend_name = backend or cfg.get("model.backend")

    frame = intervals
    if train_days is not None:
        keep = set(train_days)
        frame = frame[frame["business_date"].isin(keep)]

    cube, index = to_cube(frame, grid)
    # Pin the trend epoch to the first training day and record where
    # extrapolation begins. Both travel with the artefacts so that predict time
    # sees exactly the feature definition fit time saw.
    spec.origin = index.days[0]
    spec.train_years_max = (index.days[-1] - index.days[0]).days / 365.25
    exclude = anomaly_mask(frame, index)
    scales = app_scales(cube, exclude)
    design = build_design(cube, index, grid, spec, scales=scales, exclude=exclude)

    growth = estimate_growth(cube, index.days, exclude)
    skew = feed_correction(cube, exclude)

    model = _build(backend_name, quantiles, cfg, grid)
    model.fit(design)

    metrics = in_sample_metrics(model, design, quantiles)
    LOG.info(
        "stage 1 fitted (%s): pinball %.4f, median coverage error %.3f",
        backend_name,
        metrics["pinball_mean"],
        metrics["coverage_abs_error_mean"],
    )
    return Stage1Artifacts(
        model=model,
        design=design,
        cube=cube,
        index=index,
        scales=scales,
        quantiles=quantiles,
        spec=spec,
        grid=grid,
        backend_name=backend_name,
        growth=growth,
        feed_correction=skew,
        metrics=metrics,
    )


def _build(name: str, quantiles: np.ndarray, cfg: Config, grid: PrimeTimeGrid):
    if name == "neuralforecast":
        nf = cfg.section("model").get("neuralforecast", {})
        return build_backend(
            name,
            quantiles,
            architecture=nf.get("architecture", "nhits"),
            input_size_days=int(nf.get("input_size_days", 20)),
            max_steps=int(nf.get("max_steps", 500)),
            accelerator=nf.get("accelerator", "cpu"),
            intervals_per_day=grid.intervals_per_day,
        )
    return build_backend(name, quantiles)


def in_sample_metrics(
    model: QuantileBackend, design: DesignMatrix, quantiles: np.ndarray
) -> dict[str, float]:
    """Pinball loss and empirical coverage on the fitted rows.

    In-sample coverage is not evidence of calibration -- that is what the
    conformal holdout and the rolling-origin backtest are for -- but a model
    that cannot even cover in sample is broken, and it is cheap to notice here.
    """
    X = design.X[design.valid]
    y = design.y[design.valid]
    pred = model.predict(X)
    losses = []
    coverage_err = []
    for i, tau in enumerate(quantiles):
        resid = y - pred[:, i]
        losses.append(float(np.mean(np.maximum(tau * resid, (tau - 1.0) * resid))))
        coverage_err.append(abs(float((y <= pred[:, i]).mean()) - float(tau)))
    return {
        "pinball_mean": float(np.mean(losses)),
        "pinball_median_q": float(losses[int(np.argmin(np.abs(quantiles - 0.5)))]),
        "coverage_abs_error_mean": float(np.mean(coverage_err)),
        "coverage_abs_error_max": float(np.max(coverage_err)),
        "n_rows": int(len(y)),
    }


def estimate_growth(
    cube: np.ndarray,
    days: Sequence[date],
    exclude: np.ndarray | None = None,
    clip: tuple[float, float] = (-0.15, 0.40),
) -> np.ndarray:
    """Per-app annual log growth rate, fitted on daily prime-time medians.

    Daily *medians* rather than daily means or peaks: the growth rate should
    describe the body of the workload, not be dragged around by the tail the
    simulation is there to model.

    Clipped to `clip` (default -15% to +40% a year). An app whose fitted growth
    hits the clip is not a modelling success, it is a conversation -- the
    clipped list goes into the run manifest and into the custodian pack.
    """
    work = cube.astype(float).copy()
    if exclude is not None:
        work[exclude] = np.nan
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # An application with an entirely anomalous day has no median for that
        # day. That is expected and handled by the `ok` mask below, so the
        # all-NaN-slice warning is noise rather than signal.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        daily = np.nanmedian(work, axis=2)          # (apps, days)
    years = np.array([(d - days[0]).days / 365.25 for d in days], dtype=float)

    rates = np.zeros(cube.shape[0])
    for a in range(cube.shape[0]):
        y = daily[a]
        ok = np.isfinite(y) & (y > 0)
        if ok.sum() < 30:
            continue
        slope, _ = np.polyfit(years[ok], np.log(y[ok]), 1)
        rates[a] = slope
    clipped = np.clip(rates, clip[0], clip[1])
    n_clipped = int((clipped != rates).sum())
    if n_clipped:
        LOG.warning(
            "%d app(s) had fitted growth outside %s and were clipped -- review these "
            "before the forecast goes to a custodian",
            n_clipped,
            clip,
        )
    LOG.info(
        "fitted annual growth: median %.1f%%, range %.1f%% to %.1f%%",
        100.0 * (np.exp(np.median(clipped)) - 1),
        100.0 * (np.exp(clipped.min()) - 1),
        100.0 * (np.exp(clipped.max()) - 1),
    )
    return clipped


def feed_correction(
    cube: np.ndarray, exclude: np.ndarray | None = None, clip: tuple[float, float] = (0.8, 1.6)
) -> np.ndarray:
    """Per-(app, interval) mean/median ratio of realised prime-time MIPS.

    The lag and rolling features were fitted against realised values. Their
    conditional mean exceeds their median under a right-skewed load
    distribution, so anything fed back into the buffer -- a predicted median,
    a climatology median -- has to be grossed up by this ratio or every rolling
    window drifts low and the model compensates in the wrong direction.

    Shape (apps, intervals).
    """
    work = cube.astype(float).copy()
    if exclude is not None:
        work[exclude] = np.nan
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(work, axis=1)
        median = np.nanmedian(work, axis=1)
    ratio = np.divide(mean, median, out=np.ones_like(mean), where=median > 0)
    return np.clip(np.nan_to_num(ratio, nan=1.0), clip[0], clip[1])


def climatology_profile(
    cube: np.ndarray, scales: np.ndarray, lookback_days: int = 250
) -> np.ndarray:
    """Robust per-(app, interval) profile in normalised units, shape (apps, intervals).

    Median over the most recent `lookback_days` business days -- one fiscal
    year by default, so the profile reflects the current shape of the workload
    rather than what it looked like three years ago.
    """
    recent = cube[:, -lookback_days:, :] / scales[:, None, None]
    with np.errstate(invalid="ignore"):
        profile = np.nanmedian(recent, axis=1)
    return np.nan_to_num(profile, nan=0.0)


def forecast_horizon(
    art: Stage1Artifacts,
    future_days: Sequence[date],
    recursive_days: int = 20,
    blend_days: int = 20,
    progress_every: int = 100,
) -> ForecastCube:
    """Generate marginal quantiles for every future business day.

    Recursion for the first `recursive_days`, climatology-anchored thereafter,
    linearly blended across `blend_days` in between. See the module docstring
    for why pure recursion is not an option at this horizon.

    Returns MIPS-scale quantiles. Nothing here is a peak: every number is the
    marginal distribution of one (app, interval) cell.
    """
    if not future_days:
        raise ValueError("forecast_horizon requires at least one future business day")

    n_apps, _, n_int = art.cube.shape
    n_q = len(art.quantiles)
    context = art.max_lag

    # Rolling history buffer in normalised units, most recent day last.
    normed = art.cube / art.scales[:, None, None]
    history = normed[:, -context:, :].copy()
    # Recursion cannot start from NaN: seed any missing context cell with the
    # app's own interval-of-day median, which is what a lag feature would have
    # meant anyway.
    history = _seed_nans(history)
    profile = climatology_profile(art.cube, art.scales)
    skew_correction = art.feed_correction
    growth = art.growth
    last_train_day = art.index.days[-1]

    out = np.empty((n_apps, len(future_days), n_int, n_q), dtype=np.float32)
    median_col = int(np.argmin(np.abs(art.quantiles - 0.5)))

    for step, day in enumerate(future_days):
        block, _names = assemble_features(
            history, n_days=1, days=[day], grid=art.grid, spec=art.spec, scales=art.scales
        )
        X = block.reshape(-1, block.shape[-1])
        pred = art.model.predict(np.nan_to_num(X, nan=0.0))
        pred = pred.reshape(n_apps, n_int, n_q)
        out[:, step] = pred.astype(np.float32)

        # What goes into tomorrow's lag buffer. The climatology anchor is grown
        # forward at the app's own fitted rate, damped past the training window.
        years_out = (day - last_train_day).days / 365.25
        damped = art.spec.trend_damping * years_out if art.spec.trend_damping < 1.0 else years_out
        anchor = profile * np.exp(growth * damped)[:, None]
        weight = _recursion_weight(step, recursive_days, blend_days)
        central = weight * pred[:, :, median_col] + (1.0 - weight) * anchor
        # Gross the central value up to a conditional mean before it enters the
        # buffer: that is what the rolling features were fitted against.
        fed = central * skew_correction
        history = np.concatenate([history[:, 1:, :], fed[:, None, :]], axis=1)
        if progress_every and (step + 1) % progress_every == 0:
            LOG.info("forecast horizon %d/%d days", step + 1, len(future_days))

    # De-normalise once, here, so no backend ever handles MIPS.
    out *= art.scales[:, None, None, None].astype(np.float32)
    cube = ForecastCube(
        q=out,
        quantiles=art.quantiles,
        apps=list(art.index.apps),
        days=list(future_days),
        n_intervals=n_int,
        backend=art.backend_name,
    )
    cube.enforce_monotone().clip_nonnegative()
    LOG.info("%s", cube.describe())
    return cube


def _recursion_weight(step: int, recursive_days: int, blend_days: int) -> float:
    """1.0 = feed the model's own median, 0.0 = feed climatology."""
    if step < recursive_days:
        return 1.0
    if blend_days <= 0:
        return 0.0
    return float(max(0.0, 1.0 - (step - recursive_days + 1) / blend_days))


def _seed_nans(history: np.ndarray) -> np.ndarray:
    """Replace NaN context cells with the app/interval median of the buffer."""
    if not np.isnan(history).any():
        return history
    with np.errstate(invalid="ignore"):
        per_cell = np.nanmedian(history, axis=1, keepdims=True)
    per_cell = np.nan_to_num(per_cell, nan=0.0)
    filled = np.where(np.isnan(history), np.broadcast_to(per_cell, history.shape), history)
    LOG.warning("seeded %d NaN context cells before recursion", int(np.isnan(history).sum()))
    return filled


def fitted_quantiles(art: Stage1Artifacts) -> np.ndarray:
    """In-sample predictive quantiles on the training cube.

    Stage 2 needs these: residuals are defined against the fitted median, and
    the bootstrap blocks are built from them.
    Shape (apps, days, intervals, quantiles), in MIPS.

    Rows the design marked invalid -- the lag/rolling burn-in at the start of
    history -- come back as NaN rather than as a prediction from zero-filled
    features. Zero-filling them produces plausible-looking numbers with
    near-degenerate spreads, and since Stage 2 divides by that spread, a
    handful of burn-in cells is enough to put standardised residuals in the
    hundreds and blow the simulated peak up by an order of magnitude. Ask for a
    prediction on features that do not exist and you get NaN.
    """
    n_apps, n_days, n_int = art.cube.shape
    n_q = len(art.quantiles)
    X = np.nan_to_num(art.design.X, nan=0.0)
    pred = art.model.predict(X)
    pred[~art.design.valid] = np.nan
    pred = pred.reshape(n_apps, n_days, n_int, n_q)
    return pred * art.scales[:, None, None, None]


def save(art: Stage1Artifacts, path: Path) -> Path:
    """Persist the fitted coefficients and everything needed to reproduce a run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": art.backend_name,
        "quantiles": art.quantiles,
        "scales": art.scales,
        "apps": np.array(art.index.apps, dtype=object),
        "days": np.array([d.isoformat() for d in art.index.days], dtype=object),
        "feature_names": np.array(art.design.feature_names, dtype=object),
        "growth": art.growth,
        "origin": str(art.spec.origin),
        "train_years_max": art.spec.train_years_max,
        "trend_damping": art.spec.trend_damping,
    }
    for attr in ("coef_", "mean_", "std_"):
        value = getattr(art.model, attr, None)
        if value is not None:
            payload[attr] = value
    np.savez_compressed(path, **payload)
    return path
