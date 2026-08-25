"""Baselines: the gate the neural model has to get through.

A global neural net over 950k rows is a defensible choice only if it beats the
things that take an afternoon. If N-HiTS cannot beat a seasonal naive with
empirical quantiles, the honest recommendation is to ship the seasonal naive
and spend the time on the coincidence model instead -- which is where the
32% error actually is.

Three levels, cheapest first:

`seasonal_naive`
    Last same-weekday value at the same interval-of-day. Quantiles come from
    the empirical distribution of that predictor's own historical errors, per
    (app, interval). No fitting at all.

`interval_climatology`
    Per-(app, interval, weekday) empirical quantiles over a trailing window,
    grown by the app's fitted trend. Surprisingly hard to beat on a two-year
    horizon, because at that range there is nothing else to know.

`sarima` / `lightgbm`
    Guarded optional imports. SARIMA is the reviewer's baseline; LightGBM with
    quantile objective is the practitioner's. Neither is required to run the
    pipeline.

Every baseline emits a ForecastCube on the same quantile grid, so
eval/rolling_origin.py scores them with the same code path as Stage 1. That
symmetry is the point: a comparison where the baseline goes through a different
evaluation route is not a comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd

from capplan.data.calendar import PrimeTimeGrid
from capplan.logging_utils import get_logger
from capplan.model.forecast import ForecastCube

LOG = get_logger(__name__)


@dataclass
class BaselineResult:
    name: str
    cube: ForecastCube
    notes: str = ""


def seasonal_naive(
    history: np.ndarray,
    hist_days: Sequence[date],
    future_days: Sequence[date],
    apps: Sequence[str],
    quantiles: Sequence[float],
    lookback_days: int = 250,
) -> BaselineResult:
    """Last same-weekday value, with empirical error quantiles.

    `history` is (apps, days, intervals) observed MIPS.

    The point forecast is trivial. The interesting half is the quantiles: the
    spread comes from how this predictor has actually behaved historically at
    that (app, interval), which is a legitimate predictive distribution and not
    a normal approximation to one.
    """
    n_apps, n_hist, n_int = history.shape
    quantiles = np.asarray(quantiles, dtype=float)

    dow_hist = np.array([d.weekday() for d in hist_days])
    recent = slice(max(0, n_hist - lookback_days), n_hist)

    # Multiplicative errors of the "same weekday last week" predictor. Ratios
    # rather than differences, so the error distribution transfers across the
    # level changes a two-year horizon will bring.
    ratios: dict[int, np.ndarray] = {}
    for dow in range(5):
        pos = [i for i in range(n_hist)[recent] if dow_hist[i] == dow]
        errs = []
        for j in range(1, len(pos)):
            prev, cur = history[:, pos[j - 1], :], history[:, pos[j], :]
            with np.errstate(divide="ignore", invalid="ignore"):
                errs.append(np.where(prev > 0, cur / prev, np.nan))
        ratios[dow] = np.stack(errs) if errs else np.ones((1, n_apps, n_int))

    last_by_dow = {}
    for dow in range(5):
        pos = [i for i in range(n_hist) if dow_hist[i] == dow]
        last_by_dow[dow] = history[:, pos[-1], :] if pos else np.nanmedian(history, axis=1)

    out = np.empty((n_apps, len(future_days), n_int, len(quantiles)), dtype=np.float32)
    for step, day in enumerate(future_days):
        dow = min(day.weekday(), 4)
        base = last_by_dow[dow]
        with np.errstate(invalid="ignore"):
            qs = np.nanquantile(ratios[dow], quantiles, axis=0)  # (q, apps, intervals)
        qs = np.nan_to_num(qs, nan=1.0)
        out[:, step] = np.moveaxis(qs * base[None, ...], 0, -1).astype(np.float32)

    cube = ForecastCube(
        q=out,
        quantiles=quantiles,
        apps=list(apps),
        days=list(future_days),
        n_intervals=n_int,
        backend="seasonal_naive",
    )
    cube.enforce_monotone().clip_nonnegative()
    return BaselineResult(
        name="seasonal_naive",
        cube=cube,
        notes=(
            "Last same-weekday value at the same interval; quantiles from the "
            "empirical week-over-week ratio distribution. No trend term -- it is "
            "expected to lose at long horizons, and by how much is the number "
            "that justifies fitting a trend at all."
        ),
    )


def interval_climatology(
    history: np.ndarray,
    hist_days: Sequence[date],
    future_days: Sequence[date],
    apps: Sequence[str],
    quantiles: Sequence[float],
    growth: np.ndarray | None = None,
    lookback_days: int = 250,
) -> BaselineResult:
    """Empirical per-(app, interval, weekday) quantiles, grown by trend.

    The one to beat. At a two-fiscal-year horizon almost all the signal is
    "what does this app normally do at this time on this weekday, and is it
    growing" -- and this computes exactly that, with no model.
    """
    n_apps, n_hist, n_int = history.shape
    quantiles = np.asarray(quantiles, dtype=float)
    dow_hist = np.array([d.weekday() for d in hist_days])
    recent = np.arange(max(0, n_hist - lookback_days), n_hist)
    growth = np.zeros(n_apps) if growth is None else growth
    last_day = hist_days[-1]

    by_dow: dict[int, np.ndarray] = {}
    for dow in range(5):
        pos = recent[dow_hist[recent] == dow]
        chunk = history[:, pos, :] if len(pos) else history[:, recent, :]
        with np.errstate(invalid="ignore"):
            by_dow[dow] = np.nan_to_num(
                np.nanquantile(chunk, quantiles, axis=1)
            )  # (q, apps, intervals)

    out = np.empty((n_apps, len(future_days), n_int, len(quantiles)), dtype=np.float32)
    for step, day in enumerate(future_days):
        years_out = (day - last_day).days / 365.25
        factor = np.exp(growth * years_out)[None, :, None]
        out[:, step] = np.moveaxis(by_dow[min(day.weekday(), 4)] * factor, 0, -1).astype(
            np.float32
        )

    cube = ForecastCube(
        q=out,
        quantiles=quantiles,
        apps=list(apps),
        days=list(future_days),
        n_intervals=n_int,
        backend="interval_climatology",
    )
    cube.enforce_monotone().clip_nonnegative()
    return BaselineResult(
        name="interval_climatology",
        cube=cube,
        notes=(
            "Empirical quantiles by (app, interval, weekday) over the trailing "
            "year, projected forward at the app's fitted growth rate. This is the "
            "gate: a neural Stage 1 that does not clear it is not earning its "
            "dependency footprint."
        ),
    )


def sarima_available() -> tuple[bool, str]:
    import importlib.util

    if importlib.util.find_spec("statsforecast") is None:
        return False, "statsforecast not installed (pip install 'capplan[baselines]')"
    return True, "statsforecast importable"


def lightgbm_available() -> tuple[bool, str]:
    import importlib.util

    if importlib.util.find_spec("lightgbm") is None:
        return False, "lightgbm not installed (pip install 'capplan[baselines]')"
    return True, "lightgbm importable"


def sarima(
    history: np.ndarray,
    hist_days: Sequence[date],
    future_days: Sequence[date],
    apps: Sequence[str],
    quantiles: Sequence[float],
    season_length: int | None = None,
) -> BaselineResult:  # pragma: no cover - optional dependency
    """Per-app SARIMA on the daily prime-time peak series.

    Fitted on the daily peak rather than the 36-per-day interval series on
    purpose: SARIMA on 950k points with a seasonal period of 36 is not a
    baseline, it is an afternoon. This is the reviewer's sanity check, and it
    answers a narrower question than Stage 1 does -- which is worth saying out
    loud when the comparison table is presented.
    """
    ok, message = sarima_available()
    if not ok:
        raise RuntimeError(message)
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    n_apps, n_hist, n_int = history.shape
    season_length = season_length or 5
    daily_peak = np.nanmax(history, axis=2)

    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": apps[a],
                    "ds": pd.to_datetime(list(hist_days)),
                    "y": daily_peak[a],
                }
            )
            for a in range(n_apps)
        ],
        ignore_index=True,
    )
    engine = StatsForecast(models=[AutoARIMA(season_length=season_length)], freq="D")
    levels = _quantiles_to_levels(quantiles)
    predicted = engine.forecast(df=frame, h=len(future_days), level=levels)
    return BaselineResult(
        name="sarima",
        cube=_daily_peak_frame_to_cube(predicted, apps, future_days, quantiles, n_int),
        notes=(
            "AutoARIMA on the daily prime-time peak per app. Forecasts a peak "
            "directly, which is exactly what the three-stage design avoids -- "
            "included because it is the comparison a reviewer will ask for."
        ),
    )


def _quantiles_to_levels(quantiles: Sequence[float]) -> list[int]:
    levels = set()
    for q in quantiles:
        if q == 0.5:
            continue
        levels.add(int(round(100 * abs(2 * q - 1))))
    return sorted(levels)


def _daily_peak_frame_to_cube(
    predicted: pd.DataFrame,
    apps: Sequence[str],
    future_days: Sequence[date],
    quantiles: Sequence[float],
    n_int: int,
) -> ForecastCube:  # pragma: no cover - optional dependency
    """Broadcast a daily-peak forecast flat across intervals.

    Deliberately crude, and labelled as such: a daily peak carries no intraday
    shape, so there is nothing honest to distribute across the 36 intervals.
    The comparison it supports is at the daily-peak level only.
    """
    quantiles = np.asarray(quantiles, dtype=float)
    out = np.zeros((len(apps), len(future_days), n_int, len(quantiles)), dtype=np.float32)
    model_col = [c for c in predicted.columns if c not in ("unique_id", "ds")][0]
    for a, app in enumerate(apps):
        rows = predicted[predicted["unique_id"] == app]
        for i, q in enumerate(quantiles):
            col = _pick_level_column(predicted.columns, model_col, float(q))
            values = rows[col].to_numpy()[: len(future_days)]
            out[a, : len(values), :, i] = values[:, None]
    cube = ForecastCube(
        q=out,
        quantiles=quantiles,
        apps=list(apps),
        days=list(future_days),
        n_intervals=n_int,
        backend="sarima",
    )
    return cube.enforce_monotone().clip_nonnegative()


def _pick_level_column(columns, model_col: str, q: float) -> str:  # pragma: no cover
    if q == 0.5:
        return model_col
    level = int(round(100 * abs(2 * q - 1)))
    suffix = "lo" if q < 0.5 else "hi"
    candidate = f"{model_col}-{suffix}-{level}"
    return candidate if candidate in columns else model_col


def lightgbm_quantile(
    design, quantiles: Sequence[float], num_leaves: int = 63, n_estimators: int = 300
):  # pragma: no cover - optional dependency
    """LightGBM with the quantile objective, one booster per quantile.

    The practitioner's baseline, and the one most likely to actually win: it
    eats the same design matrix Stage 1 uses, so a win here says the value was
    in the features, not the architecture. That is a finding, not a failure.
    """
    ok, message = lightgbm_available()
    if not ok:
        raise RuntimeError(message)
    import lightgbm as lgb

    X = design.X[design.valid]
    y = design.y[design.valid]
    models = []
    for q in quantiles:
        booster = lgb.LGBMRegressor(
            objective="quantile",
            alpha=float(q),
            num_leaves=num_leaves,
            n_estimators=n_estimators,
            verbose=-1,
        )
        booster.fit(X, y, feature_name=list(design.feature_names))
        models.append(booster)
    LOG.info("fitted %d LightGBM quantile boosters", len(models))
    return models


def run_gate(
    history: np.ndarray,
    hist_days: Sequence[date],
    future_days: Sequence[date],
    apps: Sequence[str],
    quantiles: Sequence[float],
    growth: np.ndarray | None = None,
) -> list[BaselineResult]:
    """All always-available baselines. Optional ones are skipped with a note."""
    results = [
        seasonal_naive(history, hist_days, future_days, apps, quantiles),
        interval_climatology(history, hist_days, future_days, apps, quantiles, growth),
    ]
    for name, check in (("sarima", sarima_available), ("lightgbm", lightgbm_available)):
        ok, message = check()
        if not ok:
            LOG.info("baseline %s skipped: %s", name, message)
    return results
