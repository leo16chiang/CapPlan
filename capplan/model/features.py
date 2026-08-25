"""Feature construction for the global interval model.

One design decision worth stating: the model is *global*. All ~35 apps train
one parameter set, with app identity entering as a learned embedding (neural
backend) or as normalised scale plus one-hot-free target encoding (linear
backend). Per-app models on three years of data would be fitting 35 separate
thin-data problems; the whole reason there is enough data for a neural net is
that the apps are pooled.

Scale handling: each app is normalised by its own robust level before fitting,
and de-normalised afterwards. Without it the loss is dominated by the two or
three largest apps and the small ones get a flat line.

Feature groups
--------------
calendar   interval-of-day, day-of-week, month, month-end, quarter-end, FY position
fourier    smooth intraday shape and annual seasonality, cheaper than 36 dummies
lags       same interval on previous business days -- the strongest signal
rolling    per-app rolling mean/std over recent days at the same interval
trend      years elapsed, so the model can carry growth two fiscal years out
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Sequence

import numpy as np
import pandas as pd

from capplan.data.calendar import PrimeTimeGrid
from capplan.logging_utils import get_logger

LOG = get_logger(__name__)


@dataclass
class FeatureSpec:
    lags_days: Sequence[int] = (1, 2, 3, 5, 10, 20)
    rolling_windows_days: Sequence[int] = (5, 20, 60)
    fourier_terms_intraday: int = 3
    fourier_terms_annual: int = 2
    include_month_end: bool = True
    include_quarter_end: bool = True
    # Fixed epoch for `years_elapsed`. It MUST be pinned at fit time and reused
    # unchanged at predict time: derived from each block's own first day it
    # silently evaluates to zero on every forecast row, which switches the
    # trend off exactly when the trend is the only thing carrying the forecast.
    origin: date | None = None
    # `years_elapsed` at the last training day. Everything past it is
    # extrapolation, and is damped by `trend_damping`.
    train_years_max: float = 0.0
    # 1.0 = undamped linear trend. Below 1.0 the trend flattens beyond the
    # training window -- the standard defence against a linear growth term
    # compounding for two fiscal years. This is the first number a reviewer
    # will challenge, so it is a config knob, not a constant.
    trend_damping: float = 1.0

    @classmethod
    def from_config(cls, cfg) -> "FeatureSpec":
        sec = cfg.section("model")["features"]
        return cls(
            trend_damping=float(sec.get("trend_damping", 1.0)),
            lags_days=tuple(sec.get("lags_days", (1, 2, 3, 5, 10, 20))),
            rolling_windows_days=tuple(sec.get("rolling_windows_days", (5, 20, 60))),
            fourier_terms_intraday=int(sec.get("fourier_terms_intraday", 3)),
            fourier_terms_annual=int(sec.get("fourier_terms_annual", 2)),
            include_month_end=bool(sec.get("include_month_end", True)),
            include_quarter_end=bool(sec.get("include_quarter_end", True)),
        )


@dataclass
class PanelIndex:
    """The (app, day, interval) cube layout everything downstream is aligned to.

    Stage 2 resamples whole day-blocks across all apps and intervals at once,
    so a single shared index is not a convenience -- it is what makes the
    cross-app coincidence structure representable at all.
    """

    apps: list[str]
    days: list[date]
    n_intervals: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self.apps), len(self.days), self.n_intervals

    @property
    def app_pos(self) -> dict[str, int]:
        return {a: i for i, a in enumerate(self.apps)}

    @property
    def day_pos(self) -> dict[date, int]:
        return {d: i for i, d in enumerate(self.days)}

    def n_rows(self) -> int:
        return len(self.apps) * len(self.days) * self.n_intervals


def to_cube(
    intervals: pd.DataFrame, grid: PrimeTimeGrid, value: str = "mips"
) -> tuple[np.ndarray, PanelIndex]:
    """Pivot the long interval table into a dense (apps, days, intervals) cube.

    Missing cells come back as NaN, deliberately: an absent SMF interval is not
    a zero, and every consumer downstream has to decide what to do about it
    rather than inheriting a silent fill.
    """
    apps = sorted(intervals["app_id"].unique())
    days = sorted(intervals["business_date"].unique())
    n_int = grid.intervals_per_day
    index = PanelIndex(apps=list(apps), days=list(days), n_intervals=n_int)

    cube = np.full(index.shape, np.nan, dtype=np.float64)
    ai = intervals["app_id"].map(index.app_pos).to_numpy()
    di = intervals["business_date"].map(index.day_pos).to_numpy()
    ii = intervals["interval_idx"].to_numpy(dtype=int)
    cube[ai, di, ii] = intervals[value].to_numpy(dtype=float)

    filled = np.isnan(cube).sum()
    if filled:
        LOG.warning("cube has %d NaN cells (%.3f%%)", filled, 100.0 * filled / cube.size)
    return cube, index


def anomaly_mask(intervals: pd.DataFrame, index: PanelIndex) -> np.ndarray:
    """Boolean cube marking labelled anomaly cells."""
    mask = np.zeros(index.shape, dtype=bool)
    if "is_anomaly" not in intervals.columns:
        return mask
    flagged = intervals[intervals["is_anomaly"]]
    if flagged.empty:
        return mask
    ai = flagged["app_id"].map(index.app_pos).to_numpy()
    di = flagged["business_date"].map(index.day_pos).to_numpy()
    ii = flagged["interval_idx"].to_numpy(dtype=int)
    keep = ~pd.isna(ai) & ~pd.isna(di)
    mask[ai[keep].astype(int), di[keep].astype(int), ii[keep]] = True
    return mask


def app_scales(cube: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Robust per-app level used to normalise the target.

    Median of the per-app prime-time values, not the mean: one DR day should
    not move an app's scale. Anomaly cells are excluded outright when a mask is
    supplied.
    """
    work = cube.copy()
    if mask is not None:
        work[mask] = np.nan
    scales = np.nanmedian(work.reshape(work.shape[0], -1), axis=1)
    # An app that is legitimately near-zero in prime time still needs a
    # positive divisor; 1.0 MIPS is below the resolution anyone cares about.
    return np.maximum(np.nan_to_num(scales, nan=1.0), 1.0)


def _fourier(position: np.ndarray, period: float, n_terms: int) -> np.ndarray:
    cols = []
    for k in range(1, n_terms + 1):
        angle = 2.0 * np.pi * k * position / period
        cols.append(np.sin(angle))
        cols.append(np.cos(angle))
    return np.stack(cols, axis=-1) if cols else np.zeros(position.shape + (0,))


def calendar_features(
    days: Sequence[date], n_intervals: int, grid: PrimeTimeGrid, spec: FeatureSpec
) -> tuple[np.ndarray, list[str]]:
    """Calendar block, shape (days, intervals, n_features).

    Purely deterministic given the date -- which is what makes a two-fiscal-year
    horizon possible at all. Everything else in the feature set decays with
    horizon; this does not.
    """
    n_days = len(days)
    names: list[str] = []
    blocks: list[np.ndarray] = []

    interval_pos = np.tile(np.arange(n_intervals, dtype=float), (n_days, 1))
    fourier_intraday = _fourier(interval_pos, n_intervals, spec.fourier_terms_intraday)
    blocks.append(fourier_intraday)
    for k in range(1, spec.fourier_terms_intraday + 1):
        names += [f"intraday_sin{k}", f"intraday_cos{k}"]

    doy = np.array([d.timetuple().tm_yday for d in days], dtype=float)[:, None]
    doy = np.repeat(doy, n_intervals, axis=1)
    fourier_annual = _fourier(doy, 365.25, spec.fourier_terms_annual)
    blocks.append(fourier_annual)
    for k in range(1, spec.fourier_terms_annual + 1):
        names += [f"annual_sin{k}", f"annual_cos{k}"]

    dow = np.array([d.weekday() for d in days])
    dow_onehot = np.zeros((n_days, 5))
    dow_onehot[np.arange(n_days), np.clip(dow, 0, 4)] = 1.0
    blocks.append(np.repeat(dow_onehot[:, None, :], n_intervals, axis=1))
    names += [f"dow_{i}" for i in range(5)]

    if spec.include_month_end:
        me = np.array([is_month_end(grid, d, 2) for d in days], dtype=float)
        blocks.append(np.repeat(me[:, None, None], n_intervals, axis=1))
        names.append("month_end")
    if spec.include_quarter_end:
        qe = np.array(
            [float(is_month_end(grid, d, 2) and d.month in (3, 6, 9, 12)) for d in days],
            dtype=float,
        )
        blocks.append(np.repeat(qe[:, None, None], n_intervals, axis=1))
        names.append("quarter_end")

    # Position within the fiscal year: lets the model carry an FY-shaped
    # profile (year-end freeze, Q1 project load) into a future FY.
    fy_pos = np.array([_fy_position(grid, d) for d in days], dtype=float)
    blocks.append(np.repeat(fy_pos[:, None, None], n_intervals, axis=1))
    names.append("fy_position")

    origin = spec.origin or days[0]
    years = np.array([(d - origin).days / 365.25 for d in days], dtype=float)
    years = damp_trend(years, spec.train_years_max, spec.trend_damping)
    blocks.append(np.repeat(years[:, None, None], n_intervals, axis=1))
    names.append("years_elapsed")

    return np.concatenate(blocks, axis=-1), names


def damp_trend(years: np.ndarray, train_years_max: float, phi: float) -> np.ndarray:
    """Compress trend extrapolation beyond the training window.

    Inside the training window the feature is untouched. Past it, each further
    year counts as `phi` years, so a damped trend approaches a horizontal
    asymptote instead of compounding to the end of the second fiscal year.
    """
    if phi >= 1.0:
        return years
    excess = np.maximum(years - train_years_max, 0.0)
    return np.minimum(years, train_years_max) + phi * excess


def is_month_end(grid: PrimeTimeGrid, day: date, n: int = 2) -> float:
    """True when fewer than `n` business days remain in `day`'s own month.

    Derived from the calendar, never from position within whatever list of days
    happens to be in hand. Deriving it positionally is correct for a contiguous
    training block and silently wrong everywhere else: a single-day forecast
    block has no following days, so every day looks like month-end -- and every
    September day then also looks like quarter-end, which showed up as a 50%
    over-forecast for the whole of September.
    """
    remaining = 0
    probe = day + timedelta(days=1)
    while probe.month == day.month:
        if grid.is_business_day(probe):
            remaining += 1
            if remaining >= n:
                return 0.0
        probe += timedelta(days=1)
    return 1.0


def _fy_position(grid: PrimeTimeGrid, day: date) -> float:
    fy = grid.fiscal_year(day)
    start, end = grid.fiscal_year_bounds(fy)
    span = (end - start).days or 1
    return (day - start).days / span


@dataclass
class DesignMatrix:
    """Flattened training design: rows are (app, day, interval)."""

    X: np.ndarray
    y: np.ndarray
    app_idx: np.ndarray
    day_idx: np.ndarray
    interval_idx: np.ndarray
    feature_names: list[str]
    scales: np.ndarray
    index: PanelIndex
    valid: np.ndarray = field(repr=False, default=None)

    def __len__(self) -> int:
        return len(self.y)


def assemble_features(
    history: np.ndarray,
    n_days: int,
    days: Sequence[date],
    grid: PrimeTimeGrid,
    spec: FeatureSpec,
    scales: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Build the feature block for the last `n_days` of `history`.

    `history` is normalised values with shape (apps, offset + n_days,
    intervals); the trailing `n_days` slices correspond to `days`, and anything
    before them is prior context supplying lag and rolling features.

    Shared by training and by recursive forecasting so that the two cannot
    drift apart -- a feature ordering mismatch between fit and predict is the
    classic way a forecaster silently produces nonsense, and the only defence
    is that there is one function.
    """
    n_apps, total_days, n_int = history.shape
    offset = total_days - n_days
    if offset < 0:
        raise ValueError("history is shorter than the requested day block")
    if len(days) != n_days:
        raise ValueError("days does not match n_days")

    cal, cal_names = calendar_features(days, n_int, grid, spec)
    cal_rep = np.broadcast_to(cal, (n_apps,) + cal.shape)

    lag_blocks, lag_names = [], []
    for lag in spec.lags_days:
        block = np.full((n_apps, n_days, n_int), np.nan)
        for d in range(n_days):
            src = offset + d - lag
            if src >= 0:
                block[:, d, :] = history[:, src, :]
        lag_blocks.append(block)
        lag_names.append(f"lag_{lag}d")

    roll_blocks, roll_names = [], []
    for window in spec.rolling_windows_days:
        mean_block = np.full((n_apps, n_days, n_int), np.nan)
        std_block = np.full((n_apps, n_days, n_int), np.nan)
        for d in range(n_days):
            lo, hi = offset + d - window, offset + d
            if hi <= 0:
                continue
            chunk = history[:, max(lo, 0) : hi, :]
            if chunk.shape[1] == 0:
                continue
            with np.errstate(invalid="ignore"):
                mean_block[:, d, :] = np.nanmean(chunk, axis=1)
                std_block[:, d, :] = np.nanstd(chunk, axis=1)
        roll_blocks += [mean_block, std_block]
        roll_names += [f"rollmean_{window}d", f"rollstd_{window}d"]

    # Per-app level, so the pooled model can tell a 2000-MIPS app from a
    # 20-MIPS one even after normalisation flattens their scale.
    log_scale = np.broadcast_to(np.log(scales)[:, None, None], (n_apps, n_days, n_int))

    stacked = np.concatenate(
        [cal_rep]
        + [b[..., None] for b in lag_blocks]
        + [b[..., None] for b in roll_blocks]
        + [log_scale[..., None]],
        axis=-1,
    )
    names = cal_names + lag_names + roll_names + ["log_app_scale"]
    return stacked, names


def build_design(
    cube: np.ndarray,
    index: PanelIndex,
    grid: PrimeTimeGrid,
    spec: FeatureSpec,
    scales: np.ndarray | None = None,
    exclude: np.ndarray | None = None,
    lag_source: np.ndarray | None = None,
) -> DesignMatrix:
    """Assemble the global design matrix from a value cube.

    `lag_source` lets a forecasting pass supply history from before the cube's
    own first day, so lag features at the start of a horizon are real values
    rather than NaN. Shape (apps, k, intervals), most recent day last.
    """
    n_apps, n_days, n_int = cube.shape
    scales = app_scales(cube, exclude) if scales is None else scales

    work = cube.copy()
    if exclude is not None:
        work[exclude] = np.nan
    normed = work / scales[:, None, None]

    history = normed
    if lag_source is not None:
        history = np.concatenate([lag_source / scales[:, None, None], normed], axis=1)

    stacked, names = assemble_features(history, n_days, index.days, grid, spec, scales)

    n_feat = stacked.shape[-1]
    X = stacked.reshape(-1, n_feat)
    y = normed.reshape(-1)

    app_idx = np.repeat(np.arange(n_apps), n_days * n_int)
    day_idx = np.tile(np.repeat(np.arange(n_days), n_int), n_apps)
    int_idx = np.tile(np.arange(n_int), n_apps * n_days)

    valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
    LOG.info(
        "design: %d rows x %d features, %d usable (%.1f%%)",
        len(y),
        n_feat,
        int(valid.sum()),
        100.0 * valid.mean(),
    )
    return DesignMatrix(
        X=X,
        y=y,
        app_idx=app_idx,
        day_idx=day_idx,
        interval_idx=int_idx,
        feature_names=names,
        scales=scales,
        index=index,
        valid=valid,
    )
