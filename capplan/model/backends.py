"""Stage 1 model backends, behind one interface.

Two are provided:

`quantile_ridge`
    Ridge-penalised linear quantile regression, fitted by IRLS on the pinball
    loss. Pure numpy, no wheels to fetch, fits ~940k x 32 in under a minute.
    It exists so the pipeline is runnable, testable and reviewable on day one,
    and so that the neural model has something honest to beat.

`neuralforecast`
    N-HiTS or TFT on PyTorch CPU, quantile loss built in, no pretrained weights
    to download. Guarded behind an optional import because torch is a large
    wheel and its presence on the internal mirror is a week-1 unknown -- see
    docs/week1_checklist.md. If it is missing, the registry says so plainly
    rather than failing halfway through a training run.

Both emit quantiles in *normalised* units (target divided by the per-app robust
scale). De-normalisation happens once, in train.py, so a backend never has to
know about MIPS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from capplan.logging_utils import get_logger
from capplan.model.features import DesignMatrix

LOG = get_logger(__name__)


@runtime_checkable
class QuantileBackend(Protocol):
    """What Stage 1 needs from a model. Nothing here mentions a peak."""

    name: str
    quantiles: np.ndarray

    def fit(self, design: DesignMatrix) -> "QuantileBackend": ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return (n_rows, n_quantiles) in normalised units."""


_REGISTRY: dict[str, type] = {}


def register(name: str):
    def wrap(cls):
        _REGISTRY[name] = cls
        cls.name = name
        return cls

    return wrap


def build_backend(name: str, quantiles: Sequence[float], **kwargs) -> QuantileBackend:
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](quantiles=np.asarray(quantiles, dtype=float), **kwargs)


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


# --------------------------------------------------------------------------
# Linear quantile regression by IRLS
# --------------------------------------------------------------------------


@register("quantile_ridge")
@dataclass
class QuantileRidge:
    """Ridge-penalised linear quantile regression via IRLS on the pinball loss.

    The pinball loss is not differentiable at zero, so IRLS smooths it: each
    iteration solves a weighted least squares problem with weights derived from
    the current residuals. The smoothing constant `epsilon` bounds the weights
    and keeps the normal equations conditioned.

    Two details matter for the tails, and both were found by checking empirical
    coverage rather than by reading the loss:

      * fixed-epsilon IRLS converges to the median quickly and to the 0.1/0.9
        quantiles slowly -- at 12 iterations the 0.9 quantile still covered
        0.78. Epsilon is therefore annealed from `epsilon_start` down to
        `epsilon`, which reaches exact coverage in about a dozen iterations
        instead of fifty.
      * quantiles are fitted in order of distance from the median, each
        warm-started from the previous solution, which keeps them close to
        non-crossing before the explicit sort in `predict`.
    """

    quantiles: np.ndarray
    alpha: float = 1.0            # ridge strength on standardised features
    max_iter: int = 40
    tol: float = 1e-6
    epsilon: float = 1e-5         # final smoothing constant
    epsilon_start: float = 0.5    # annealed down to `epsilon`
    max_rows: int = 400_000       # IRLS subsample cap; full data is not needed
    seed: int = 0
    coef_: np.ndarray = field(default=None, repr=False)
    mean_: np.ndarray = field(default=None, repr=False)
    std_: np.ndarray = field(default=None, repr=False)

    def fit(self, design: DesignMatrix) -> "QuantileRidge":
        X = design.X[design.valid]
        y = design.y[design.valid]
        rng = np.random.default_rng(self.seed)
        if len(y) > self.max_rows:
            pick = rng.choice(len(y), size=self.max_rows, replace=False)
            X, y = X[pick], y[pick]
            LOG.info("IRLS on a %d-row subsample of %d", self.max_rows, int(design.valid.sum()))

        self.mean_ = X.mean(axis=0)
        self.std_ = np.maximum(X.std(axis=0), 1e-8)
        Xs = np.hstack([np.ones((len(X), 1)), (X - self.mean_) / self.std_])

        order = np.argsort(np.abs(self.quantiles - 0.5))
        coefs = np.zeros((len(self.quantiles), Xs.shape[1]))
        warm = None
        for pos in order:
            tau = float(self.quantiles[pos])
            coefs[pos] = self._fit_one(Xs, y, tau, warm)
            warm = coefs[pos]
        self.coef_ = coefs
        LOG.info("fitted %d quantiles on %d rows x %d features", len(self.quantiles), *Xs.shape)
        return self

    def _fit_one(self, Xs: np.ndarray, y: np.ndarray, tau: float, warm) -> np.ndarray:
        n, p = Xs.shape
        beta = warm.copy() if warm is not None else _ols(Xs, y, self.alpha)
        penalty = self.alpha * np.eye(p)
        penalty[0, 0] = 0.0  # never shrink the intercept
        # Geometric annealing schedule for the smoothing constant.
        decay = (self.epsilon / self.epsilon_start) ** (1.0 / max(self.max_iter - 1, 1))
        eps = self.epsilon_start
        for _ in range(self.max_iter):
            resid = y - Xs @ beta
            absr = np.maximum(np.abs(resid), eps)
            # Pinball weight: tau above the line, (1-tau) below, scaled by 1/|r|.
            w = np.where(resid >= 0, tau, 1.0 - tau) / absr
            XtW = Xs.T * w
            new = np.linalg.solve(XtW @ Xs + penalty, XtW @ y)
            shift = np.max(np.abs(new - beta))
            beta = new
            eps = max(eps * decay, self.epsilon)
            if shift < self.tol and eps <= self.epsilon:
                break
        return beta

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("QuantileRidge.predict called before fit")
        Xs = np.hstack([np.ones((len(X), 1)), (X - self.mean_) / self.std_])
        out = Xs @ self.coef_.T
        # Independently fitted quantiles cross; sorting is the standard repair
        # and cannot degrade calibration.
        return np.sort(out, axis=1)


def _ols(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    p = X.shape[1]
    penalty = alpha * np.eye(p)
    penalty[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + penalty, X.T @ y)


# --------------------------------------------------------------------------
# Neural backend (optional dependency)
# --------------------------------------------------------------------------


class NeuralUnavailable(RuntimeError):
    """Raised when the neural backend is requested but torch is not installed."""


def neural_available() -> tuple[bool, str]:
    """Check for torch/neuralforecast without importing them eagerly.

    Called by the week-1 proxy check and by the CLI, so a missing wheel is a
    clear message at the start of a run rather than a traceback in the middle
    of one.
    """
    import importlib.util

    for module in ("torch", "neuralforecast"):
        if importlib.util.find_spec(module) is None:
            return False, (
                f"{module} is not installed. `neuralforecast` pulls torch, which is a "
                "large wheel -- confirm it is on the internal mirror "
                "(pip install 'capplan[neural]') before relying on this backend."
            )
    return True, "torch and neuralforecast are importable"


@register("neuralforecast")
@dataclass
class NeuralForecastBackend:
    """N-HiTS / TFT wrapper emitting the same quantile contract.

    Deliberately thin. The neural model's job is the marginal per (app,
    interval) and nothing more -- the joint structure is Stage 2's problem, and
    keeping that boundary sharp is what lets the backend be swapped without
    touching the simulator.
    """

    quantiles: np.ndarray
    architecture: str = "nhits"
    input_size_days: int = 20
    max_steps: int = 500
    accelerator: str = "cpu"
    intervals_per_day: int = 36
    model_: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        ok, message = neural_available()
        if not ok:
            raise NeuralUnavailable(message)

    def fit(self, design: DesignMatrix) -> "NeuralForecastBackend":
        import pandas as pd
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import MQLoss
        from neuralforecast.models import NHITS, TFT

        frame = self._to_nixtla_frame(design)
        horizon = self.intervals_per_day
        input_size = self.input_size_days * self.intervals_per_day
        loss = MQLoss(quantiles=list(self.quantiles))
        common = dict(
            h=horizon,
            input_size=input_size,
            loss=loss,
            max_steps=self.max_steps,
            scaler_type="robust",
        )
        model = NHITS(**common) if self.architecture == "nhits" else TFT(**common)
        # freq='B' is a placeholder: within prime time the series is contiguous
        # by construction, so the model sees an evenly spaced sequence and the
        # overnight/weekend gaps never enter a window.
        self.model_ = NeuralForecast(models=[model], freq="15min")
        self.model_.fit(frame)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - needs torch
        raise NotImplementedError(
            "The neural backend forecasts sequences, not design rows. Call "
            "capplan.model.train.fit_stage1 with backend='neuralforecast', which "
            "routes prediction through NeuralForecast.predict()."
        )

    def _to_nixtla_frame(self, design: DesignMatrix):  # pragma: no cover - needs torch
        import pandas as pd

        index = design.index
        n_int = index.n_intervals
        rows = []
        for a, app in enumerate(index.apps):
            mask = design.app_idx == a
            rows.append(
                pd.DataFrame(
                    {
                        "unique_id": app,
                        "ds": pd.to_datetime(
                            [index.days[d] for d in design.day_idx[mask]]
                        )
                        + pd.to_timedelta(design.interval_idx[mask] * 15, unit="m"),
                        "y": design.y[mask] * design.scales[a],
                    }
                )
            )
        return pd.concat(rows, ignore_index=True).sort_values(["unique_id", "ds"])
