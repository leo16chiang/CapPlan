"""Coverage and PIT diagnostics.

Coverage answers "does the 90% interval contain 90% of outcomes". PIT answers
the sharper question: "is the *whole* predictive distribution right". A forecast
can have correct 90% coverage and still be badly wrong in the middle, and for a
distribution whose upper tail is going to be read off and turned into a purchase
order, the middle being wrong is not a detail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def coverage_table(
    observed: np.ndarray, predicted_q: np.ndarray, quantiles: np.ndarray
) -> pd.DataFrame:
    """Nominal vs empirical coverage per quantile, with a binomial standard error.

    The standard error matters: on a 60-day holdout the sampling noise on an
    empirical coverage estimate is a couple of percentage points, so a 0.88
    against a nominal 0.90 is not evidence of anything.
    """
    ok = np.isfinite(observed)
    n = int(ok.sum())
    rows = []
    for i, tau in enumerate(quantiles):
        empirical = float((observed[ok] <= predicted_q[..., i][ok]).mean())
        se = float(np.sqrt(max(tau * (1 - tau), 1e-12) / max(n, 1)))
        rows.append(
            {
                "quantile": float(tau),
                "empirical": empirical,
                "error": empirical - float(tau),
                "binomial_se": se,
                "z": (empirical - float(tau)) / se if se > 0 else np.nan,
                "n": n,
            }
        )
    return pd.DataFrame(rows)


def pit_values(observed: np.ndarray, predicted_q: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Probability integral transform of observations under the predicted CDF.

    The predicted CDF is only known at the quantile grid, so this interpolates
    between grid points. With nine quantiles that is coarse but adequate; the
    interpolation error is far smaller than the sampling error at the sizes
    involved here.
    """
    ok = np.isfinite(observed)
    y = observed[ok]
    grid = predicted_q[..., :][ok]           # (n, n_quantiles)
    out = np.empty(len(y))
    for i in range(len(y)):
        out[i] = np.interp(y[i], grid[i], quantiles, left=0.0, right=1.0)
    return out


def pit_uniformity(pit: np.ndarray) -> dict[str, float]:
    """Kolmogorov-Smirnov test of PIT uniformity, plus shape diagnostics.

    Reading the result:
      mean > 0.5   the forecast is centred too low
      var  < 1/12  the forecast is over-dispersed (intervals too wide)
      var  > 1/12  under-dispersed -- the dangerous direction here, because it
                   means the simulated peak tail is thinner than reality's
    """
    pit = pit[np.isfinite(pit)]
    if pit.size == 0:
        return {"ks_stat": float("nan"), "ks_pvalue": float("nan"), "n": 0}
    ks = stats.kstest(pit, "uniform")
    return {
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "pit_mean": float(pit.mean()),
        "pit_var": float(pit.var()),
        "uniform_var": 1.0 / 12.0,
        "dispersion": "under" if pit.var() > 1.0 / 12.0 else "over",
        "n": int(pit.size),
    }


def pinball_loss(
    observed: np.ndarray, predicted_q: np.ndarray, quantiles: np.ndarray
) -> dict[str, float]:
    """Mean pinball loss, overall and per quantile."""
    ok = np.isfinite(observed)
    y = observed[ok]
    per_quantile = {}
    for i, tau in enumerate(quantiles):
        resid = y - predicted_q[..., i][ok]
        per_quantile[f"pinball_q{tau:g}"] = float(
            np.mean(np.maximum(tau * resid, (tau - 1.0) * resid))
        )
    per_quantile["pinball_mean"] = float(np.mean(list(per_quantile.values())))
    return per_quantile


def interval_score(
    observed: np.ndarray,
    predicted_q: np.ndarray,
    quantiles: np.ndarray,
    level: float = 0.9,
) -> float:
    """Winkler interval score: width, penalised for misses.

    A single number that cannot be gamed by widening the interval, which the
    coverage table alone can be.
    """
    alpha = 1.0 - level
    lo_i = int(np.argmin(np.abs(quantiles - alpha / 2)))
    hi_i = int(np.argmin(np.abs(quantiles - (1 - alpha / 2))))
    ok = np.isfinite(observed)
    y = observed[ok]
    lo = predicted_q[..., lo_i][ok]
    hi = predicted_q[..., hi_i][ok]
    score = (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
    return float(np.mean(score))
