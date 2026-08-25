"""Gaussian copula on PIT-transformed residuals -- the comparison model.

More principled than the block bootstrap and less defensible in the room, which
is a real tension rather than a rhetorical one:

  * It can generate coincidence patterns that never occurred historically. The
    bootstrap cannot. For a *tail* quantity two fiscal years out, that is a
    genuine advantage.
  * It assumes the dependence structure is Gaussian, so it has no tail
    dependence: extreme co-movement is systematically understated, and the
    understatement is worst exactly where a capacity plan is most exposed.
  * It requires estimating a 35x35 correlation matrix from residuals that are
    not independent across time, so the effective sample size is well below
    the nominal one. Shrinkage is not optional.

Run both. If the two agree on the fiscal-year figure, that agreement is worth
more than either number alone. If they disagree, the disagreement is the
finding, and it belongs in the pack rather than being resolved by preference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from capplan.logging_utils import get_logger
from capplan.sim.residuals import ResidualPanel

LOG = get_logger(__name__)


@dataclass
class GaussianCopula:
    """Fitted copula over the flattened (app x interval) residual vector.

    The dependence is modelled jointly across apps *and* intervals-of-day, so
    the sampled day carries intra-day shape as well as cross-app coincidence --
    the same two properties the block bootstrap gets for free. The dimension is
    apps x intervals (35 x 36 = 1260), which is far more than the ~750 days
    available, so the correlation matrix is rank-deficient by construction and
    shrinkage is doing real work rather than tidying up.
    """

    chol: np.ndarray                 # (dim, dim) lower Cholesky factor
    marginals: np.ndarray            # (dim, n_samples) empirical marginal pool
    n_apps: int
    n_intervals: int
    shrinkage: float
    eff_sample: int
    seed: int = 0

    def __post_init__(self) -> None:
        # Sorted once so the empirical inverse CDF is pure index arithmetic.
        self._sorted = np.sort(self.marginals, axis=1)

    @property
    def dim(self) -> int:
        return self.chol.shape[0]

    def draw(self, n_days: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample residual days. Returns (n_days, n_apps, n_intervals).

        Each day is drawn independently: this copula models the joint structure
        *within* a day (across apps and across intervals-of-day), which is what
        a peak depends on. Day-to-day persistence is not modelled here, and that
        is a difference from the block bootstrap with block_days > 1 rather than
        an oversight -- it is one of the things the two-model comparison exposes.
        """
        rng = rng or np.random.default_rng(self.seed)
        z = rng.standard_normal((n_days, self.dim)) @ self.chol.T
        u = stats.norm.cdf(z)
        return self._invert(u).reshape(n_days, self.n_apps, self.n_intervals)

    def _invert(self, u: np.ndarray) -> np.ndarray:
        """Vectorised empirical inverse CDF across all dimensions at once.

        The obvious `np.quantile` per dimension is 1,260 calls per simulated
        day, which at 10,000 paths x 512 days is not a slow implementation, it
        is a non-terminating one. This does the same interpolation with index
        arithmetic in one pass.
        """
        m = self._sorted.shape[1]
        pos = np.clip(u, 0.0, 1.0) * (m - 1)
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, m - 1)
        frac = (pos - lo).astype(self._sorted.dtype)
        cols = np.arange(self.dim)[None, :]
        low = self._sorted[cols, lo]
        high = self._sorted[cols, hi]
        return low + frac * (high - low)

    def stream(self, size: int, n_days: int, rng: np.random.Generator):
        """Lazy per-day view matching the block bootstrap's interface."""
        return _CopulaStream(copula=self, size=size, n_days=n_days, rng=rng)

    def diagnostics(self) -> dict[str, float]:
        return {
            "dim": self.dim,
            "shrinkage": self.shrinkage,
            "effective_sample_days": self.eff_sample,
            "condition_number": float(np.linalg.cond(self.chol @ self.chol.T)),
        }


@dataclass
class _CopulaStream:
    """Generates one day's residual slab at a time, never a whole chunk."""

    copula: "GaussianCopula"
    size: int
    n_days: int
    rng: np.random.Generator

    def day(self, d: int) -> np.ndarray:
        z = self.rng.standard_normal((self.size, self.copula.dim)) @ self.copula.chol.T
        u = stats.norm.cdf(z)
        return self.copula._invert(u).reshape(
            self.size, self.copula.n_apps, self.copula.n_intervals
        )

    def nbytes(self) -> int:
        return 0


def fit_gaussian_copula(
    panel: ResidualPanel, shrinkage: float = 0.1, seed: int = 0
) -> GaussianCopula:
    """Fit a Gaussian copula to PIT-transformed aligned residual days."""
    block = panel.usable_block()                      # (days, apps, intervals)
    n_days, n_apps, n_int = block.shape
    flat = block.reshape(n_days, n_apps * n_int)      # (days, dim)
    dim = flat.shape[1]

    if n_days <= dim:
        LOG.warning(
            "copula: %d residual days for a %d-dimensional joint. The sample "
            "correlation is rank-deficient; results lean on shrinkage=%.2f and "
            "should be treated as a comparison, not an answer.",
            n_days,
            dim,
            shrinkage,
        )

    # Probability integral transform via ranks, then to the normal scale.
    ranks = stats.rankdata(flat, axis=0)
    u = (ranks - 0.5) / n_days
    z = stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))

    corr = np.corrcoef(z, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    # Shrink toward the identity: Ledoit-Wolf in spirit, with the intensity
    # supplied rather than estimated, because the residuals are serially
    # dependent and the usual estimator assumes they are not.
    corr = (1.0 - shrinkage) * corr + shrinkage * np.eye(dim)
    corr = _nearest_positive_definite(corr)

    chol = np.linalg.cholesky(corr)
    LOG.info(
        "copula fitted: dim %d from %d residual days, shrinkage %.2f, condition %.1f",
        dim,
        n_days,
        shrinkage,
        float(np.linalg.cond(corr)),
    )
    return GaussianCopula(
        chol=chol,
        marginals=flat.T.copy(),
        n_apps=n_apps,
        n_intervals=n_int,
        shrinkage=shrinkage,
        eff_sample=n_days,
        seed=seed,
    )


def _nearest_positive_definite(matrix: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Clip eigenvalues up to `floor` and renormalise to a correlation matrix."""
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    if values.min() >= floor:
        return matrix
    values = np.maximum(values, floor)
    repaired = vectors @ np.diag(values) @ vectors.T
    scale = np.sqrt(np.diag(repaired))
    return repaired / np.outer(scale, scale)


def tail_dependence_check(panel: ResidualPanel, threshold: float = 0.9) -> dict[str, float]:
    """Empirical upper tail dependence vs what a Gaussian copula would give.

    For each app pair, the observed rate of joint exceedance above `threshold`.
    A Gaussian copula has zero asymptotic tail dependence, so if the empirical
    rate materially exceeds the Gaussian-implied one, the copula will understate
    coincident peaks -- which is the specific failure that matters here, and the
    reason the bootstrap is primary.
    """
    block = panel.usable_block()
    n_days, n_apps, n_int = block.shape
    flat = np.moveaxis(block, 1, 0).reshape(n_apps, -1)
    ranks = stats.rankdata(flat, axis=1) / flat.shape[1]
    exceed = ranks > threshold

    joint, gaussian = [], []
    corr = np.corrcoef(stats.norm.ppf(np.clip(ranks, 1e-6, 1 - 1e-6)))
    for i in range(n_apps):
        for j in range(i + 1, n_apps):
            both = float((exceed[i] & exceed[j]).mean())
            marginal = float(exceed[i].mean())
            joint.append(both / marginal if marginal > 0 else np.nan)
            gaussian.append(_gaussian_tail_dependence(corr[i, j], threshold))

    observed = float(np.nanmean(joint))
    implied = float(np.nanmean(gaussian))
    LOG.info(
        "upper tail dependence at the %.0fth percentile: observed %.3f vs "
        "Gaussian-implied %.3f",
        100 * threshold,
        observed,
        implied,
    )
    return {
        "tail_threshold": threshold,
        "observed_tail_dependence": observed,
        "gaussian_implied_tail_dependence": implied,
        "tail_dependence_gap": observed - implied,
    }


def _gaussian_tail_dependence(rho: float, threshold: float) -> float:
    """P(U_j > t | U_i > t) under a bivariate Gaussian copula."""
    rho = float(np.clip(rho, -0.999, 0.999))
    z = stats.norm.ppf(threshold)
    joint = stats.multivariate_normal.cdf(
        [-z, -z], mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]]
    )
    return float(joint / (1.0 - threshold)) if threshold < 1 else float("nan")
