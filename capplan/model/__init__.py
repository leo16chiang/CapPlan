"""Stage 1: the interval forecaster.

A global model over 15-minute prime-time series, one series per app. It emits a
*marginal* predictive distribution for every (app, future interval) -- and
nothing else. It does not forecast a peak, because a peak is a maximum and a
maximum is not something a mean-based (or even a quantile-based) point
forecaster estimates without bias when the data are thin and the tail is heavy.

The peak comes out of Stage 3, as a property of simulated paths.
"""

from capplan.model.forecast import ForecastCube

__all__ = ["ForecastCube"]
