"""Evaluation.

The part people skip is `sim_backtest`. Scoring the point forecast tells you
whether Stage 1 works. It tells you nothing at all about whether the simulated
*peak* distribution is right, and the peak distribution is the deliverable.

The question that matters: over prior fiscal years, does the realised LPAR peak
from SMF 70-1 actually fall where the simulated distribution said it would? If
the historical reconstruction of coincidence is wrong, the forward simulation
is wrong in the same direction, and nobody finds out until a hardware config
has already been signed.
"""

from capplan.eval.coverage import coverage_table, pit_values, pit_uniformity
from capplan.eval.rolling_origin import RollingOriginResult, rolling_origin
from capplan.eval.sim_backtest import SimBacktestResult, backtest_simulation
from capplan.eval.vs_submissions import compare_to_submissions

__all__ = [
    "RollingOriginResult",
    "SimBacktestResult",
    "backtest_simulation",
    "compare_to_submissions",
    "coverage_table",
    "pit_uniformity",
    "pit_values",
    "rolling_origin",
]
