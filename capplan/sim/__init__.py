"""Stages 2 and 3: dependence, simulation, reduction.

Stage 1 hands over marginals. Marginals are not enough: a peak of a sum depends
entirely on whether the apps are high at the same moment, and the marginal
distributions contain no information about that at all. Stage 2 supplies the
joint structure -- across time within a day, and across apps at the same
instant. Stage 3 samples paths from it, sums across apps interval by interval,
and only then reduces to a peak.

Peak of the sum, never sum of peaks.
"""

from capplan.sim.reducers import REDUCERS, PathSummary, get_reducer, register_reducer

__all__ = ["REDUCERS", "PathSummary", "get_reducer", "register_reducer"]
