"""CapPlan -- prime-time peak MIPS forecasting for z/OS capacity planning.

Three stages, and the model never forecasts a peak:

    Stage 1  capplan.model   marginal predictive distribution per (app, interval)
    Stage 2  capplan.sim     dependence across time and across apps
    Stage 3  capplan.sim     sample paths, sum across apps, then reduce

The peak is a property of the simulated sum, not an output of the forecaster.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
