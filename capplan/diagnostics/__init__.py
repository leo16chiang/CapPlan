"""Pre-model diagnostics.

Two of these are worth building before any model exists, and both are pure SQL
over data already held:

    coincidence     historical sum-of-app-peaks vs realised LPAR peak
    submission_bias per-app custodian forecast bias

Together they answer whether the rest of the project is worth doing. If peaks
effectively do sum at this site, Stage 2 is not earning its keep; if the
custodians are already unbiased, the benchmark to beat is higher than expected.
"""

from capplan.diagnostics.runner import (
    DiagnosticsResult,
    run_all,
    run_coincidence,
    run_data_profile,
    run_submission_bias,
)

__all__ = [
    "DiagnosticsResult",
    "run_all",
    "run_coincidence",
    "run_data_profile",
    "run_submission_bias",
]
