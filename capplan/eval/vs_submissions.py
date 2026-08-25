"""Compare the model against what custodians actually submitted.

The pre-model SQL diagnostic scores the custodians against realised outcomes.
This scores the *model* against the custodians, which is the comparison that
decides whether anything changes. Two ways to lose:

  1. The model is no better than the submissions. Then the honest output is the
     bias table -- de-biasing existing submissions is cheap, immediate, and
     needs no model at all.
  2. The model is better on average but worse on the handful of large apps that
     drive the LPAR. Average improvement across 35 apps is not the objective;
     the objective is the LPAR peak, and a 2,000-MIPS app getting worse while
     thirty 20-MIPS apps get better is a regression dressed as progress. Hence
     the MIPS-weighted columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from capplan.logging_utils import get_logger

LOG = get_logger(__name__)


def compare_to_submissions(
    model_app_peaks: pd.DataFrame,
    submissions: pd.DataFrame,
    realised: pd.DataFrame,
) -> pd.DataFrame:
    """Per-app comparison of model vs submission against realised peaks.

    `model_app_peaks`  app_id, fiscal_year, model_peak_mips
    `submissions`      app_id, fiscal_year, submitted_peak_mips
    `realised`         app_id, fiscal_year, realised_peak_mips
    """
    merged = (
        realised.merge(submissions, on=["app_id", "fiscal_year"], how="inner")
        .merge(model_app_peaks, on=["app_id", "fiscal_year"], how="inner")
    )
    if merged.empty:
        LOG.warning("no overlapping (app, fiscal_year) rows to compare")
        return merged

    merged["submission_error"] = merged["submitted_peak_mips"] - merged["realised_peak_mips"]
    merged["model_error"] = merged["model_peak_mips"] - merged["realised_peak_mips"]
    for side in ("submission", "model"):
        merged[f"{side}_abs_pct"] = (
            100.0 * merged[f"{side}_error"].abs() / merged["realised_peak_mips"]
        )
    merged["model_better"] = merged["model_abs_pct"] < merged["submission_abs_pct"]
    return merged.sort_values("realised_peak_mips", ascending=False)


def summarise(comparison: pd.DataFrame) -> dict[str, float]:
    """Headline numbers, unweighted and MIPS-weighted."""
    if comparison.empty:
        return {"n_apps": 0}
    weights = comparison["realised_peak_mips"].to_numpy(dtype=float)
    weights = weights / weights.sum()
    out = {
        "n_comparisons": int(len(comparison)),
        "model_better_share": float(comparison["model_better"].mean()),
        "submission_mape": float(comparison["submission_abs_pct"].mean()),
        "model_mape": float(comparison["model_abs_pct"].mean()),
        "submission_mape_mips_weighted": float(
            np.sum(weights * comparison["submission_abs_pct"].to_numpy())
        ),
        "model_mape_mips_weighted": float(
            np.sum(weights * comparison["model_abs_pct"].to_numpy())
        ),
        "submission_bias_pct": float(
            100.0 * (comparison["submission_error"] / comparison["realised_peak_mips"]).mean()
        ),
        "model_bias_pct": float(
            100.0 * (comparison["model_error"] / comparison["realised_peak_mips"]).mean()
        ),
    }
    # The comparison that decides deployment: the big apps are the LPAR.
    top = comparison.head(max(1, len(comparison) // 5))
    out["model_better_share_top_quintile"] = float(top["model_better"].mean())
    LOG.info(
        "model vs submissions: MAPE %.1f%% vs %.1f%% (MIPS-weighted %.1f%% vs %.1f%%), "
        "model better on %.0f%% of apps and %.0f%% of the top quintile",
        out["model_mape"], out["submission_mape"],
        out["model_mape_mips_weighted"], out["submission_mape_mips_weighted"],
        100 * out["model_better_share"], 100 * out["model_better_share_top_quintile"],
    )
    return out


def debiased_submissions(
    submissions: pd.DataFrame, bias_summary: pd.DataFrame
) -> pd.DataFrame:
    """Submissions corrected by each app's historical bias.

    The cheapest possible intervention and therefore the real benchmark: if the
    model cannot beat a de-biased submission, it is not worth deploying.
    """
    merged = submissions.merge(
        bias_summary[["app_id", "bias_ratio", "n_cycles"]], on="app_id", how="left"
    )
    merged["bias_ratio"] = merged["bias_ratio"].fillna(1.0)
    # One cycle is not a bias estimate, it is a single observation.
    merged.loc[merged["n_cycles"].fillna(0) < 2, "bias_ratio"] = 1.0
    merged["debiased_peak_mips"] = merged["submitted_peak_mips"] / merged["bias_ratio"]
    return merged
