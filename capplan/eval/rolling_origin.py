"""Rolling-origin evaluation of Stage 1 and the baselines.

Scores the *marginals*. Necessary but not sufficient: a model can win here and
still produce a wrong peak distribution, because the peak depends on the joint
structure this never touches. sim_backtest.py is the one that tests the thing
being sold; this one tests the thing Stage 1 is responsible for.

Every model -- Stage 1, seasonal naive, interval climatology -- goes through the
same scoring path, because a comparison in which the baseline takes a different
route is not a comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd

from capplan.config import Config
from capplan.data.calendar import PrimeTimeGrid
from capplan.eval.coverage import coverage_table, interval_score, pinball_loss, pit_uniformity, pit_values
from capplan.logging_utils import get_logger
from capplan.model.baselines import run_gate
from capplan.model.calibrate import _observed_cube
from capplan.model.train import fit_stage1, forecast_horizon

LOG = get_logger(__name__)


@dataclass
class RollingOriginResult:
    scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    coverage: pd.DataFrame = field(default_factory=pd.DataFrame)

    def leaderboard(self) -> pd.DataFrame:
        """Mean score per model across folds, best pinball first."""
        if self.scores.empty:
            return self.scores
        return (
            self.scores.groupby("model")[
                ["pinball_mean", "interval_score_90", "pit_ks_stat", "coverage_abs_error"]
            ]
            .mean()
            .sort_values("pinball_mean")
            .reset_index()
        )

    def gate_verdict(self, champion: str = "stage1") -> str:
        """Did Stage 1 actually earn its dependency footprint?"""
        board = self.leaderboard()
        if board.empty or champion not in set(board["model"]):
            return "INCONCLUSIVE: no folds scored."
        best = board.iloc[0]["model"]
        champion_score = float(board.loc[board["model"] == champion, "pinball_mean"].iloc[0])
        best_score = float(board.iloc[0]["pinball_mean"])
        if best == champion:
            runner_up = board.iloc[1] if len(board) > 1 else None
            margin = (
                100.0 * (float(runner_up["pinball_mean"]) - champion_score) / champion_score
                if runner_up is not None
                else float("nan")
            )
            return (
                f"GATE PASSED: {champion} leads on pinball by {margin:.1f}% over "
                f"{runner_up['model'] if runner_up is not None else 'nothing'}."
            )
        return (
            f"GATE FAILED: {best} beats {champion} on pinball "
            f"({best_score:.4f} vs {champion_score:.4f}). Ship the baseline and spend "
            "the time on the coincidence model, which is where the larger error is."
        )


def rolling_origin(
    intervals: pd.DataFrame,
    grid: PrimeTimeGrid,
    cfg: Config,
    n_folds: int | None = None,
    fold_step_days: int | None = None,
    horizon_days: int = 60,
    include_baselines: bool = True,
) -> RollingOriginResult:
    """Score Stage 1 and the baselines over expanding-window origins."""
    n_folds = n_folds or int(cfg.get("evaluation.rolling_origin.n_folds", 4))
    fold_step_days = fold_step_days or int(cfg.get("evaluation.rolling_origin.fold_step_days", 60))
    min_train = int(cfg.get("evaluation.rolling_origin.min_train_days", 250))
    quantiles = np.asarray(cfg.get("model.quantiles"), dtype=float)

    all_days = sorted(intervals["business_date"].unique())
    origins = _origins(all_days, n_folds, fold_step_days, horizon_days, min_train)
    if not origins:
        LOG.warning("not enough history for a rolling-origin evaluation")
        return RollingOriginResult()

    score_rows, coverage_rows = [], []
    for fold, origin_pos in enumerate(origins):
        train_days = all_days[: origin_pos + 1]
        eval_days = all_days[origin_pos + 1 : origin_pos + 1 + horizon_days]
        LOG.info(
            "fold %d/%d: train through %s (%d days), evaluate %d days",
            fold + 1, len(origins), train_days[-1], len(train_days), len(eval_days),
        )
        train_frame = intervals[intervals["business_date"].isin(set(train_days))]
        art = fit_stage1(train_frame, grid, cfg)
        observed = _observed_cube(intervals, eval_days, art.index.apps, grid)

        candidates = {"stage1": forecast_horizon(art, eval_days, progress_every=0)}
        if include_baselines:
            for baseline in run_gate(
                art.cube, art.index.days, eval_days, art.index.apps, quantiles, art.growth
            ):
                candidates[baseline.name] = baseline.cube

        for name, cube in candidates.items():
            predicted = cube.q.astype(float)
            table = coverage_table(observed, predicted, quantiles)
            pit = pit_values(observed, predicted, quantiles)
            uniformity = pit_uniformity(pit)
            score_rows.append(
                {
                    "fold": fold,
                    "origin": train_days[-1],
                    "model": name,
                    **pinball_loss(observed, predicted, quantiles),
                    "interval_score_90": interval_score(observed, predicted, quantiles, 0.9),
                    "coverage_abs_error": float(table["error"].abs().mean()),
                    "pit_ks_stat": uniformity["ks_stat"],
                    "pit_mean": uniformity.get("pit_mean", np.nan),
                    "pit_dispersion": uniformity.get("dispersion", ""),
                }
            )
            table["fold"] = fold
            table["model"] = name
            coverage_rows.append(table)

    result = RollingOriginResult(
        scores=pd.DataFrame(score_rows),
        coverage=pd.concat(coverage_rows, ignore_index=True) if coverage_rows else pd.DataFrame(),
    )
    LOG.info("%s", result.gate_verdict())
    return result


def _origins(
    all_days: Sequence[date],
    n_folds: int,
    step: int,
    horizon: int,
    min_train: int,
) -> list[int]:
    """Origin positions, most recent last, each leaving a full horizon after it."""
    last = len(all_days) - horizon - 1
    positions = []
    for k in range(n_folds):
        pos = last - k * step
        if pos < min_train:
            break
        positions.append(pos)
    return sorted(positions)
