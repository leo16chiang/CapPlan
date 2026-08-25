"""Scenarios: the questions asked in the room.

"What if the payments migration lands in Q2?" "What if we assume 10% growth on
the top five instead of what you fitted?" These arrive during the interview,
and being able to answer them in seconds rather than in a follow-up email is
most of what makes the model credible.

Scenarios are applied to the *forecast cube*, not by refitting. That is a
deliberate limit and worth stating: a scenario is a stated override of the
forecast, not a claim that the model would have learned it. Rescaling an app's
forecast by 1.3 is exactly as defensible as the person who asked for the 1.3,
and the pack records who asked.

Three levers, which is all anyone has actually asked for:
  `growth_override`  replace an app's fitted annual growth
  `level_multiplier` scale an app's whole forecast
  `step_change`      a multiplier applied from a stated date onwards, for a
                     migration or a decommission
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Mapping

import numpy as np

from capplan.logging_utils import get_logger
from capplan.model.forecast import ForecastCube

LOG = get_logger(__name__)


@dataclass
class Scenario:
    """A named, attributed set of overrides."""

    name: str
    requested_by: str = ""
    rationale: str = ""
    growth_override: Mapping[str, float] = field(default_factory=dict)
    level_multiplier: Mapping[str, float] = field(default_factory=dict)
    step_change: Mapping[str, tuple[date, float]] = field(default_factory=dict)

    def touched_apps(self) -> set[str]:
        return (
            set(self.growth_override)
            | set(self.level_multiplier)
            | set(self.step_change)
        )

    def describe(self) -> str:
        lines = [f"Scenario: {self.name}"]
        if self.requested_by:
            lines.append(f"  requested by: {self.requested_by}")
        if self.rationale:
            lines.append(f"  rationale: {self.rationale}")
        for app, rate in sorted(self.growth_override.items()):
            lines.append(f"  {app}: annual growth overridden to {100 * rate:+.1f}%")
        for app, mult in sorted(self.level_multiplier.items()):
            lines.append(f"  {app}: level scaled by {mult:.2f}x")
        for app, (when, mult) in sorted(self.step_change.items()):
            lines.append(f"  {app}: {mult:.2f}x step change from {when}")
        if len(lines) == 1:
            lines.append("  (no overrides -- baseline)")
        return "\n".join(lines)


def apply_scenario(
    cube: ForecastCube,
    scenario: Scenario,
    fitted_growth: Mapping[str, float] | None = None,
    anchor: date | None = None,
) -> ForecastCube:
    """Return a new cube with the scenario applied.

    The original is left untouched, so baseline and scenario can be simulated
    from the same fit and compared without a second training run.

    Every override is multiplicative and applied to every quantile alike. That
    means a scenario shifts the *location* of the predictive distribution and
    leaves its relative spread as fitted -- an override is a statement about the
    level, not a claim to know how uncertainty changes with it.
    """
    out = ForecastCube(
        q=cube.q.copy(),
        quantiles=cube.quantiles.copy(),
        apps=list(cube.apps),
        days=list(cube.days),
        n_intervals=cube.n_intervals,
        backend=cube.backend,
        calibrated=cube.calibrated,
    )
    anchor = anchor or cube.days[0]
    unknown = scenario.touched_apps() - set(cube.apps)
    if unknown:
        # Loud, because a typo'd app id silently doing nothing is how a
        # scenario gets presented as applied when it was not.
        raise KeyError(f"scenario {scenario.name!r} names unknown apps: {sorted(unknown)}")

    years_out = np.array([(d - anchor).days / 365.25 for d in cube.days])

    for app, new_rate in scenario.growth_override.items():
        pos = out.app_pos(app)
        old_rate = float((fitted_growth or {}).get(app, 0.0))
        # Re-base rather than compound: divide out what was fitted, multiply in
        # what was asked for.
        factor = np.exp((new_rate - old_rate) * years_out)
        out.q[pos] *= factor[:, None, None].astype(np.float32)
        LOG.info(
            "scenario %s: %s growth %.1f%% -> %.1f%% (%.2fx by horizon end)",
            scenario.name, app, 100 * old_rate, 100 * new_rate, float(factor[-1]),
        )

    for app, multiplier in scenario.level_multiplier.items():
        out.q[out.app_pos(app)] *= np.float32(multiplier)
        LOG.info("scenario %s: %s level x%.2f", scenario.name, app, multiplier)

    for app, (when, multiplier) in scenario.step_change.items():
        pos = out.app_pos(app)
        mask = np.array([d >= when for d in cube.days])
        out.q[pos, mask] *= np.float32(multiplier)
        LOG.info(
            "scenario %s: %s x%.2f from %s (%d of %d days)",
            scenario.name, app, multiplier, when, int(mask.sum()), len(mask),
        )

    return out.enforce_monotone().clip_nonnegative()


def baseline() -> Scenario:
    return Scenario(name="baseline", rationale="fitted model, no overrides")


def compare(baseline_values: np.ndarray, scenario_values: np.ndarray) -> dict[str, float]:
    """Headline delta between two simulated reducer distributions."""
    b50, s50 = float(np.quantile(baseline_values, 0.5)), float(np.quantile(scenario_values, 0.5))
    b95, s95 = float(np.quantile(baseline_values, 0.95)), float(np.quantile(scenario_values, 0.95))
    return {
        "baseline_p50": b50,
        "scenario_p50": s50,
        "delta_p50_mips": s50 - b50,
        "delta_p50_pct": 100.0 * (s50 / b50 - 1.0) if b50 else float("nan"),
        "baseline_p95": b95,
        "scenario_p95": s95,
        "delta_p95_mips": s95 - b95,
        "delta_p95_pct": 100.0 * (s95 / b95 - 1.0) if b95 else float("nan"),
    }
