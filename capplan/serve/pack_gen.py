"""Custodian interview pack.

The deliverable is not a number. It is a number that survives an interview with
the person who owns the application, and that interview goes badly in
predictable ways:

  "Where did that come from?"          -> the growth rate, and how it was fitted
  "That's not what I submitted."       -> their submission history and its bias
  "My app peaks at 9am, not 2pm."      -> the observed peak-interval distribution
  "So we need the sum of these?"       -> no, and here is the coincidence factor
  "What about the DR test?"            -> excluded, labelled, and quantified
  "How do I know any of this is right?"-> the simulation backtest

So the pack answers those six questions per app, in that order, before anyone
asks. Markdown, because it renders in the wiki the capacity team already uses
and diffs cleanly between cycles -- last cycle's pack next to this one, with
the changed numbers visible, is worth more than any chart.

Everything in it is traceable to a run id.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from capplan.logging_utils import get_logger
from capplan.sim.reducers import REDUCERS, TARGET_FAMILY
from capplan.sim.simulate import SimulationResult

LOG = get_logger(__name__)


def generate_pack(
    result: SimulationResult,
    out_dir: Path,
    run_id: str,
    growth: Mapping[str, float] | None = None,
    submission_bias: pd.DataFrame | None = None,
    peak_interval_spread: pd.DataFrame | None = None,
    coincidence_summary: pd.DataFrame | None = None,
    backtest_verdict: str = "",
    normalisation_audit: pd.DataFrame | None = None,
    anomaly_report: pd.DataFrame | None = None,
    reducer: str = "annual_max",
    interval_minutes: int = 15,
    prime_window: str = "08:00-17:00",
) -> dict[str, Path]:
    """Write the overview page and one page per application."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    overview = out_dir / "00_overview.md"
    overview.write_text(
        _overview(
            result,
            run_id,
            reducer,
            coincidence_summary,
            backtest_verdict,
            normalisation_audit,
            anomaly_report,
            interval_minutes,
            prime_window,
        ),
        encoding="utf-8",
    )
    written["overview"] = overview

    for app in result.apps:
        page = out_dir / f"app_{app}.md"
        page.write_text(
            _app_page(
                result,
                app,
                run_id,
                reducer,
                growth or {},
                submission_bias,
                peak_interval_spread,
            ),
            encoding="utf-8",
        )
        written[app] = page

    LOG.info("wrote custodian pack: %d pages -> %s", len(written), out_dir)
    return written


def _overview(
    result: SimulationResult,
    run_id: str,
    reducer: str,
    coincidence_summary: pd.DataFrame | None,
    backtest_verdict: str,
    normalisation_audit: pd.DataFrame | None,
    anomaly_report: pd.DataFrame | None,
    interval_minutes: int,
    prime_window: str,
) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Prime-time peak MIPS forecast",
        "",
        f"Run `{run_id}` | generated {stamp} | dependence model: "
        f"`{result.dependence}` | {result.n_paths:,} simulated paths",
        "",
        f"Scope: production LPARs, prime time ({prime_window} on business days, "
        f"{interval_minutes}-minute intervals), {len(result.apps)} applications, "
        f"horizon {result.days[0]} to {result.days[-1]}.",
        "",
        "## Headline",
        "",
        _fy_table(result, reducer),
        "",
        "## Which number you want depends on which decision you are making",
        "",
        "There is no single 'the peak'. These are the same simulation reduced "
        "different ways, and the differences between them are larger than the "
        "modelling uncertainty within any one of them:",
        "",
        _reducer_comparison(result),
        "",
        "The rolling 4-hour average is not a convention chosen here. IBM's "
        "sub-capacity MLC billing takes the highest 4-hour rolling average MSU "
        "each day and charges on the highest across the month, so if the "
        "question is software cost, that is the definition to match rather than "
        "a candidate to weigh. If the question is hardware sizing, the interval "
        "peak is the right target -- the machine has to survive the moment, not "
        "the four hours around it.",
        "",
        "If you need both, ask for both. The simulation is identical; only the "
        "final reduction differs, and running it twice costs nothing.",
        "",
        "## Why the application numbers do not add up to this",
        "",
        "Every application peaks at a different moment. Adding their individual "
        "peaks assumes they all peak simultaneously, which they do not.",
        "",
    ]
    by_target = {
        k: result.diagnostics.get(k)
        for k in ("coincidence_interval_peak", "coincidence_r4ha")
    }
    coincidence = result.diagnostics.get("simulated_coincidence_daily_mean")
    if coincidence:
        overstatement = 100.0 * (1.0 / coincidence - 1.0)
        lines += [
            f"- Simulated daily coincidence factor: **{coincidence:.3f}**",
            f"- Adding up application peaks overstates the LPAR peak by about "
            f"**{overstatement:.0f}%**",
            "",
        ]
        if by_target.get("coincidence_r4ha") and by_target.get("coincidence_interval_peak"):
            lines += [
                f"- For the interval peak the factor is "
                f"**{by_target['coincidence_interval_peak']:.3f}**; for the rolling "
                f"4-hour average it is **{by_target['coincidence_r4ha']:.3f}**.",
                "",
                "  Averaging over four hours smooths out the timing differences that "
                "stop peaks from summing, so the coincidence correction matters less "
                "for the MLC cost figure than for the hardware one. Worth knowing "
                "before deciding how much weight to put on this part of the method.",
                "",
            ]
    if coincidence_summary is not None and not coincidence_summary.empty:
        lines += [
            "Measured on history, per LPAR and fiscal year:",
            "",
            _md_table(
                coincidence_summary[
                    ["lpar", "fiscal_year", "n_days", "coincidence_mean", "coincidence_p05"]
                ].round(3)
            ),
            "",
            "The simulated factor above should sit close to these. If it does not, "
            "the forecast is wrong in the same direction and the rest of this pack "
            "should be treated as provisional.",
            "",
        ]

    lines += ["## Does the simulation actually work", ""]
    lines.append(backtest_verdict or "_No backtest was run for this pack._")
    lines += [
        "",
        "## What is excluded",
        "",
        "- Off-prime hours, non-production LPARs.",
        "- DR, IST and GCC SDF activity. These are labelled and removed as "
        "forecast targets. They still land in prime time and the hardware still "
        "has to survive them -- see the counts below -- but nobody wants a "
        "two-year forecast of a disaster-recovery exercise.",
        "- Cost. This forecast is in MIPS.",
        "",
        "Capture ratio and MIPS-per-MSU are applied exactly as supplied by the "
        "systems programmers. They are not estimated here, and the values used "
        "on every row are recorded below.",
        "",
    ]
    if anomaly_report is not None and not anomaly_report.empty:
        top = anomaly_report.sort_values("intervals", ascending=False).head(15)
        lines += ["### Excluded event activity", "", _md_table(top.round(1)), ""]
    if normalisation_audit is not None and not normalisation_audit.empty:
        lines += [
            "### Normalisation applied",
            "",
            _md_table(normalisation_audit.round(3)),
            "",
        ]

    lines += [
        "## How to read a number in this pack",
        "",
        "The model never forecasts a peak. It forecasts the distribution of each "
        f"application's load in each {interval_minutes}-minute prime-time interval, "
        "samples whole days from that distribution keeping the applications "
        "correlated as they historically are, adds the applications up interval "
        "by interval, and only then takes a maximum. Every figure here is a peak "
        "of a sum.",
        "",
        f"A `p95` column means: in 95% of simulated futures, the {reducer} figure "
        "came in at or below this.",
        "",
    ]
    return "\n".join(lines)


def _app_page(
    result: SimulationResult,
    app: str,
    run_id: str,
    reducer: str,
    growth: Mapping[str, float],
    submission_bias: pd.DataFrame | None,
    peak_interval_spread: pd.DataFrame | None,
) -> str:
    pos = result.apps.index(app)
    peaks = result.app_peaks[:, pos]
    rate = float(growth.get(app, float("nan")))

    lines = [
        f"# {app} -- prime-time peak MIPS",
        "",
        f"Run `{run_id}`. Horizon {result.days[0]} to {result.days[-1]}.",
        "",
        "## Your application's own peak",
        "",
        "| statistic | MIPS |",
        "|---|---|",
        f"| median simulated peak | {np.quantile(peaks, 0.5):,.0f} |",
        f"| 90th percentile | {np.quantile(peaks, 0.9):,.0f} |",
        f"| 95th percentile | {np.quantile(peaks, 0.95):,.0f} |",
        f"| 99th percentile | {np.quantile(peaks, 0.99):,.0f} |",
        "",
        "> These are **marginal** figures: the peak this application reaches "
        "somewhere in the horizon. Do not add them across applications. The "
        "combined figure in the overview is lower, because applications peak at "
        "different times.",
        "",
        "## Where the growth rate came from",
        "",
    ]
    if np.isfinite(rate):
        lines += [
            f"Fitted annual growth: **{100 * (np.exp(rate) - 1):+.1f}%**.",
            "",
            "Estimated by regressing the log of your application's daily median "
            "prime-time MIPS on time, over the full history in scope. Daily "
            "*medians* rather than peaks, so the growth rate describes the body of "
            "your workload and is not dragged around by a handful of unusual "
            "afternoons.",
            "",
            "If this does not match what you expect, that is the most useful thing "
            "you can tell us. It is a single number, it is the main driver of the "
            "two-year figure, and it can be overridden as a stated scenario.",
            "",
        ]
    else:
        lines += ["_No fitted growth rate available for this application._", ""]

    if submission_bias is not None and not submission_bias.empty:
        rows = submission_bias[submission_bias["app_id"] == app]
        if not rows.empty:
            row = rows.iloc[0]
            direction = "above" if row["bias_ratio"] > 1 else "below"
            lines += [
                "## Your previous submissions",
                "",
                f"Across {int(row['n_cycles'])} submission cycles, your submitted "
                f"peak has averaged **{abs(100 * (row['bias_ratio'] - 1)):.0f}% "
                f"{direction}** what was actually measured "
                f"(bias ratio {row['bias_ratio']:.2f}).",
                "",
                f"The sign was consistent in {100 * row['sign_consistency']:.0f}% of "
                "cycles"
                + (
                    ", so this looks like a systematic difference in method rather "
                    "than noise, and it is worth understanding which of us is "
                    "measuring the wrong thing."
                    if row["sign_consistency"] > 0.7
                    else ", so it looks more like noise than a systematic difference."
                ),
                "",
            ]

    if peak_interval_spread is not None and not peak_interval_spread.empty:
        rows = peak_interval_spread[peak_interval_spread["app_id"] == app]
        if not rows.empty:
            top = rows.sort_values("share", ascending=False).head(5)
            lines += [
                "## When your application peaks",
                "",
                "Interval index 0 is the first prime-time interval of the day.",
                "",
                _md_table(
                    top[["interval_idx", "n_days_peaking_here", "share"]].round(3)
                ),
                "",
                "This is the mechanism behind the combined figure being lower than "
                "the sum of the parts. If your application peaked at the same "
                "moment as everything else, it would not be.",
                "",
            ]

    lines += [
        "## What we would like from you",
        "",
        "1. Is the growth rate above plausible for the next two fiscal years?",
        "2. Is anything landing in this window that history cannot know about -- "
        "a migration, a decommission, a new business line, a volume commitment?",
        "3. If your own number differs materially, what is it measuring? Peak "
        "interval, daily average, month-end, something else?",
        "",
        "Answers to 1 and 2 can be applied as a named scenario and re-run in "
        "minutes, with your name recorded against the assumption.",
        "",
    ]
    return "\n".join(lines)


def _fy_table(result: SimulationResult, reducer: str) -> str:
    if reducer not in result.reducer_by_fy:
        return f"_Reducer `{reducer}` was not computed in this run._"
    rows = []
    for fy, values in sorted(result.reducer_by_fy[reducer].items()):
        rows.append(
            {
                "fiscal year": fy,
                "median": f"{np.quantile(values, 0.5):,.0f}",
                "p90": f"{np.quantile(values, 0.9):,.0f}",
                "p95": f"{np.quantile(values, 0.95):,.0f}",
                "p99": f"{np.quantile(values, 0.99):,.0f}",
            }
        )
    header = (
        f"Combined prime-time peak MIPS across all {len(result.apps)} applications, "
        f"reduced as `{reducer}` ({REDUCERS[reducer].description})"
    )
    return header + "\n\n" + _md_table(pd.DataFrame(rows))


def _reducer_comparison(result: SimulationResult) -> str:
    if not result.reducer_by_fy:
        return "_No reducers computed._"
    last_fy = max(next(iter(result.reducer_by_fy.values())).keys())
    labels = {
        "hardware": "hardware sizing",
        "software_cost": "software (MLC) cost",
        "sustained": "trending / chargeback",
        "duration": "duration",
    }
    rows = []
    for name, by_fy in sorted(
        result.reducer_by_fy.items(), key=lambda kv: TARGET_FAMILY.get(kv[0], "")
    ):
        values = by_fy[last_fy]
        rows.append(
            {
                "decides": labels.get(TARGET_FAMILY.get(name, ""), "-"),
                "reducer": name,
                f"FY{last_fy} median": f"{np.quantile(values, 0.5):,.0f}",
                f"FY{last_fy} p95": f"{np.quantile(values, 0.95):,.0f}",
                "means": REDUCERS[name].description.split(".")[0],
            }
        )
    return _md_table(pd.DataFrame(rows))


def _md_table(frame: pd.DataFrame) -> str:
    """Minimal markdown table. No dependency on tabulate."""
    if frame.empty:
        return "_(no rows)_"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(str(c) for c in columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c], str(c)) for c in columns) + " |")
    return "\n".join(lines)


# Columns that are identifiers rather than quantities: no thousands separator,
# because "FY2,027" is not a fiscal year anyone recognises.
_IDENTIFIER_COLUMNS = ("year", "idx", "index", "id")


def _fmt(value, column: str = "") -> str:
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    identifier = any(token in column.lower() for token in _IDENTIFIER_COLUMNS)
    if isinstance(value, (int, np.integer)):
        return str(int(value)) if identifier else f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "-"
        # Counts and indices arrive as float64 out of DuckDB; rendering an
        # interval index as "9.000" makes the table look like it is reporting a
        # precision it does not have.
        if float(value).is_integer():
            return str(int(value)) if identifier else f"{int(value):,}"
        return f"{value:,.3f}" if abs(value) < 1000 else f"{value:,.0f}"
    return str(value)
