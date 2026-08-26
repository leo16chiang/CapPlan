"""End-to-end pipeline stages, wired to the registry.

Each function is one CLI verb, takes config, and writes a versioned run
directory with a manifest. They chain by run id rather than by passing objects
around, so any stage can be re-run against an earlier stage's output without
re-running everything before it -- which matters when Stage 1 takes a minute and
the simulation takes two.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from capplan import paths
from capplan.config import Config
from capplan.data.calendar import grid_from_config
from capplan.data.event_labels import anomaly_report
from capplan.data.ingest import bootstrap_synthetic_lake, read_intervals, read_table
from capplan.data.mips_normalisation import NormalisationTable, audit_summary
from capplan.diagnostics import run_all
from capplan.diagnostics.runner import verdict as diagnostics_verdict
from capplan.logging_utils import get_logger
from capplan.model.calibrate import apply_conformal, calibrate_stage1
from capplan.model.features import anomaly_mask
from capplan.model.forecast import ForecastCube
from capplan.model.train import fit_stage1, fitted_quantiles, forecast_horizon
from capplan.registry import Registry, Run
from capplan.serve.forecast_store import publish
from capplan.serve.pack_gen import generate_pack
from capplan.sim.residuals import compute_residuals, dependence_report
from capplan.sim.simulate import build_sampler, simulate

LOG = get_logger(__name__)


def _registry(cfg: Config) -> Registry:
    return Registry(cfg.get("registry.root", "artefacts"))


# --------------------------------------------------------------------------


def load_sources_config(path: str | Path = "config/sources.yaml") -> dict:
    """Query templates and column maps. Never credentials."""
    import yaml

    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"{resolved} not found. It holds the SQL that pulls your SMF tables; "
            "copy the template from the repository and edit the table names."
        )
    return yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}


def build_source(cfg: Config, name: str, sources_cfg: dict, seed: int = 7):
    """Construct a source by name from config/sources.yaml."""
    from capplan.data.sources import get_source

    section = dict(sources_cfg.get(name) or {})
    if name == "synthetic":
        section.setdefault("grid", grid_from_config(cfg))
        section.setdefault("n_apps", int(cfg.get("scope.top_n_apps", 35)))
        section.setdefault("seed", seed)
    return get_source(name, **section)


def stage_probe(
    cfg: Config,
    source_name: str,
    start: date,
    end: date,
    sources_path: str | Path = "config/sources.yaml",
) -> Run:
    """Sample the source and check it against what CapPlan assumes.

    Run this before a full extract, and again after every edit to the query
    templates. It reads a few hundred rows, not the archive.
    """
    sources_cfg = load_sources_config(sources_path)
    source = build_source(cfg, source_name, sources_cfg)
    run = _registry(cfg).new_run("probe", config=cfg, tags=[source_name])

    report = source.probe(start, end)
    findings = _probe_findings(report, cfg)
    report["findings"] = findings
    run.write_json("probe.json", report)
    run.record("findings", findings)
    run.finalise()

    LOG.info("probe of %s over %s..%s", source_name, start, end)
    for line in findings:
        LOG.info("  %s", line)
    return run


def _probe_findings(report: dict, cfg: Config) -> list[str]:
    """Turn a raw profile into the sentences a reviewer needs to read.

    Deliberately opinionated. A profile nobody interprets is a profile nobody
    acts on, and every check below corresponds to a way the pipeline goes
    quietly wrong rather than loudly wrong.
    """
    out: list[str] = []
    expected_minutes = int(cfg.get("calendar.interval_minutes"))

    intervals = report.get("intervals") or {}
    if not intervals.get("rows_sampled"):
        out.append(
            "FAIL intervals: the query returned nothing. Nothing else can run."
        )
    else:
        observed = intervals.get("interval_minutes_mode")
        if observed is None:
            out.append("WARN intervals: could not infer the interval length from the sample.")
        elif abs(observed - expected_minutes) > 1e-6:
            out.append(
                f"FAIL intervals: SMF intervals look like {observed:g} minutes, but "
                f"calendar.interval_minutes is {expected_minutes}. Fix the config before "
                "anything else -- the prime-time grid, the row count that justifies a "
                "neural Stage 1, and every downstream index depend on it."
            )
        else:
            per_day = intervals.get("intervals_per_day_implied")
            out.append(
                f"OK intervals: {observed:g}-minute intervals, matching config"
                + (f" ({per_day} per prime-time day)" if per_day else "")
            )
        distinct = intervals.get("distinct_app_id", 0)
        target = int(cfg.get("scope.top_n_apps", 35))
        if distinct <= 1:
            out.append(
                f"FAIL intervals: only {distinct} distinct app_id in the sample. The "
                "service-class/report-class to application mapping is probably not "
                "joining -- check SMF.APP_MAPPING."
            )
        elif distinct < target:
            out.append(
                f"WARN intervals: {distinct} distinct app_id in the sample against a "
                f"top_n_apps of {target}. Fine if the sample window is short; a problem "
                "if it is not."
            )
        else:
            out.append(f"OK intervals: {distinct} distinct app_id in the sample")
        if not any(k.startswith(("msu_", "mips_")) for k in intervals):
            out.append("FAIL intervals: neither msu nor mips came back.")
        for column in ("msu", "mips"):
            if intervals.get(f"{column}_nulls"):
                out.append(
                    f"WARN intervals: {intervals[f'{column}_nulls']} null {column} values "
                    "in the sample. A missing interval is not a zero -- decide which "
                    "this is before ingesting."
                )

    totals = report.get("lpar_totals") or {}
    if not totals.get("rows_sampled"):
        out.append(
            "FAIL lpar_totals: nothing came back. Without realised SMF 70-1 LPAR peaks "
            "there is no simulation backtest, and without that there is no evidence the "
            "coincidence model is right. This is the most important dependency in the "
            "project -- resolve it now, not in week six."
        )
    else:
        out.append(f"OK lpar_totals: {totals['rows_sampled']} rows sampled")

    events = report.get("events") or {}
    if not events.get("rows_sampled"):
        out.append(
            "WARN events: no DR/IST/GCC SDF windows. They will only be caught by the "
            "robust-z spike detector, which flags candidates for a human rather than "
            "labelling them. Workable, worse."
        )
    else:
        out.append(
            f"OK events: {events['rows_sampled']} windows, types "
            f"{events.get('sample_event_type', [])}"
        )

    submissions = report.get("submissions") or {}
    if not submissions.get("rows_sampled"):
        out.append(
            "WARN submissions: none found. No per-app bias score, and no benchmark to "
            "beat -- the model would have nothing to be better than."
        )
    else:
        out.append(f"OK submissions: {submissions['rows_sampled']} rows sampled")
    return out


def stage_ingest(
    cfg: Config,
    source_name: str = "synthetic",
    start: date | None = None,
    end: date | None = None,
    seed: int = 7,
    sources_path: str | Path = "config/sources.yaml",
) -> Run:
    """Extract from a source, scope it, and land it in the lake."""
    from capplan.data.ingest import scope_intervals, write_lake
    from capplan.data.sources.sql import extract_to_frames

    grid = grid_from_config(cfg)
    run = _registry(cfg).new_run("ingest", config=cfg, tags=[source_name])

    if source_name == "existing":
        intervals = read_intervals()
        run.record_inputs({"source": "existing lake", "rows": len(intervals)})
        run.record_metrics({"rows": len(intervals), "apps": intervals["app_id"].nunique()})
        run.finalise()
        return run

    if start is None or end is None:
        raise ValueError(
            "ingest needs --from and --to. An unbounded extract against a warehouse "
            "is how you find out what your DBA's alerting threshold is."
        )

    sources_cfg = load_sources_config(sources_path) if source_name != "synthetic" else {}

    # Refuse a configuration that would add the same CPU twice. This has to
    # happen before extraction, not after: every downstream stage would be
    # arithmetically correct on top of a double-counted base, so nothing later
    # would look wrong.
    roles = (sources_cfg.get(source_name) or {}).get("table_roles")
    if roles:
        from capplan.data.tables import validate_roles

        for warning in validate_roles(roles):
            LOG.warning("table roles: %s", warning)
            run.manifest.setdefault("record", {}).setdefault("role_warnings", []).append(warning)

    source = build_source(cfg, source_name, sources_cfg, seed=seed)
    frames, extract = extract_to_frames(
        source, ("intervals", "lpar_totals", "events", "submissions"), start, end
    )
    run.record_inputs({"source": source_name, "from": str(start), "to": str(end)})
    run.write_json("extract_report.json", extract.to_dict())

    if "intervals" not in frames:
        run.finalise(status="failed", note="source returned no interval rows")
        raise RuntimeError(
            f"source {source_name!r} returned no interval rows for {start}..{end}. "
            "Run `capplan probe` to see what the queries actually return."
        )

    # Label anomalies before scoping, so an event window that covers a
    # non-prime interval still marks the prime-time part of the same day.
    intervals = frames["intervals"]
    if "events" in frames:
        from capplan.data.event_labels import label_from_windows

        intervals = label_from_windows(intervals, frames["events"])

    scoped, report = scope_intervals(
        intervals, grid, cfg, NormalisationTable.from_config(cfg)
    )
    from capplan.data.schema import conform

    written = write_lake(
        {
            name: conform(frame, name, grid)
            for name, frame in (
                ("intervals", scoped),
                ("events", frames.get("events")),
                ("submissions", frames.get("submissions")),
                ("lpar_totals", frames.get("lpar_totals")),
            )
            if frame is not None
        }
    )
    run.record_metrics({**report.to_dict(), **extract.to_dict()["rows_by_table"]})
    run.write_json("ingest_report.json", report.to_dict())
    run.record("written", {k: str(v) for k, v in written.items()})
    run.finalise()
    return run


def stage_diagnostics(cfg: Config) -> Run:
    """The two pure-SQL checks. Run these before deciding to model anything."""
    run = _registry(cfg).new_run("diagnostics", config=cfg)
    result = run_all(exclude_anomalies=True)
    for path in result.write(run.dir / "frames"):
        run.add_artefact(path, role=path.stem)
    run.record_metrics(result.headline)
    call = diagnostics_verdict(result.headline)
    run.record("verdict", call)
    run.write_json("verdict.json", {"verdict": call, "headline": result.headline})
    run.finalise()
    LOG.info("%s", call)
    return run


def stage_coincidence(cfg: Config, sample_days: int | None = None) -> Run:
    """Estimate the coincidence factor from sub-daily data and store it.

    Needed only when the forecast will run at daily grain. At sub-daily grain
    the coincidence is in the data and the simulation measures it directly.
    """
    from capplan.sim.coincidence_factor import estimate_from_intervals, save

    grid = grid_from_config(cfg)
    intervals = read_intervals()
    run = _registry(cfg).new_run("coincidence", config=cfg)

    if sample_days:
        keep = sorted(intervals["business_date"].unique())[-sample_days:]
        intervals = intervals[intervals["business_date"].isin(set(keep))]

    try:
        lpar_totals = read_table("lpar_totals")
    except FileNotFoundError:
        lpar_totals = None
        LOG.warning(
            "no lpar_totals: the factor's numerator will be the reconstructed sum of "
            "scoped applications rather than the realised LPAR peak, so work not "
            "attributed to any scoped application is invisible to it."
        )

    factor = estimate_from_intervals(intervals, grid, lpar_totals=lpar_totals)
    adequacy = factor.sample_adequacy()

    save(factor, run.path("coincidence_factor.npz"))
    run.add_artefact(run.dir / "coincidence_factor.npz", role="coincidence_factor")
    run.record_metrics({**factor.summary(), **{k: v for k, v in adequacy.items() if k != "verdict"}})
    run.record("verdict", adequacy["verdict"])
    run.write_json("coincidence.json", {**factor.summary(), **adequacy})
    run.finalise()
    LOG.info("%s", adequacy["verdict"])
    return run


def load_coincidence_factor(cfg: Config, run_id: str | None = None):
    """Latest stored factor, or None if none has been estimated."""
    from capplan.sim.coincidence_factor import load as load_factor

    try:
        source = _registry(cfg).resolve("coincidence", run_id)
    except FileNotFoundError:
        return None
    path = source.dir / "coincidence_factor.npz"
    return load_factor(path) if path.exists() else None


def stage_train(cfg: Config, calibrate: bool = True) -> Run:
    """Fit Stage 1 and, by default, learn the conformal adjustment."""
    grid = grid_from_config(cfg)
    intervals = read_intervals()
    run = _registry(cfg).new_run("train", config=cfg)

    art = fit_stage1(intervals, grid, cfg)
    run.record_metrics(art.metrics)
    run.record(
        "growth_annual_pct",
        {
            app: round(100 * (float(np.exp(g)) - 1), 2)
            for app, g in zip(art.index.apps, art.growth)
        },
    )
    np.savez_compressed(
        run.path("stage1.npz"),
        scales=art.scales,
        growth=art.growth,
        feed_correction=art.feed_correction,
        apps=np.array(art.index.apps, dtype=object),
        days=np.array([d.isoformat() for d in art.index.days], dtype=object),
        coef=getattr(art.model, "coef_", np.zeros(0)),
        mean=getattr(art.model, "mean_", np.zeros(0)),
        std=getattr(art.model, "std_", np.zeros(0)),
        origin=str(art.spec.origin),
        train_years_max=art.spec.train_years_max,
    )
    run.add_artefact(run.dir / "stage1.npz", role="stage1_model")

    if calibrate:
        adjustment, metrics = calibrate_stage1(art, intervals, cfg)
        np.savez_compressed(
            run.path("conformal.npz"),
            per_app=adjustment.per_app,
            pooled=adjustment.pooled,
            quantiles=adjustment.quantiles,
            apps=np.array(adjustment.apps, dtype=object),
            method=adjustment.method,
        )
        run.add_artefact(run.dir / "conformal.npz", role="conformal")
        run.record_metrics({f"calibration.{k}": v for k, v in metrics.items()})

    run.finalise()
    return run


def stage_simulate(
    cfg: Config,
    n_paths: int | None = None,
    reducers: tuple[str, ...] = ("annual_max", "mean_of_monthly_peaks", "p95_of_daily_peaks"),
    apply_calibration: bool = True,
) -> Run:
    """Stages 1 through 3 end to end, producing the fiscal-year distributions."""
    grid = grid_from_config(cfg)
    intervals = read_intervals()
    n_paths = n_paths or int(cfg.get("simulation.n_paths"))
    run = _registry(cfg).new_run("simulate", config=cfg)

    art = fit_stage1(intervals, grid, cfg)
    panel = compute_residuals(
        observed=art.cube,
        predicted_q=fitted_quantiles(art),
        quantiles=art.quantiles,
        days=art.index.days,
        apps=art.index.apps,
        anomaly=anomaly_mask(intervals, art.index),
        scaling=cfg.get("simulation.block_bootstrap.residual_scale", "spread"),
    )
    run.record_metrics(dependence_report(panel))
    growth_by_app = {
        app: float(g) for app, g in zip(art.index.apps, art.growth)
    }
    run.write_json(
        "growth.json",
        {app: round(100 * (float(np.exp(g)) - 1), 3) for app, g in growth_by_app.items()},
        role="growth_annual_pct",
    )
    np.savez_compressed(
        run.path("growth.npz"),
        apps=np.array(art.index.apps, dtype=object),
        growth=art.growth,
    )
    run.add_artefact(run.dir / "growth.npz", role="growth")

    anchor = max(art.index.days)
    future = grid.horizon_business_days(anchor, int(cfg.get("model.horizon_fiscal_years")))
    forecast = forecast_horizon(art, future)

    if apply_calibration:
        adjustment, metrics = calibrate_stage1(art, intervals, cfg)
        forecast = apply_conformal(forecast, adjustment)
        run.record_metrics({f"calibration.{k}": v for k, v in metrics.items()})

    sampler = build_sampler(cfg, panel)
    # A daily-grain forecast has a degenerate interval axis, so "peak of the
    # sum" collapses to "sum of application peaks" unless a factor measured on
    # sub-daily data is applied.
    factor = load_coincidence_factor(cfg) if grid.intervals_per_day == 1 else None
    if grid.intervals_per_day == 1 and factor is None:
        LOG.warning(
            "daily grain with no stored coincidence factor. Run `capplan coincidence` "
            "against sub-daily data first, or read the result as an upper bound."
        )
    result = simulate(
        forecast,
        sampler,
        grid,
        n_paths=n_paths,
        path_chunk=min(int(cfg.get("simulation.path_chunk")), n_paths),
        reducers=reducers,
        seed=int(cfg.get("simulation.seed")),
        dependence=cfg.get("simulation.dependence"),
        r4ha_hours=float(cfg.get("simulation.r4ha_hours", 4.0)),
        coincidence_factor=factor,
    )
    run.record_metrics(result.diagnostics)
    result.save(run.path("simulation.npz"))
    run.add_artefact(run.dir / "simulation.npz", role="simulation")

    tables = pd.concat([result.fy_table(name) for name in reducers], ignore_index=True)
    tables.to_parquet(run.path("fy_summary.parquet"), index=False)
    run.add_artefact(run.dir / "fy_summary.parquet", role="fy_summary")
    run.record("fy_summary", tables.to_dict(orient="records"))

    published = publish(result, run_id=run.run_id)
    run.record("published", {k: str(v) for k, v in published.items()})
    run.finalise()

    LOG.info("\n%s", tables.to_string(index=False))
    return run


def stage_backtest(cfg: Config, n_paths: int = 1000, reducer: str = "annual_max") -> Run:
    """Backtest the simulation against realised LPAR peaks."""
    from capplan.eval.sim_backtest import backtest_simulation

    grid = grid_from_config(cfg)
    intervals = read_intervals()
    lpar_totals = read_table("lpar_totals")
    run = _registry(cfg).new_run("backtest", config=cfg)

    result = backtest_simulation(
        intervals, lpar_totals, grid, cfg, n_paths=n_paths, reducer=reducer
    )
    run.record_metrics(result.summary())
    call = result.verdict()
    run.record("verdict", call)
    if result.folds:
        result.frame().to_parquet(run.path("folds.parquet"), index=False)
        run.add_artefact(run.dir / "folds.parquet", role="folds")
    run.write_json("verdict.json", {"verdict": call, "summary": result.summary()})
    run.finalise()
    LOG.info("\n%s", call)
    return run


def stage_evaluate(cfg: Config, horizon_days: int = 60, n_folds: int | None = None) -> Run:
    """Rolling-origin scoring of Stage 1 against the baselines."""
    from capplan.eval.rolling_origin import rolling_origin

    grid = grid_from_config(cfg)
    intervals = read_intervals()
    run = _registry(cfg).new_run("evaluate", config=cfg)

    result = rolling_origin(intervals, grid, cfg, n_folds=n_folds, horizon_days=horizon_days)
    if not result.scores.empty:
        result.scores.to_parquet(run.path("scores.parquet"), index=False)
        run.add_artefact(run.dir / "scores.parquet", role="scores")
        result.coverage.to_parquet(run.path("coverage.parquet"), index=False)
        run.add_artefact(run.dir / "coverage.parquet", role="coverage")
        board = result.leaderboard()
        run.record("leaderboard", board.to_dict(orient="records"))
        LOG.info("\n%s", board.to_string(index=False))
    call = result.gate_verdict()
    run.record("gate_verdict", call)
    run.finalise()
    LOG.info("%s", call)
    return run


def stage_pack(cfg: Config, simulate_run_id: str | None = None) -> Run:
    """Build the custodian interview pack from a simulation run."""
    from capplan.sim.simulate import SimulationResult

    registry = _registry(cfg)
    source = registry.resolve("simulate", simulate_run_id)
    run = registry.new_run("pack", config=cfg)

    result = _load_simulation(source.dir / "simulation.npz", grid=grid_from_config(cfg))
    intervals = read_intervals()
    diagnostics = run_all(exclude_anomalies=True)

    # Growth comes from the simulate run that produced these numbers, so the
    # rate quoted to a custodian is the one actually used. Falling back to the
    # latest train run would risk quoting a rate from a different fit.
    growth = _load_growth(source.dir / "growth.npz")
    if not growth:
        try:
            train = registry.resolve("train")
            growth = _load_growth(train.dir / "stage1.npz")
            LOG.info("using growth rates from train run %s", train.run_id)
        except FileNotFoundError:
            LOG.warning(
                "no growth rates found; the pack will say so rather than omit the "
                "section silently"
            )

    backtest_verdict = ""
    try:
        backtest_verdict = registry.resolve("backtest").manifest.get("record", {}).get(
            "verdict", ""
        )
    except FileNotFoundError:
        LOG.info("no backtest run found; the pack will say so rather than imply a pass")

    pack_dir = Path(cfg.get("serve.pack_dir", "artefacts/packs")) / run.run_id
    written = generate_pack(
        result,
        pack_dir,
        run_id=run.run_id,
        growth=growth,
        submission_bias=diagnostics.frames.get("submission_bias_summary"),
        peak_interval_spread=diagnostics.frames.get("peak_interval_spread"),
        coincidence_summary=diagnostics.frames.get("coincidence_summary"),
        backtest_verdict=backtest_verdict,
        normalisation_audit=audit_summary(intervals),
        anomaly_report=anomaly_report(intervals),
        interval_minutes=int(cfg.get("calendar.interval_minutes")),
        prime_window=f"{cfg.get('calendar.prime_start')}-{cfg.get('calendar.prime_end')}",
    )
    run.record("pack_dir", str(pack_dir))
    run.record("pages", len(written))
    run.record("source_simulate_run", source.run_id)
    run.finalise()
    LOG.info("pack written to %s (%d pages)", pack_dir, len(written))
    return run


def _load_growth(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=True) as data:
        if "growth" not in data.files or "apps" not in data.files:
            return {}
        return {str(a): float(g) for a, g in zip(data["apps"], data["growth"])}


def _load_simulation(path: Path, grid):
    from capplan.sim.reducers import PathSummary, get_reducer, reduce_by_fiscal_year
    from capplan.sim.simulate import SimulationResult

    def _optional(data, key):
        """A zero-length array is the on-disk sentinel for 'not accumulated'.

        Restoring it as an empty array instead of None would make the R4HA
        reducers fail deep inside a per-path loop rather than at the boundary.
        """
        if key not in data.files:
            return None
        value = data[key]
        return value if value.size else None

    with np.load(path, allow_pickle=True) as data:
        result = SimulationResult(
            daily_peaks=data["daily_peaks"],
            daily_means=data["daily_means"],
            daily_sum_app_peaks=data["daily_sum_app_peaks"],
            app_peaks=data["app_peaks"],
            daily_r4ha=_optional(data, "daily_r4ha"),
            daily_sum_app_r4ha=_optional(data, "daily_sum_app_r4ha"),
            days=[date.fromisoformat(str(d)) for d in data["days"]],
            apps=[str(a) for a in data["apps"]],
            fiscal_years=data["fiscal_years"],
            dependence=str(data["dependence"]),
            n_paths=int(data["daily_peaks"].shape[0]),
        )
        for key in data.files:
            if key.startswith("reducer__"):
                result.reducer_values[key[len("reducer__") :]] = data[key]

    # Reducer-by-fiscal-year is cheap to rebuild and not worth persisting.
    for name in result.reducer_values:
        reducer = get_reducer(name)
        by_fy: dict[int, list[float]] = {}
        for p in range(result.n_paths):
            path_summary = PathSummary(
                daily_peaks=result.daily_peaks[p],
                daily_means=result.daily_means[p],
                days=result.days,
                daily_r4ha=(
                    None if result.daily_r4ha is None else result.daily_r4ha[p]
                ),
                fiscal_years=result.fiscal_years,
            )
            for fy, value in reduce_by_fiscal_year(
                path_summary, reducer, result.fiscal_years
            ).items():
                by_fy.setdefault(fy, []).append(value)
        result.reducer_by_fy[name] = {fy: np.asarray(v) for fy, v in by_fy.items()}
    result.diagnostics = {**result.coincidence(), **result.coincidence_by_target()}
    return result
