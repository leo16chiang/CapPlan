"""CapPlan command line.

    capplan check          week-1 environment and dependency check
    capplan sources        list configured data sources
    capplan tables         IZPCA source-table catalogue: grain and role
    capplan probe          sample a source and check it against our assumptions
    capplan ingest         extract from a source and land it in the lake
    capplan diagnostics    the two pure-SQL checks -- run these first
    capplan train          fit Stage 1 and calibrate
    capplan simulate       Stages 1-3, producing the fiscal-year distributions
    capplan evaluate       rolling-origin scoring against the baselines
    capplan backtest       backtest the simulation against realised LPAR peaks
    capplan pack           build the custodian interview pack
    capplan reducers       list the available fiscal-year conventions
    capplan runs           list versioned runs

The intended order for a first pass is `check`, `probe`, `ingest`,
`diagnostics`, and then a decision about whether to continue. `probe` is cheap
and catches the mistakes that are expensive later; `diagnostics` can tell you to
stop, and that is a legitimate outcome.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from capplan import __version__
from capplan.config import load_config
from capplan.envfile import describe as describe_dotenv
from capplan.envfile import load as load_dotenv
from capplan.logging_utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="capplan", description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None, help="path to capplan.yaml")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--env-file",
        action="append",
        default=None,
        help="extra dotenv file to load (repeatable); .env and .env.local load by default",
    )
    parser.add_argument("--version", action="version", version=f"capplan {__version__}")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted-path config override, e.g. --set simulation.n_paths=1000",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="environment and dependency check")

    p = sub.add_parser("ingest", help="extract from a source into the parquet lake")
    p.add_argument(
        "--source",
        default="synthetic",
        help="db2 | files | synthetic | existing (default: synthetic)",
    )
    p.add_argument("--synthetic", action="store_true", help="alias for --source synthetic")
    p.add_argument("--from", dest="date_from", default=None, help="YYYY-MM-DD, inclusive")
    p.add_argument("--to", dest="date_to", default=None, help="YYYY-MM-DD, inclusive")
    p.add_argument("--sources-config", default="config/sources.yaml")
    p.add_argument("--seed", type=int, default=7)

    p = sub.add_parser("probe", help="sample a source and report what came back")
    p.add_argument("--source", default="db2")
    p.add_argument("--from", dest="date_from", default=None)
    p.add_argument("--to", dest="date_to", default=None)
    p.add_argument("--sources-config", default="config/sources.yaml")

    sub.add_parser("sources", help="list configured data sources")
    sub.add_parser("tables", help="IZPCA source-table catalogue: grain and role")

    sub.add_parser("diagnostics", help="coincidence factor and submission bias (SQL only)")

    p = sub.add_parser("train", help="fit Stage 1 and calibrate")
    p.add_argument("--no-calibrate", action="store_true")

    p = sub.add_parser("simulate", help="Stages 1-3 end to end")
    p.add_argument("--paths", type=int, default=None)
    p.add_argument(
        "--reducer",
        action="append",
        default=None,
        help="repeatable; defaults to annual_max, mean_of_monthly_peaks, p95_of_daily_peaks",
    )
    p.add_argument("--no-calibrate", action="store_true")

    p = sub.add_parser("evaluate", help="rolling-origin scoring against the baselines")
    p.add_argument("--horizon-days", type=int, default=60)
    p.add_argument("--folds", type=int, default=None)

    p = sub.add_parser("backtest", help="backtest the simulation, not the point forecast")
    p.add_argument("--paths", type=int, default=1000)
    p.add_argument("--reducer", default="annual_max")

    p = sub.add_parser("pack", help="build the custodian interview pack")
    p.add_argument("--run-id", default=None, help="simulate run id (default: latest)")

    sub.add_parser("reducers", help="list fiscal-year reduction conventions")

    p = sub.add_parser("runs", help="list versioned runs")
    p.add_argument("kind", nargs="?", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    # Before anything reads os.environ. Real environment variables still win.
    load_dotenv(tuple(args.env_file) if args.env_file else None or (".env", ".env.local"))

    if args.command == "check":
        return _check()
    if args.command == "reducers":
        from capplan.sim.reducers import describe_reducers

        print(describe_reducers())
        return 0
    if args.command == "sources":
        return _sources()
    if args.command == "tables":
        from capplan.data.tables import describe_catalogue

        print(describe_catalogue())
        return 0

    cfg = load_config(args.config)
    if args.set:
        cfg = cfg.with_overrides(dict(_parse_override(item) for item in args.set))

    from capplan import pipeline

    if args.command == "ingest":
        source = "synthetic" if args.synthetic else args.source
        start, end = _window(args, cfg, source)
        run = pipeline.stage_ingest(
            cfg,
            source_name=source,
            start=start,
            end=end,
            seed=args.seed,
            sources_path=args.sources_config,
        )
    elif args.command == "probe":
        start, end = _window(args, cfg, args.source, default_days=7)
        run = pipeline.stage_probe(
            cfg, args.source, start, end, sources_path=args.sources_config
        )
    elif args.command == "diagnostics":
        run = pipeline.stage_diagnostics(cfg)
    elif args.command == "train":
        run = pipeline.stage_train(cfg, calibrate=not args.no_calibrate)
    elif args.command == "simulate":
        reducers = tuple(args.reducer) if args.reducer else (
            "annual_max",
            "mean_of_monthly_peaks",
            "p95_of_daily_peaks",
        )
        run = pipeline.stage_simulate(
            cfg,
            n_paths=args.paths,
            reducers=reducers,
            apply_calibration=not args.no_calibrate,
        )
    elif args.command == "evaluate":
        run = pipeline.stage_evaluate(cfg, horizon_days=args.horizon_days, n_folds=args.folds)
    elif args.command == "backtest":
        run = pipeline.stage_backtest(cfg, n_paths=args.paths, reducer=args.reducer)
    elif args.command == "pack":
        run = pipeline.stage_pack(cfg, simulate_run_id=args.run_id)
    elif args.command == "runs":
        return _runs(cfg, args.kind)
    else:  # pragma: no cover - argparse enforces this
        raise SystemExit(f"unhandled command {args.command}")

    print(f"\nrun {run.kind}/{run.run_id} -> {run.dir}")
    return 0


def _window(args, cfg, source: str, default_days: int | None = None):
    """Resolve --from / --to into dates.

    A probe defaults to the last `default_days` of history, because the point is
    to look at a little data quickly. An ingest has no default: an unbounded
    extract against a production warehouse should be a deliberate act.
    """
    from datetime import date, timedelta

    start = date.fromisoformat(args.date_from) if args.date_from else None
    end = date.fromisoformat(args.date_to) if args.date_to else None

    if source == "synthetic":
        grid_end = end or date.today()
        return start or (grid_end - timedelta(days=365 * 3)), grid_end
    if default_days is not None:
        end = end or date.today()
        start = start or (end - timedelta(days=default_days))
    if start is None or end is None:
        raise SystemExit(
            f"--from and --to are required for --source {source}. "
            "An unbounded extract against a warehouse is not a default."
        )
    if start > end:
        raise SystemExit(f"--from ({start}) is after --to ({end})")
    return start, end


def _sources() -> int:
    from capplan.data.sources.base import available_sources
    from capplan.data.sources.db2 import ENV_PREFIX, pyodbc_available

    print("sources:")
    for name in available_sources():
        print(f"  {name}")
    ok, message = pyodbc_available()
    print(f"\ndb2 driver: {'ok' if ok else 'unavailable'} -- {message}")
    print(f"\ndotenv: {describe_dotenv()}")
    print("\ndb2 credentials come from the environment or .env, never from config/:")
    for suffix in ("DSN", "HOST", "PORT", "DATABASE", "USER", "PASSWORD", "PASSWORD_CMD", "DRIVER"):
        name = f"{ENV_PREFIX}{suffix}"
        state = "set" if os.environ.get(name) else "--"
        print(f"  [{state:>3}] {name}")
    print("\nqueries live in config/sources.yaml; run `capplan probe` after editing them.")
    return 0


def _parse_override(item: str) -> tuple[str, object]:
    if "=" not in item:
        raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
    key, _, raw = item.partition("=")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key, value


def _runs(cfg, kind: str | None) -> int:
    from capplan.registry import Registry

    registry = Registry(cfg.get("registry.root", "artefacts"))
    kinds = [kind] if kind else [
        "probe", "ingest", "diagnostics", "train", "simulate", "evaluate", "backtest", "pack",
    ]
    for k in kinds:
        ids = registry.list_runs(k)
        if ids:
            print(f"{k}:")
            for run_id in ids:
                print(f"  {run_id}")
    return 0


def _check() -> int:
    """Week-1 environment check. The proxy question, answered before anything else."""
    import importlib.util

    print(f"capplan {__version__}, python {sys.version.split()[0]}\n")
    required = ["numpy", "scipy", "pandas", "duckdb", "pyarrow", "yaml"]
    optional = {
        "pyodbc": "Db2 extraction. Also needs the IBM Db2 ODBC driver on the machine.",
        "torch": "Stage 1 neural backend. LARGE WHEEL -- this is the week-1 proxy question.",
        "neuralforecast": "N-HiTS / TFT. Pulls torch.",
        "statsforecast": "SARIMA baseline.",
        "lightgbm": "Gradient-boosted quantile baseline.",
        "polars": "Optional, if the interval joins get heavy.",
    }
    failures = 0
    print("required:")
    for module in required:
        ok = importlib.util.find_spec(module) is not None
        print(f"  [{'ok' if ok else 'MISSING'}] {module}")
        failures += 0 if ok else 1
    print("\noptional:")
    for module, why in optional.items():
        ok = importlib.util.find_spec(module) is not None
        print(f"  [{'ok' if ok else '--'}] {module:<16} {why}")

    if importlib.util.find_spec("torch") is None:
        print(
            "\ntorch is not installed. Everything runs without it -- Stage 1 falls back\n"
            "to the quantile_ridge backend, which clears the baseline gate on its own.\n"
            "Confirm torch is on the internal mirror before planning around N-HiTS:\n"
            "    pip download torch --no-deps -d /tmp/torchcheck\n"
        )
    if failures:
        print(f"\n{failures} required package(s) missing.")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
