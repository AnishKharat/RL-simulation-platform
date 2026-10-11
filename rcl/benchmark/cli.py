"""``python -m rcl.benchmark``: run a benchmark in-process and export the report.

One command from grid to table: runs every compatible (environment, controller) pair for each
seed, stores everything in a local database, and writes ``report.md``, ``report.html``,
``summary.csv`` and ``plots/*.svg`` (means with 95% intervals across seeds).
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rcl.analytics import Report, SummaryRow, build_report, degradation_curves, summarize
from rcl.experiments import (
    METRIC_INFO,
    BenchmarkSpec,
    ConfigError,
    MlflowTracker,
    nominal_rows,
    plan,
    run_stored_benchmark,
    sweep_axis,
)
from rcl.persistence import Repository
from rcl.registry import Registry, discover, get_registry

log = logging.getLogger(__name__)
_CSV_FIELDS = (
    "env_id controller_id metric_name unit mean std ci_low ci_high ci_method n_valid n_total "
    "n_censored low_n"
).split()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m rcl.benchmark",
        description="Benchmark controllers across seeds and write a comparison report.",
    )
    p.add_argument("--spec", type=Path, help="BenchmarkSpec JSON file (overrides the grid options)")
    p.add_argument("--name", default="benchmark")
    p.add_argument("--envs", nargs="+", help="environment ids (default: all registered)")
    p.add_argument("--controllers", nargs="+", help="controller ids (default: all registered)")
    p.add_argument("--seeds", type=int, default=5, help="number of seeds: 0..N-1 (default 5)")
    p.add_argument("--episodes", type=int, default=5, help="evaluation episodes per run")
    p.add_argument(
        "--timesteps", type=int, default=20_000, help="training budget for RL controllers"
    )
    p.add_argument("--out", type=Path, default=Path("benchmark-out"), help="output directory")
    p.add_argument("--db", help="database URL (default: sqlite file inside --out)")
    p.add_argument("--mlflow-uri", help="log training curves/checkpoints to this MLflow store")
    return p


def spec_from_args(args: argparse.Namespace, registry: Registry) -> BenchmarkSpec:
    """The grid from ``--spec`` or from the options; ids default to everything registered."""
    if args.spec is not None:
        return BenchmarkSpec.model_validate_json(args.spec.read_text(encoding="utf-8"))
    envs = args.envs or [e.id for e in registry.entries("environment")]
    ctls = args.controllers or [c.id for c in registry.entries("controller")]
    trains = any(
        c.metadata.requires_training for c in registry.entries("controller") if c.id in ctls
    )
    return BenchmarkSpec.model_validate(
        {
            "name": args.name,
            "environments": [{"id": i} for i in envs],
            "controllers": [{"id": i} for i in ctls],
            "seeds": list(range(args.seeds)),
            "episodes": args.episodes,
            "training": {"total_timesteps": args.timesteps} if trains else None,
        }
    )


def _write_outputs(out: Path, rows: list[SummaryRow], report: Report) -> None:
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(report.markdown, encoding="utf-8")
    (out / "report.html").write_text(report.html, encoding="utf-8")
    for name, svg in report.plots.items():
        (out / "plots" / name).write_text(svg, encoding="utf-8")
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(_CSV_FIELDS)
        w.writerows([[getattr(r, k) for k in _CSV_FIELDS] for r in rows])


def _write_robustness(out: Path, rows: list[dict[str, Any]], axis: tuple[str, str, float]) -> None:
    """``robustness.csv``: one line per (env, controller, metric, level) of the sweep."""
    fields = "env_id controller_id metric_name level mean ci_low ci_high n_valid relative".split()
    metrics = sorted({r["metric_name"] for r in rows})
    with (out / "robustness.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow([*fields, "breakdown"])
        for m in metrics:
            for c in degradation_curves(rows, *axis, m):
                for p in c.points:
                    w.writerow(
                        [c.env_id, c.controller_id, m, p.level, p.summary.mean, p.summary.ci_low]
                        + [p.summary.ci_high, p.summary.n_valid, p.relative]
                        + [c.breakdown if m == "success_rate" else ""]  # fraction metrics only
                    )


def main(argv: Sequence[str] | None = None) -> int:
    """Returns 0 on success, 1 if any run failed (the report is still written), 2 for a bad
    specification (nothing is run)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    registry = get_registry()
    discover()
    try:
        spec = spec_from_args(args, registry)
        _, skipped = plan(spec, registry)
    except (ConfigError, ValueError) as exc:
        errors = exc.errors if isinstance(exc, ConfigError) else [{"loc": [], "msg": str(exc)}]
        for e in errors:
            sys.stderr.write(f"error: {'.'.join(str(p) for p in e['loc'])}: {e['msg']}\n")
        return 2
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    repo = Repository(args.db or f"sqlite:///{(out / 'rcl.db').as_posix()}", out / "artifacts")
    tracker = MlflowTracker(args.mlflow_uri) if args.mlflow_uri else None
    suite_id, bench_id = repo.create_suite(spec.name, spec.model_dump(mode="json"))
    run_stored_benchmark(repo, registry, bench_id, suite_id, spec, tracker)
    all_rows = repo.suite_metric_rows(suite_id)
    summary = summarize(nominal_rows(all_rows, spec, registry))
    report = build_report(
        spec.name, spec.model_dump(mode="json"), summary, METRIC_INFO, [vars(k) for k in skipped]
    )
    _write_outputs(out, summary, report)
    if (axis := sweep_axis(spec, registry)) is not None:
        _write_robustness(out, all_rows, axis)
    sys.stdout.write(report.markdown)
    bench = repo.get_benchmark_run(bench_id) or {}
    log.info("report written to %s", out.resolve())
    if bench.get("status") != "done":
        sys.stderr.write(f"warning: benchmark {bench.get('status')}: {bench.get('error')}\n")
        return 1
    return 0
