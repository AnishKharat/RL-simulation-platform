"""Background execution of runs (ADR 0007): a thread pool that persists status and results."""

from __future__ import annotations

import logging
import math
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from rcl.core.config import ExperimentConfig
from rcl.experiments.benchmark import BenchmarkSpec, plan
from rcl.experiments.manager import RunResult, derive_seeds, execute, resolve_config
from rcl.experiments.tracking import Tracker
from rcl.metrics import MetricValue
from rcl.persistence import Repository
from rcl.registry import Registry

log = logging.getLogger(__name__)
_TRAIN_SHARE = 0.9  # progress fraction reserved for training; evaluation/saving fill the rest


def _metric_rows(m: MetricValue, episode: int | None) -> list[dict[str, Any]]:
    """One row for the primary value and one per companion (data-model.md)."""
    pairs = [(m.name, m.value, m.unit)] + [(n, v, u) for n, (v, u) in m.companions.items()]
    return [
        {
            "metric_name": n,
            "episode_index": episode,
            "value": v if math.isfinite(v) else None,
            "unit": u,
            "status": m.status,
        }
        for n, v, u in pairs
    ]


def to_rows(result: RunResult) -> dict[str, Any]:
    """Flatten a RunResult into the keyword arguments of ``Repository.save_results``."""
    episodes, metrics = [], []
    for i, (t, ms) in enumerate(zip(result.trajectories, result.episode_metrics, strict=True)):
        arrays = {"time": t.time, "states": t.states, "observations": t.observations}
        arrays |= {"actions": t.actions, "rewards": t.rewards}
        if t.commanded_actions is not None:
            arrays["commanded_actions"] = t.commanded_actions
        episodes.append(
            dict(
                index=i,
                seed=t.seed,
                terminated=t.terminated,
                truncated=t.truncated,
                success=t.success,
                n_steps=t.n_steps,
                arrays=arrays,
            )
        )
        metrics += [row for m in ms for row in _metric_rows(m, i)]
    metrics += [row for m in result.metrics for row in _metric_rows(m, None)]
    return {
        "versions": result.record.versions,
        "code_version": result.record.code_version,
        "task_spec": result.task.model_dump(mode="json"),
        "episodes": episodes,
        "metrics": metrics,
        "train_curve": list(result.train.curve) if result.train else [],
    }


def _record_training(
    repo: Repository, tracker: Tracker | None, run_id: str, result: RunResult
) -> None:
    """Link the checkpoint in the DB and, if a tracker is configured, log curve + checkpoint."""
    ckpt = repo.checkpoint_dir(run_id)
    if not ckpt.is_dir() or result.train is None:
        return
    mlflow_id = None
    if tracker is not None:
        rec = result.record
        try:
            mlflow_id = tracker.log_training(
                run_name=f"{rec.config.name}:seed{rec.seed}",
                params={
                    "run_id": run_id,
                    "environment": rec.config.environment.id,
                    "controller": rec.config.controller.id,
                    "config_hash": rec.config_hash,
                    "seed": str(rec.seed),
                    "train_seed": str(rec.derived_seeds["train"]),
                    "code_version": rec.code_version,
                },
                curve=list(result.train.curve),
                checkpoint_dir=ckpt,
            )
        except Exception:
            log.warning("tracking run %s failed", run_id, exc_info=True)
    try:
        repo.link_checkpoint(run_id, ckpt, mlflow_id)
    except Exception:
        log.warning("linking checkpoint of %s failed", run_id, exc_info=True)


def run_and_store(
    repo: Repository,
    registry: Registry,
    run_id: str,
    config: ExperimentConfig,
    seed: int,
    tracker: Tracker | None = None,
) -> bool:
    """Execute one queued run and persist its results; returns False if it failed.

    Never raises: any exception becomes a ``failed`` run (a worker must never die silently).
    """
    stage, trained = "starting", False

    def tick(**fields: Any) -> None:
        try:  # a lost progress update must never fail the run
            repo.update_run(run_id, **fields)
        except Exception:
            log.warning("progress update for %s failed", run_id, exc_info=True)

    def on_stage(name: str) -> None:
        nonlocal stage, trained
        stage, trained = name, trained or name == "training"
        tick(stage=name, progress=_TRAIN_SHARE if name == "evaluating" and trained else 0.0)

    def progress(done: int, total: int) -> None:
        tick(progress=_TRAIN_SHARE * min(done / total, 1.0))

    try:
        repo.update_run(run_id, status="running")
        result = execute(
            config,
            seed=seed,
            registry=registry,
            checkpoint_dir=repo.checkpoint_dir(run_id),
            on_stage=on_stage,
            progress=progress,
        )
        stage = "saving"
        repo.update_run(run_id, stage=stage, progress=0.95)
        repo.save_results(run_id, **to_rows(result))
        if result.train is not None:  # DB is already complete: tracking problems never fail a run
            _record_training(repo, tracker, run_id, result)
        return True
    except Exception as exc:  # job boundary
        log.exception("run %s failed in stage %s", run_id, stage)
        repo.fail_run(run_id, type(exc).__name__, str(exc), stage)
        return False


def run_stored_benchmark(
    repo: Repository,
    registry: Registry,
    bench_id: str,
    suite_id: str,
    spec: BenchmarkSpec,
    tracker: Tracker | None = None,
) -> None:
    """Run a benchmark whose suite row exists: create every experiment/run (queued) up front,
    then execute them sequentially. A failed run does not stop the others; the benchmark run
    ends ``failed`` if any run failed, ``done`` otherwise."""
    try:
        repo.update_benchmark_run(bench_id, status="running")
        pairs, _ = plan(spec, registry)
        queued: list[tuple[str, ExperimentConfig, int]] = []
        for env, ctl in pairs:
            first = spec.experiment(env, ctl, spec.seeds[0])
            resolved, _, _ = resolve_config(first, registry)
            exp_id, run_id = repo.create_experiment(
                resolved, spec.seeds[0], derive_seeds(spec.seeds[0]), suite_id=suite_id
            )
            queued.append((run_id, resolved, spec.seeds[0]))
            for seed in spec.seeds[1:]:
                queued.append((repo.add_run(exp_id, seed, derive_seeds(seed)), resolved, seed))
        failed = 0
        for done, (run_id, config, seed) in enumerate(queued, start=1):
            failed += not run_and_store(repo, registry, run_id, config, seed, tracker)
            repo.update_benchmark_run(bench_id, progress=done / len(queued))
        if failed:
            error = {"type": "RunsFailed", "message": f"{failed} of {len(queued)} runs failed"}
            repo.update_benchmark_run(bench_id, status="failed", error=error)
        else:
            repo.update_benchmark_run(bench_id, status="done", progress=1.0)
    except Exception as exc:  # job boundary
        log.exception("benchmark %s failed", bench_id)
        error = {"type": type(exc).__name__, "message": str(exc)}
        repo.update_benchmark_run(bench_id, status="failed", error=error)


class JobManager:
    """Runs experiments and benchmarks on a bounded thread pool; failures are recorded."""

    def __init__(
        self,
        repo: Repository,
        registry: Registry,
        max_workers: int = 2,
        tracker: Tracker | None = None,
    ) -> None:
        self._repo, self._registry, self._tracker = repo, registry, tracker
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rcl-job")

    def submit(self, run_id: str, config: ExperimentConfig, seed: int) -> Future[bool]:
        return self._pool.submit(
            run_and_store, self._repo, self._registry, run_id, config, seed, self._tracker
        )

    def submit_benchmark(self, bench_id: str, suite_id: str, spec: BenchmarkSpec) -> Future[None]:
        return self._pool.submit(
            run_stored_benchmark,
            self._repo,
            self._registry,
            bench_id,
            suite_id,
            spec,
            self._tracker,
        )

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
