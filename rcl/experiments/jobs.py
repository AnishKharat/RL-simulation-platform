"""Background execution of runs (ADR 0007): a thread pool that persists status and results."""

from __future__ import annotations

import logging
import math
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from rcl.core.config import ExperimentConfig
from rcl.experiments.manager import RunResult, execute
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


class JobManager:
    """Runs experiments on a bounded thread pool; any exception becomes a ``failed`` run."""

    def __init__(self, repo: Repository, registry: Registry, max_workers: int = 2) -> None:
        self._repo, self._registry = repo, registry
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rcl-job")

    def submit(self, run_id: str, config: ExperimentConfig, seed: int) -> Future[None]:
        return self._pool.submit(self._run, run_id, config, seed)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)

    def _run(self, run_id: str, config: ExperimentConfig, seed: int) -> None:
        repo, stage, trained = self._repo, "starting", False

        def tick(**fields: Any) -> None:
            try:  # a lost progress update must never fail the run
                repo.update_run(run_id, **fields)
            except Exception:
                log.warning("progress update for %s failed", run_id, exc_info=True)

        def on_stage(name: str) -> None:
            nonlocal stage, trained
            stage, trained = name, trained or name == "training"
            tick(
                stage=name,
                progress=_TRAIN_SHARE if name == "evaluating" and trained else 0.0,
            )

        def progress(done: int, total: int) -> None:
            tick(progress=_TRAIN_SHARE * min(done / total, 1.0))

        try:
            repo.update_run(run_id, status="running")
            result = execute(
                config,
                seed=seed,
                registry=self._registry,
                checkpoint_dir=repo.checkpoint_dir(run_id),
                on_stage=on_stage,
                progress=progress,
            )
            stage = "saving"
            repo.update_run(run_id, stage=stage, progress=0.95)
            repo.save_results(run_id, **to_rows(result))
        except Exception as exc:  # job boundary: a worker must never die silently
            log.exception("run %s failed in stage %s", run_id, stage)
            repo.fail_run(run_id, type(exc).__name__, str(exc), stage)
