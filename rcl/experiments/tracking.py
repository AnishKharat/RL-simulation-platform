"""Optional MLflow sink for TRAINING artifacts and curves (ADR 0008, data-model.md).

Roles: the SQL DB stays the source of truth for the catalogue, results and comparisons and
answers every query without MLflow. MLflow only gets what trackers are good at: the learning
curve and the checkpoint files of runs that trained. Evaluation metrics are NOT duplicated.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol


class Tracker(Protocol):
    def log_training(
        self,
        *,
        run_name: str,
        params: dict[str, str],
        curve: list[tuple[int, float]],
        checkpoint_dir: Path | None,
    ) -> str:
        """Record one trained run; returns the tracker's run id."""


class MlflowTracker:
    """Logs to an MLflow tracking server / store (``tracking_uri``) through a client object,
    so no global MLflow state is touched (safe next to the job thread pool).

    ``mlflow`` is imported lazily: the platform runs without it installed.
    """

    def __init__(self, tracking_uri: str, experiment: str = "rcl") -> None:
        self._uri, self._experiment = tracking_uri, experiment

    def log_training(
        self,
        *,
        run_name: str,
        params: dict[str, str],
        curve: list[tuple[int, float]],
        checkpoint_dir: Path | None,
    ) -> str:
        from mlflow import MlflowClient
        from mlflow.entities import Metric

        client = MlflowClient(tracking_uri=self._uri)
        exp = client.get_experiment_by_name(self._experiment)
        try:
            exp_id = exp.experiment_id if exp else client.create_experiment(self._experiment)
        except Exception:  # two jobs created it at once: the other one won, look it up again
            found = client.get_experiment_by_name(self._experiment)
            if found is None:
                raise
            exp_id = found.experiment_id
        run = client.create_run(exp_id, run_name=run_name)
        run_id: str = run.info.run_id
        status = "FINISHED"
        try:
            for key, text in params.items():
                client.log_param(run_id, key, text)
            now = int(time.time() * 1000)
            points = [Metric("train/mean_episode_return", v, now, step) for step, v in curve]
            for i in range(0, len(points), 1000):  # MLflow accepts at most 1000 metrics per batch
                client.log_batch(run_id, metrics=points[i : i + 1000])
            if checkpoint_dir is not None and checkpoint_dir.is_dir():
                client.log_artifacts(run_id, str(checkpoint_dir), "checkpoint")
        except Exception:
            status = "FAILED"
            raise
        finally:
            client.set_terminated(run_id, status=status)
        return run_id
