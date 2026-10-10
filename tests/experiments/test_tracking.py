from pathlib import Path

import pytest

from rcl.core.config import ExperimentConfig
from rcl.experiments import JobManager, MlflowTracker, derive_seeds
from rcl.persistence import Repository
from rcl.registry import discover, get_registry

PPO = {"n_steps": 32, "batch_size": 32, "n_epochs": 1}


def config(ctl: str, **extra) -> ExperimentConfig:
    d = {
        "name": "t",
        "seed": 3,
        "environment": {"id": "cartpole"},
        "controller": {"id": ctl, "params": PPO if ctl == "ppo" else {}},
        "evaluation": {"episodes": 1},
        **extra,
    }
    return ExperimentConfig.model_validate(d)


class FakeTracker:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[dict] = []
        self.fail = fail

    def log_training(self, **kw) -> str:
        self.calls.append(kw)
        if self.fail:
            raise ConnectionError("tracking server down")
        return "mlflow-run-1"


def run_one(tmp_path: Path, ctl: str, tracker, **extra):
    discover()
    repo = Repository(f"sqlite:///{tmp_path / 'db.sqlite'}", tmp_path / "artifacts")
    cfg = config(ctl, **extra)
    _, run_id = repo.create_experiment(cfg, 3, derive_seeds(3))
    jobs = JobManager(repo, get_registry(), max_workers=1, tracker=tracker)
    try:
        jobs.submit(run_id, cfg, 3).result(timeout=300)
    finally:
        jobs.shutdown()
    return repo, run_id


def test_trained_run_logs_curve_and_links_checkpoint(tmp_path):
    tracker = FakeTracker()
    repo, run_id = run_one(tmp_path, "ppo", tracker, training={"total_timesteps": 64})
    (call,) = tracker.calls
    assert call["params"]["controller"] == "ppo" and call["params"]["seed"] == "3"
    assert call["checkpoint_dir"] == repo.checkpoint_dir(run_id)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in call["curve"])
    (ckpt,) = [a for a in repo.get_artifacts(run_id) if a["kind"] == "checkpoint"]
    assert (
        ckpt["mlflow_run_id"] == "mlflow-run-1"
        and ckpt["size_bytes"] > 0
        and len(ckpt["sha256"]) == 64
    )
    assert repo.get_run(run_id)["status"] == "done"


def test_tracker_failure_never_fails_the_run_and_db_keeps_the_pointer(tmp_path):
    repo, run_id = run_one(
        tmp_path, "ppo", FakeTracker(fail=True), training={"total_timesteps": 64}
    )
    assert repo.get_run(run_id)["status"] == "done"
    (ckpt,) = [a for a in repo.get_artifacts(run_id) if a["kind"] == "checkpoint"]
    assert ckpt["mlflow_run_id"] is None


def test_classical_runs_are_not_tracked(tmp_path):
    tracker = FakeTracker()
    repo, run_id = run_one(tmp_path, "pid", tracker)
    assert tracker.calls == [] and repo.get_run(run_id)["status"] == "done"
    assert not [a for a in repo.get_artifacts(run_id) if a["kind"] == "checkpoint"]


def test_runs_without_a_tracker_still_link_the_checkpoint(tmp_path):
    repo, run_id = run_one(tmp_path, "ppo", None, training={"total_timesteps": 64})
    (ckpt,) = [a for a in repo.get_artifacts(run_id) if a["kind"] == "checkpoint"]
    assert ckpt["mlflow_run_id"] is None


def test_mlflow_tracker_logs_params_curve_and_checkpoint_files(tmp_path):
    pytest.importorskip("mlflow")
    from mlflow import MlflowClient

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.zip").write_bytes(b"weights")
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_id = MlflowTracker(uri, experiment="rcl-test").log_training(
        run_name="r", params={"seed": "1"}, curve=[(10, 1.0), (20, 2.5)], checkpoint_dir=ckpt
    )
    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(run_id)
    assert run.data.params == {"seed": "1"} and run.info.status == "FINISHED"
    hist = client.get_metric_history(run_id, "train/mean_episode_return")
    assert [(m.step, m.value) for m in sorted(hist, key=lambda m: m.step)] == [(10, 1.0), (20, 2.5)]
    assert [a.path for a in client.list_artifacts(run_id, "checkpoint")] == ["checkpoint/model.zip"]
