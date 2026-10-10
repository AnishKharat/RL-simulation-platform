import math

import pytest

from rcl.core.config import ExperimentConfig
from rcl.experiments import JobManager, derive_seeds
from rcl.persistence import Repository
from rcl.registry import discover, get_registry


def cfg(env="cartpole", **extra) -> ExperimentConfig:
    d = {
        "name": "t",
        "seed": 2,
        "environment": {"id": env},
        "controller": {"id": "pid"},
        "evaluation": {"episodes": 2},
        **extra,
    }
    return ExperimentConfig.model_validate(d)


@pytest.fixture
def setup(tmp_path):
    discover()
    repo = Repository(f"sqlite:///{tmp_path / 'db.sqlite'}", tmp_path / "artifacts")
    jobs = JobManager(repo, get_registry(), max_workers=1)
    yield repo, jobs
    jobs.shutdown()


def test_successful_job_persists_results(setup):
    repo, jobs = setup
    _, run = repo.create_experiment(cfg(), 2, derive_seeds(2))
    jobs.submit(run, cfg(), 2).result(timeout=60)
    r = repo.get_run(run)
    assert (r["status"], r["progress"], r["stage"]) == ("done", 1.0, None)
    assert r["versions"]["python"] and r["task_spec"]["horizon_steps"] == 500
    res = repo.get_results(run)
    assert [e["index"] for e in res["episodes"]] == [0, 1]
    names = {(m["metric_name"], m["episode_index"]) for m in res["metrics"]}
    assert {("return", None), ("angle_error", 1), ("max_abs_angle_error", None)} <= names
    traj = repo.load_trajectory(run, 0)
    assert traj["states"].shape[1] == 4 and len(traj["time"]) == len(traj["states"])
    assert not any(math.isnan(m["value"]) for m in res["metrics"] if m["value"] is not None)


def test_failing_job_is_recorded_not_raised(setup):
    repo, jobs = setup
    bad = cfg(env="does_not_exist")
    _, run = repo.create_experiment(bad, 2, derive_seeds(2))
    jobs.submit(run, bad, 2).result(timeout=60)
    r = repo.get_run(run)
    assert r["status"] == "failed" and r["finished_at"]
    assert r["error"]["type"] == "ConfigError" and "does_not_exist" in r["error"]["message"]
