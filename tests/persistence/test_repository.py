import numpy as np
import pytest

from rcl.core.config import ExperimentConfig
from rcl.persistence import Repository

CFG = ExperimentConfig.model_validate(
    {
        "name": "t",
        "seed": 1,
        "environment": {"id": "e"},
        "controller": {"id": "c"},
        "evaluation": {"episodes": 1},
    }
)


@pytest.fixture
def repo(tmp_path):
    return Repository(f"sqlite:///{tmp_path / 'db.sqlite'}", tmp_path / "artifacts")


def finished(repo):
    exp, run = repo.create_experiment(CFG, 1, {"train": 2, "eval": 3})
    arrays = {"states": np.arange(6.0).reshape(3, 2), "rewards": np.ones(2)}
    repo.save_results(
        run,
        versions={"python": "3"},
        code_version="abc",
        task_spec={"horizon_steps": 5},
        episodes=[
            dict(
                index=0,
                seed=3,
                terminated=False,
                truncated=True,
                success=False,
                n_steps=2,
                arrays=arrays,
            )
        ],
        metrics=[
            dict(metric_name="return", episode_index=0, value=2.0, unit="reward", status="ok"),
            dict(
                metric_name="return",
                episode_index=None,
                value=None,
                unit="reward",
                status="undefined",
            ),
        ],
        train_curve=[(10, 1.5)],
    )
    return exp, run


def test_create_queued_run_and_experiment_detail(repo):
    exp, run = repo.create_experiment(CFG, 5, {"train": 2, "eval": 3})
    r = repo.get_run(run)
    assert (r["status"], r["progress"], r["seed"], r["derived_seeds"]) == (
        "queued",
        0.0,
        5,
        {"train": 2, "eval": 3},
    )
    d = repo.get_experiment(exp)
    assert d["config"] == CFG.model_dump(mode="json") and d["config_hash"] == CFG.config_hash()
    assert [x["run_id"] for x in d["runs"]] == [run]
    assert repo.get_run("nope") is None and repo.get_experiment("nope") is None


def test_status_lifecycle_and_failure(repo):
    _, run = repo.create_experiment(CFG, 1, {"train": 2, "eval": 3})
    repo.update_run(run, status="running", stage="training", progress=0.4)
    r = repo.get_run(run)
    assert r["status"] == "running" and r["started_at"] and r["finished_at"] is None
    repo.fail_run(run, "ValueError", "boom", "training")
    r = repo.get_run(run)
    assert r["status"] == "failed" and r["finished_at"] and r["error_stage"] == "training"
    assert r["error"] == {"type": "ValueError", "message": "boom", "stage": "training"}
    with pytest.raises(KeyError):
        repo.update_run("nope", status="running")


def test_save_results_roundtrip(repo):
    _, run = finished(repo)
    r = repo.get_run(run)
    assert (r["status"], r["progress"], r["stage"], r["code_version"]) == ("done", 1.0, None, "abc")
    res = repo.get_results(run)
    assert res["episodes"] == [
        dict(index=0, seed=3, terminated=False, truncated=True, success=False, n_steps=2)
    ]
    assert {
        (m["metric_name"], m["episode_index"], m["value"], m["status"]) for m in res["metrics"]
    } == {
        ("return", 0, 2.0, "ok"),
        ("return", None, None, "undefined"),
    }
    assert res["train_curve"] == [[10, 1.5]]
    traj = repo.load_trajectory(run, 0)
    assert np.array_equal(traj["states"], np.arange(6.0).reshape(3, 2))
    assert repo.load_trajectory(run, 7) is None


def test_fail_orphans_only_touches_unfinished(repo):
    _, done = finished(repo)
    _, queued = repo.create_experiment(CFG, 1, {"train": 2, "eval": 3})
    assert repo.fail_orphans() == 1
    assert repo.get_run(done)["status"] == "done"
    r = repo.get_run(queued)
    assert r["status"] == "failed" and r["error"]["type"] == "interrupted"
