import numpy as np
import pytest
from sqlalchemy import create_engine, text

from rcl.core.config import ExperimentConfig
from rcl.persistence import Repository


def cfg(env="e", ctl="c") -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "t",
            "seed": 0,
            "environment": {"id": env},
            "controller": {"id": ctl},
            "evaluation": {"episodes": 1},
        }
    )


@pytest.fixture
def repo(tmp_path):
    return Repository(f"sqlite:///{tmp_path / 'db.sqlite'}", tmp_path / "artifacts")


def finish(repo, run_id, value):
    episode = {
        "index": 0,
        "seed": 1,
        "terminated": False,
        "truncated": True,
        "success": False,
        "n_steps": 1,
        "arrays": {"time": np.zeros(2)},
    }
    metrics = [
        {
            "metric_name": "return",
            "episode_index": None,
            "value": value,
            "unit": "r",
            "status": "ok",
        },
        {"metric_name": "return", "episode_index": 0, "value": 99.0, "unit": "r", "status": "ok"},
    ]
    repo.save_results(
        run_id,
        versions={},
        code_version="x",
        task_spec={},
        episodes=[episode],
        metrics=metrics,
        train_curve=[],
    )


def test_suite_experiments_runs_and_status(repo):
    suite, bench = repo.create_suite("s", {"seeds": [0, 1]})
    exp, run0 = repo.create_experiment(cfg(), 0, {"train": 1, "eval": 2}, suite_id=suite)
    run1 = repo.add_run(exp, 1, {"train": 3, "eval": 4})
    repo.update_benchmark_run(bench, status="running", progress=0.5)
    b = repo.get_benchmark_run(bench)
    assert (b["status"], b["progress"], b["suite_id"]) == ("running", 0.5, suite) and b[
        "started_at"
    ]
    repo.update_benchmark_run(bench, status="failed", error={"type": "X", "message": "m"})
    assert repo.get_benchmark_run(bench)["finished_at"]
    d = repo.get_suite(suite)
    assert d["spec"] == {"seeds": [0, 1]} and d["benchmark_runs"][0]["status"] == "failed"
    (e,) = d["experiments"]
    assert [(r["run_id"], r["seed"]) for r in e["runs"]] == [(run0, 0), (run1, 1)]
    assert repo.get_suite("nope") is None and repo.get_benchmark_run("nope") is None
    assert [s["suite_id"] for s in repo.list_suites()] == [suite]
    with pytest.raises(KeyError):
        repo.update_benchmark_run("nope", status="done")


def test_metric_rows_only_finished_runs_and_run_level_values(repo):
    suite, _ = repo.create_suite("s", {})
    exp, r0 = repo.create_experiment(cfg("env", "ctl"), 0, {"train": 1, "eval": 2}, suite_id=suite)
    r1 = repo.add_run(exp, 1, {"train": 3, "eval": 4})
    finish(repo, r0, 1.5)  # r1 stays queued -> excluded
    _, other = repo.create_experiment(cfg(), 0, {"train": 1, "eval": 2})  # not in the suite
    finish(repo, other, 7.0)
    rows = repo.suite_metric_rows(suite)
    assert rows == [
        {
            "env_id": "env",
            "controller_id": "ctl",
            "seed": 0,
            "run_id": r0,
            "metric_name": "return",
            "value": 1.5,
            "unit": "r",
            "status": "ok",
            "perturbations": [],
        }
    ]
    assert r1 not in {r["run_id"] for r in rows}


def test_phase1_database_gets_suite_column(tmp_path):
    url = f"sqlite:///{tmp_path / 'old.sqlite'}"
    old = (
        "CREATE TABLE experiment (id VARCHAR(32) PRIMARY KEY, name VARCHAR, env_id VARCHAR, "
        "controller_id VARCHAR, config JSON, config_hash VARCHAR, created_at DATETIME)"
    )
    with create_engine(url).begin() as c:
        c.execute(text(old))
    repo = Repository(url, tmp_path / "a")
    suite, _ = repo.create_suite("s", {})
    repo.create_experiment(cfg(), 0, {"train": 1, "eval": 2}, suite_id=suite)
    assert len(repo.get_suite(suite)["experiments"]) == 1


def test_fail_orphans_also_fails_unfinished_benchmark_runs(repo):
    _, running = repo.create_suite("s", {})
    _, done = repo.create_suite("s2", {})
    repo.update_benchmark_run(running, status="running")
    repo.update_benchmark_run(done, status="done")
    assert repo.fail_orphans() == 1
    b = repo.get_benchmark_run(running)
    assert b["status"] == "failed" and b["error"]["type"] == "interrupted" and b["finished_at"]
    assert repo.get_benchmark_run(done)["status"] == "done"
