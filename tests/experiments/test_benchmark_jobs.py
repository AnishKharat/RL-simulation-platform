from typing import Any, ClassVar

import numpy as np
import pytest

from rcl.controllers.classical.pid import PID, PIDParams
from rcl.core.controller import BaseController
from rcl.core.types import ActionSpaceKind, ControllerFamily, ControllerMetadata, EnvSpec
from rcl.environments.cartpole import CartPole, CartPoleParams
from rcl.experiments import BenchmarkSpec, JobManager
from rcl.persistence import Repository
from rcl.registry import Registry, discover, get_registry


class Boom(BaseController):
    """Valid controller whose predict always raises: every run of it must end `failed`."""

    metadata: ClassVar[ControllerMetadata] = ControllerMetadata(
        name="Boom",
        version="0.0.1",
        family=ControllerFamily.CLASSICAL,
        supported_action_spaces=(ActionSpaceKind.CONTINUOUS,),
        requires_training=False,
    )

    def configure(
        self, env_spec: EnvSpec, params: dict[str, Any], linear_model: Any = None
    ) -> None: ...
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Any:
        raise RuntimeError("boom")

    def save(self, path: Any) -> None: ...
    def load(self, path: Any) -> None: ...


@pytest.fixture
def repo(tmp_path):
    return Repository(f"sqlite:///{tmp_path / 'db.sqlite'}", tmp_path / "artifacts")


def spec(**over) -> BenchmarkSpec:
    d = {
        "name": "bench",
        "environments": [{"id": "cartpole"}],
        "controllers": [{"id": "pid"}, {"id": "lqr"}],
        "seeds": [0, 1],
        "episodes": 1,
        **over,
    }
    return BenchmarkSpec.model_validate(d)


def run(repo, registry, s) -> tuple[str, str]:
    jobs = JobManager(repo, registry, max_workers=1)
    try:
        suite, bench = repo.create_suite(s.name, s.model_dump(mode="json"))
        jobs.submit_benchmark(bench, suite, s).result(timeout=300)
    finally:
        jobs.shutdown()
    return suite, bench


def test_benchmark_persists_shared_experiments_and_all_seed_runs(repo):
    discover()
    suite, bench = run(repo, get_registry(), spec())
    b = repo.get_benchmark_run(bench)
    assert (b["status"], b["progress"], b["error"]) == ("done", 1.0, None)
    d = repo.get_suite(suite)
    assert d["spec"]["seeds"] == [0, 1]
    assert [(e["controller_id"], [r["seed"] for r in e["runs"]]) for e in d["experiments"]] == [
        ("pid", [0, 1]),
        ("lqr", [0, 1]),
    ]
    assert all(r["status"] == "done" for e in d["experiments"] for r in e["runs"])
    assert len({e["config_hash"] for e in d["experiments"]}) == 2  # one hash per cell, any seed
    rows = repo.suite_metric_rows(suite)
    assert {(r["controller_id"], r["seed"]) for r in rows} == {
        (c, s) for c in ("pid", "lqr") for s in (0, 1)
    }
    assert {"return", "settling_time", "control_effort_rate"} <= {r["metric_name"] for r in rows}


def test_failed_runs_do_not_stop_the_benchmark_and_are_excluded_from_metrics(repo):
    reg = Registry()
    reg.register("environment", "cartpole", CartPole, CartPoleParams)
    reg.register("controller", "pid", PID, PIDParams)
    reg.register("controller", "boom", Boom)
    suite, bench = run(repo, reg, spec(controllers=[{"id": "boom"}, {"id": "pid"}]))
    b = repo.get_benchmark_run(bench)
    assert b["status"] == "failed" and b["error"]["message"] == "2 of 4 runs failed"
    statuses = {
        (e["controller_id"], r["status"])
        for e in repo.get_suite(suite)["experiments"]
        for r in e["runs"]
    }
    assert statuses == {("boom", "failed"), ("pid", "done")}
    assert {r["controller_id"] for r in repo.suite_metric_rows(suite)} == {"pid"}


def test_unplannable_spec_marks_the_benchmark_failed(repo):
    discover()
    suite, bench = run(repo, get_registry(), spec(environments=[{"id": "nope"}]))
    b = repo.get_benchmark_run(bench)
    assert b["status"] == "failed" and b["error"]["type"] == "ConfigError"
    assert repo.get_suite(suite)["experiments"] == []
