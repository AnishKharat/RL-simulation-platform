import pytest

from rcl.analytics import degradation_curves
from rcl.core.config import PerturbationConfig
from rcl.experiments import BenchmarkSpec, ConfigError, JobManager, execute, execute_sweep, plan
from rcl.persistence import Repository
from rcl.registry import discover, get_registry

LEVELS = [0.0, 0.5, 2.0]


def stack(std):
    return [PerturbationConfig(id="action_noise", params={"std": std}, apply_to="eval")]


def spec(**over):
    d = {
        "name": "rob",
        "environments": [{"id": "cartpole"}],
        "controllers": [{"id": "pid"}],
        "seeds": [0, 1],
        "episodes": 2,
        "perturbation": {"id": "action_noise", "levels": [0.5, 2.0]},
        **over,
    }
    return BenchmarkSpec.model_validate(d)


@pytest.fixture
def repo(tmp_path):
    return Repository(f"sqlite:///{tmp_path / 'db.sqlite'}", tmp_path / "artifacts")


def config(sp):
    return sp.experiment(sp.environments[0], sp.controllers[0], 0)


def test_sweep_levels_share_eval_seeds_and_nominal_equals_a_plain_run():
    discover()
    cfg = config(spec())
    results = execute_sweep(cfg, [stack(s) for s in LEVELS], seed=0)
    assert [r.record.config.perturbations[0].params["std"] for r in results] == LEVELS
    assert len({r.record.config_hash for r in results}) == 3
    assert {tuple(t.seed for t in r.trajectories) for r in results} == {
        tuple(t.seed for t in results[0].trajectories)
    }
    plain = execute(cfg, seed=0)
    assert [(m.name, m.value) for m in results[0].metrics] == [
        (m.name, m.value) for m in plain.metrics
    ]
    noisy = {m.name: m.value for m in results[2].metrics}
    assert noisy["angle_error"] != {m.name: m.value for m in plain.metrics}["angle_error"]


def test_plan_adds_the_nominal_level_and_rejects_bad_perturbations():
    discover()
    reg = get_registry()
    pairs, _ = plan(spec(), reg)
    assert len(pairs) == 1
    with pytest.raises(ConfigError) as e:
        plan(spec(perturbation={"id": "nope", "levels": [1.0]}), reg)
    assert e.value.errors[0]["loc"] == ["perturbation", "id"]
    with pytest.raises(ConfigError) as e:
        plan(spec(perturbation={"id": "action_noise", "levels": [-1.0]}), reg)
    assert e.value.errors[0]["loc"][0] == "perturbation"


def test_stored_sweep_makes_one_experiment_per_level_and_a_degradation_curve(repo):
    discover()
    sp = spec()
    jobs = JobManager(repo, get_registry(), max_workers=1)
    try:
        suite, bench = repo.create_suite(sp.name, sp.model_dump(mode="json"))
        jobs.submit_benchmark(bench, suite, sp).result(timeout=300)
    finally:
        jobs.shutdown()
    assert repo.get_benchmark_run(bench)["status"] == "done"
    exps = repo.get_suite(suite)["experiments"]
    assert len(exps) == 3 and all(len(e["runs"]) == 2 for e in exps)
    assert all(r["status"] == "done" for e in exps for r in e["runs"])
    (curve,) = degradation_curves(
        repo.suite_metric_rows(suite), "action_noise", "std", 0.0, "angle_error"
    )
    assert [p.level for p in curve.points] == [0.0, 0.5, 2.0]
    assert all(p.summary.n_valid == 2 for p in curve.points)
