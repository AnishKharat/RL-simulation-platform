import numpy as np
import pytest

from rcl.experiments import BenchmarkSpec, ConfigError, plan, run_benchmark
from rcl.registry import discover, get_registry

PPO_TINY = {"n_steps": 32, "batch_size": 32, "n_epochs": 1}


@pytest.fixture(autouse=True, scope="module")
def _discovered():
    discover()


def spec(**over) -> BenchmarkSpec:
    d = {
        "name": "b",
        "environments": [{"id": "cartpole"}],
        "controllers": [{"id": "pid"}, {"id": "lqr"}],
        "seeds": [0, 1],
        "episodes": 2,
        **over,
    }
    return BenchmarkSpec.model_validate(d)


def test_grid_cells_skips_and_seed_order():
    s = spec(environments=[{"id": "cartpole"}, {"id": "cartpole_discrete"}])
    r = run_benchmark(s)
    assert [(c.env_id, c.controller_id) for c in r.cells] == [
        ("cartpole", "pid"),
        ("cartpole", "lqr"),
    ]
    assert [(k.env_id, k.controller_id) for k in r.skipped] == [
        ("cartpole_discrete", "pid"),
        ("cartpole_discrete", "lqr"),
    ]
    assert all("not compatible" in k.reason for k in r.skipped)
    for c in r.cells:
        assert [run.record.seed for run in c.runs] == [0, 1]
        assert all(len(run.trajectories) == 2 for run in c.runs)


def test_fairness_same_initial_states_and_definition_for_every_controller():
    r = run_benchmark(spec())
    pid, lqr = r.cells
    for a, b in zip(pid.runs, lqr.runs, strict=True):
        assert a.record.derived_seeds == b.record.derived_seeds
        for ta, tb in zip(a.trajectories, b.trajectories, strict=True):
            assert ta.seed == tb.seed and np.array_equal(ta.states[0], tb.states[0])
        assert a.task == b.task  # same horizon / tolerance
        assert a.record.config.environment == b.record.config.environment
        assert a.record.config.evaluation == b.record.config.evaluation
    s0, s1 = pid.runs[0].trajectories[0].states[0], pid.runs[1].trajectories[0].states[0]
    assert not np.array_equal(s0, s1)  # different run seeds -> different episodes


def test_same_spec_reproduces_the_same_metrics():
    def table(r):
        return [
            (c.controller_id, [(m.name, m.value) for run in c.runs for m in run.metrics])
            for c in r.cells
        ]

    assert table(run_benchmark(spec())) == table(run_benchmark(spec()))


def test_rl_controller_trained_with_shared_budget_and_progress_stop():
    s = spec(
        controllers=[{"id": "pid"}, {"id": "ppo", "params": PPO_TINY}],
        seeds=[0, 1],
        episodes=1,
        training={"total_timesteps": 64},
    )
    seen = []
    r = run_benchmark(s, progress=lambda d, t: seen.append((d, t)) or None)
    assert seen[-1] == (4, 4)
    ppo = r.cells[1].runs[0]
    assert ppo.train is not None and ppo.train.timesteps >= 64 and r.cells[0].runs[0].train is None
    stopped = run_benchmark(s, progress=lambda d, t: d >= 1)
    assert [len(c.runs) for c in stopped.cells] == [1]


def test_invalid_specs_are_rejected_before_any_run():
    with pytest.raises(ConfigError) as exc:
        run_benchmark(spec(controllers=[{"id": "ppo"}]))  # needs a training section
    assert exc.value.errors[0]["type"] == "missing" and exc.value.errors[0]["cell"] == [
        "cartpole",
        "ppo",
    ]
    with pytest.raises(ConfigError):
        run_benchmark(spec(environments=[{"id": "nope"}]))
    with pytest.raises(ConfigError, match="no compatible"):
        run_benchmark(spec(environments=[{"id": "cartpole_discrete"}]))
    for bad in ({"seeds": [1, 1]}, {"seeds": [-1]}, {"seeds": []}, {"episodes": 0}):
        with pytest.raises(ValueError):
            spec(**bad)
    with pytest.raises(ValueError, match="unique"):
        spec(controllers=[{"id": "pid"}, {"id": "pid"}])


def test_plan_uses_given_registry():
    assert get_registry().get("controller", "lqr")
    pairs, skipped = plan(spec(), get_registry())
    assert len(pairs) == 2 and skipped == []
