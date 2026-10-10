import numpy as np
import pytest

from rcl.controllers.classical.pid import PID
from rcl.core.config import ExperimentConfig
from rcl.environments.cartpole import CartPole
from rcl.experiments import ConfigError, derive_seeds, execute, rollout
from rcl.registry import discover


@pytest.fixture(autouse=True, scope="module")
def _discovered():
    discover()


def cfg(env="cartpole", ctl="pid", **extra) -> ExperimentConfig:
    d = {
        "name": "t",
        "seed": 3,
        "environment": {"id": env},
        "controller": {"id": ctl},
        "evaluation": {"episodes": 2},
        **extra,
    }
    return ExperimentConfig.model_validate(d)


def test_rollout_contract_seeds_shapes_and_success():
    env, pid = CartPole(), PID()
    pid.configure(env.get_env_spec(), {})
    trajs = rollout(env, pid, 3, seed=10)
    assert [t.seed for t in trajs] == [10, 11, 12]
    for t in trajs:
        assert t.actions.shape == (t.n_steps, 1) and t.commanded_actions.shape == (t.n_steps, 1)
        assert len(t.states) == t.n_steps + 1 and not (t.terminated or t.truncated) == t.success


def test_rollout_failure_terminates_and_horizon_truncates():
    env = CartPole(max_steps=5)
    zero = PID()
    zero.configure(env.get_env_spec(), {"kp": 0.0, "kd": 0.0})
    (t,) = rollout(env, zero, 1, seed=0)
    assert t.truncated and not t.terminated and t.n_steps == 5

    long_env = CartPole()
    (t,) = rollout(long_env, zero, 1, seed=0)
    assert t.terminated and not t.truncated and not t.success  # falls over without control


def test_rollout_is_deterministic():
    env, pid = CartPole(), PID()
    pid.configure(env.get_env_spec(), {})
    a, b = rollout(env, pid, 2, 5), rollout(env, pid, 2, 5)
    assert all(np.array_equal(x.states, y.states) for x, y in zip(a, b, strict=True))


def test_execute_pid_result_record_and_metrics():
    r = execute(cfg(controller={"id": "pid", "params": {"kp": 55.0}}))
    rec = r.record
    assert rec.seed == 3 and set(rec.derived_seeds) == {"train", "eval"}
    assert rec.derived_seeds == derive_seeds(3)
    assert rec.config.controller.params["kp"] == 55.0 and "kd" in rec.config.controller.params
    assert rec.config.environment.params["force_mag"] == 10.0  # defaults resolved
    assert rec.config_hash == rec.config.config_hash() and "env:cartpole" in rec.versions
    assert r.train is None and len(r.trajectories) == 2 == len(r.episode_metrics)
    assert [m.name for m in r.metrics] == ["return", "angle_error"]


def test_same_seed_reproduces_and_seed_override_changes():
    a, b, c = execute(cfg()), execute(cfg()), execute(cfg(), seed=4)
    assert a.metrics == b.metrics
    assert all(
        np.array_equal(x.states, y.states)
        for x, y in zip(a.trajectories, b.trajectories, strict=True)
    )
    assert a.record.seed == 3 and c.record.seed == 4
    assert a.record.config_hash == c.record.config_hash  # hash excludes the seed
    assert not np.array_equal(a.trajectories[0].states[0], c.trajectories[0].states[0])


def test_execute_rejects_bad_configs_before_work():
    def types(**kw):
        with pytest.raises(ConfigError) as exc:
            execute(cfg(**kw))
        return {(tuple(e["loc"]), e["type"]) for e in exc.value.errors}

    assert types(env="cartpole_discrete") == {(("controller", "id"), "incompatible")}
    assert types(env="nope") == {(("environment", "id"), "unknown_id")}
    bad = types(controller={"id": "pid", "params": {"kq": 1}})
    assert bad == {(("controller", "params", "kq"), "extra_forbidden")}
    assert types(ctl="ppo") == {(("training",), "missing")}
    assert types(perturbations=[{"id": "noise"}]) == {(("perturbations",), "unsupported")}
    assert len(types(env="nope", ctl="nope")) == 2  # all problems reported at once


def test_execute_ppo_trains_reports_and_checkpoints(tmp_path):
    stages, ticks = [], []
    config = cfg(
        ctl="ppo",
        training={"total_timesteps": 64},
        evaluation={"episodes": 1},
    )
    config = config.model_copy(
        update={
            "controller": config.controller.model_copy(
                update={"params": {"n_steps": 32, "batch_size": 32, "n_epochs": 1}}
            )
        }
    )
    r = execute(
        config,
        checkpoint_dir=tmp_path / "ckpt",
        on_stage=stages.append,
        progress=lambda d, t: ticks.append(d),
    )
    assert stages == ["training", "evaluating"] and ticks and r.train.timesteps >= 64
    assert (tmp_path / "ckpt" / "model.zip").exists()
    again = execute(config)
    assert again.metrics == r.metrics  # trained policy + evaluation reproduce for the same seed
    assert again.record.versions["device"] == "cpu"
