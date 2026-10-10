import numpy as np
import pytest

from rcl.environments.cartpole import CartPole, CartPoleDiscrete
from rcl.environments.inverted_pendulum import InvertedPendulum
from rcl.registry import discover, get_registry

CASES = [
    ("cartpole", CartPole, np.array([5.0])),
    ("cartpole_discrete", CartPoleDiscrete, 1),
    ("inverted_pendulum", InvertedPendulum, np.array([1.0])),
]


@pytest.mark.parametrize(("env_id", "cls", "action"), CASES)
def test_shapes_spec_and_labels(env_id, cls, action):
    env = cls()
    obs, _ = env.reset(seed=1)
    spec, meta = env.get_env_spec(), cls.env_metadata
    assert spec.env_id == env_id and obs.shape == spec.obs_shape
    assert len(meta.state_labels) == spec.state_dim == env.get_state().shape[0]
    assert (spec.action_space_kind == meta.action_space_kind) and env.get_linearization() is None
    assert not meta.linearizable
    obs, r, term, trunc, info = env.step(action)
    assert obs.shape == spec.obs_shape and isinstance(r, float) and not (term and trunc)
    assert info["applied_action"].shape == spec.action_shape
    assert env.get_task_spec().horizon_steps == env.max_steps


@pytest.mark.parametrize(("env_id", "cls", "action"), CASES)
def test_seeded_determinism(env_id, cls, action):
    def run(seed):
        env = cls()
        env.reset(seed=seed)
        return np.array([env.step(action)[0] for _ in range(20)])

    assert np.array_equal(run(3), run(3)) and not np.array_equal(run(3), run(4))


@pytest.mark.parametrize(("env_id", "cls", "action"), CASES)
def test_registered_by_discovery(env_id, cls, action):
    discover()
    assert get_registry().get("environment", env_id).cls is cls


def test_truncates_at_horizon_and_terminates_on_fall():
    env = InvertedPendulum(max_steps=3, max_torque=1.0)
    env.reset(seed=0)
    flags = [env.step(np.zeros(1))[2:4] for _ in range(3)]
    assert flags[-1] == (False, True)
    cp = CartPoleDiscrete()
    cp.reset(seed=0)
    assert any(cp.step(1)[2] for _ in range(100))  # constant push topples the pole


def test_discrete_action_mapping():
    env = CartPoleDiscrete(force_mag=7.0)
    env.reset(seed=0)
    assert env.step(0)[4]["applied_action"] == -7.0 and env.step(1)[4]["applied_action"] == 7.0
