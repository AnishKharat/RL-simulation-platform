import numpy as np
import pytest

from rcl.controllers.classical.pid import PID
from rcl.environments.cartpole import CartPole, CartPoleDiscrete
from rcl.environments.inverted_pendulum import InvertedPendulum
from rcl.registry import discover, get_registry


def run(env, params, steps=300, seed=0):
    c = PID()
    c.configure(env.get_env_spec(), params)
    obs, _ = env.reset(seed=seed)
    c.reset()
    for _ in range(steps):
        a = c.predict(obs)
        assert a.shape == env.get_env_spec().action_shape
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            return term
    return False


def test_balances_cartpole_and_pendulum():
    assert run(CartPole(), {}) is False
    assert run(InvertedPendulum(), {"kp": -30.0, "kd": -6.0, "obs_index": 0}) is False


def test_rejects_discrete_and_bad_index():
    with pytest.raises(ValueError):
        PID().configure(CartPoleDiscrete().get_env_spec(), {})
    with pytest.raises(ValueError):
        PID().configure(CartPole().get_env_spec(), {"obs_index": 9})


def test_output_clipped_and_anti_windup():
    c = PID()
    c.configure(CartPole().get_env_spec(), {"kp": 1000.0, "ki": 5.0, "integral_limit": 0.05})
    for _ in range(50):
        u = c.predict(np.array([0, 0, 0.1, 0.0]))
    assert abs(float(u[0])) == 10.0 and c._integral == 0.05
    c.reset()
    assert c._integral == 0.0


def test_save_load_roundtrip(tmp_path):
    env = CartPole()
    a = PID()
    a.configure(env.get_env_spec(), {"kp": 40.0})
    a.save(tmp_path)
    b = PID()
    b.load(tmp_path)
    obs = np.array([0.0, 0.1, 0.05, 0.2])
    assert np.array_equal(a.predict(obs), b.predict(obs))


def test_registered_and_compatible():
    discover()
    reg = get_registry()
    assert reg.is_compatible("pid", "cartpole") and reg.is_compatible("pid", "inverted_pendulum")
    assert not reg.is_compatible("pid", "cartpole_discrete")
