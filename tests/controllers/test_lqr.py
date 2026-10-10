import numpy as np
import pytest

from rcl.controllers.classical.lqr import LQR, solve_dlqr
from rcl.environments.cartpole import CartPole, CartPoleDiscrete
from rcl.environments.inverted_pendulum import InvertedPendulum
from rcl.experiments import rollout
from rcl.registry import discover, get_registry


def make(env, params=None) -> LQR:
    c = LQR()
    c.configure(env.get_env_spec(), params or {}, env.get_linearization())
    return c


def test_riccati_gain_stabilises_the_linear_model():
    lin = CartPole().get_linearization()
    k = solve_dlqr(lin.A, lin.B, np.eye(4), np.eye(1))
    assert np.max(np.abs(np.linalg.eigvals(lin.A - lin.B @ k))) < 1.0
    assert np.max(np.abs(np.linalg.eigvals(lin.A))) > 1.0  # open loop is unstable


@pytest.mark.parametrize("env_cls", [CartPole, InvertedPendulum])
def test_balances_nonlinear_plant(env_cls):
    env = env_cls()
    trajs = rollout(env, make(env), 3, seed=0)
    assert all(not t.terminated for t in trajs)
    assert all(t.success for t in trajs)  # held inside the TaskSpec tolerance


def test_action_shape_clipping_and_cost_weights():
    env = CartPole()
    c = make(env, {"r": 1e-3})
    u = c.predict(np.array([0.0, 0.0, 0.2, 0.0]))
    assert u.shape == (1,) and abs(float(u[0])) == 10.0  # saturates at force_mag
    heavy = make(env, {"q_diag": [1, 1, 100, 1]})
    light = make(env, {"q_diag": [1, 1, 1, 1]})
    obs = np.array([0.0, 0.0, 0.05, 0.0])
    assert abs(float(heavy.predict(obs)[0])) > abs(float(light.predict(obs)[0]))


def test_rejects_missing_model_discrete_and_bad_q():
    with pytest.raises(ValueError, match="linearisation"):
        LQR().configure(CartPole().get_env_spec(), {}, None)
    with pytest.raises(ValueError, match="continuous"):
        LQR().configure(CartPoleDiscrete().get_env_spec(), {}, CartPole().get_linearization())
    with pytest.raises(ValueError, match="q_diag needs 4"):
        make(CartPole(), {"q_diag": [1.0]})
    with pytest.raises(ValueError, match=">= 0"):
        make(CartPole(), {"q_diag": [1, 1, -1, 1]})


def test_save_load_roundtrip(tmp_path):
    env = CartPole()
    a = make(env)
    a.save(tmp_path)
    b = LQR()
    b.load(tmp_path)
    obs = np.array([0.1, -0.2, 0.05, 0.3])
    assert np.array_equal(a.predict(obs), b.predict(obs))


def test_registry_compatibility_follows_declared_linearizability():
    discover()
    reg = get_registry()
    assert reg.is_compatible("lqr", "cartpole") and reg.is_compatible("lqr", "inverted_pendulum")
    assert not reg.is_compatible("lqr", "cartpole_discrete")
