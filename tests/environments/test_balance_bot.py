import numpy as np
import pytest

pytest.importorskip("pybullet")

from rcl.controllers.classical.lqr import LQR  # noqa: E402
from rcl.controllers.classical.pid import PID  # noqa: E402
from rcl.environments.balance_bot import BalanceBot  # noqa: E402
from rcl.experiments import rollout  # noqa: E402
from rcl.registry import discover, get_registry  # noqa: E402


@pytest.fixture
def env():
    e = BalanceBot()
    yield e
    e.close()


def test_registered_with_consistent_spec_labels_and_linearizability(env):
    discover()
    assert get_registry().get("environment", "balance_bot").cls is BalanceBot
    spec, meta = env.get_env_spec(), BalanceBot.env_metadata
    obs, _ = env.reset(seed=0)
    assert obs.shape == spec.obs_shape == (4,) and len(meta.state_labels) == spec.state_dim
    assert meta.physics_backend == "pybullet" and spec.dt == pytest.approx(0.02)
    assert (env.get_linearization() is not None) == meta.linearizable
    _, r, term, trunc, info = env.step(np.array([0.1]))
    assert info["applied_action"].shape == (1,) and isinstance(r, float) and not (term and trunc)


def test_seeded_determinism_across_instances(env):
    def run(e, seed):
        e.reset(seed=seed)
        return np.array([e.step(np.array([0.05]))[0] for _ in range(30)])

    other = BalanceBot()
    try:
        assert np.array_equal(run(env, 3), run(other, 3))
    finally:
        other.close()
    assert not np.array_equal(run(env, 3), run(env, 4))


def test_falls_without_control_and_truncates_at_horizon(env):
    env.reset(seed=0)
    for _ in range(200):
        obs, _, term, trunc, _ = env.step(np.zeros(1))
        if term or trunc:
            break
    assert term and abs(obs[0]) > 0.5
    short = BalanceBot(max_steps=3)
    try:
        lqr = LQR()
        lqr.configure(short.get_env_spec(), {}, short.get_linearization())
        (t,) = rollout(short, lqr, 1, seed=0)
        assert t.truncated and t.n_steps == 3
    finally:
        short.close()


def test_positive_torque_drives_forward_and_pitches_back(env):
    env.reset(seed=0)
    env._sim.set_state(env._full(np.zeros(4)))
    for _ in range(3):
        obs, *_ = env.step(np.array([0.3]))
    assert obs[3] > 0 and obs[0] < 0  # x_dot forward, chassis reacts backward


def test_linearisation_is_unstable_open_loop_and_leaves_the_episode_untouched(env):
    env.reset(seed=1)
    env.step(np.array([0.05]))
    before = env.get_state()
    lin = env.get_linearization()
    assert np.array_equal(env.get_state(), before)
    assert lin.A.shape == (4, 4) and lin.B.shape == (4, 1)
    assert np.max(np.abs(np.linalg.eigvals(lin.A))) > 1.0  # inverted pendulum
    assert lin.B[1, 0] < 0  # torque pitches the chassis back


@pytest.mark.parametrize(
    ("ctl", "params"),
    [(LQR, {}), (PID, {"obs_index": 0, "kp": 3.0, "kd": 0.3})],
)
def test_existing_classical_controllers_balance_it_unchanged(env, ctl, params):
    c = ctl()
    c.configure(env.get_env_spec(), params, env.get_linearization())
    trajs = rollout(env, c, 3, seed=10)
    assert all(not t.terminated for t in trajs)


def test_render_and_idempotent_close():
    e = BalanceBot()
    e.reset(seed=0)
    img = e.render()
    assert img.shape == (240, 320, 3) and img.dtype == np.uint8
    e.close()
    e.close()
