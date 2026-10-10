import numpy as np
import pytest

from rcl.controllers.rl.ppo import PPO
from rcl.core.types import TrainConfig
from rcl.environments.cartpole import CartPole, CartPoleDiscrete
from rcl.environments.inverted_pendulum import InvertedPendulum
from rcl.registry import discover, get_registry

PARAMS = {"n_steps": 32, "batch_size": 32, "n_epochs": 1}
CFG = TrainConfig(total_timesteps=64, seed=7)
OBS = np.array([0.0, 0.1, 0.05, -0.2])


def trained(env_cls, seed=7, progress=None):
    c = PPO()
    c.configure(env_cls().get_env_spec(), PARAMS)
    res = c.train(env_cls, CFG.model_copy(update={"seed": seed}), progress)
    return c, res


@pytest.mark.parametrize("env_cls", [CartPole, CartPoleDiscrete])
def test_train_predict_shapes(env_cls):
    c, res = trained(env_cls)
    spec = env_cls().get_env_spec()
    a = c.predict(OBS)
    assert res.timesteps >= 64
    assert isinstance(a, int) if spec.n_actions else a.shape == spec.action_shape


def test_same_seed_same_policy_different_seed_differs():
    a, b, c = trained(CartPole)[0], trained(CartPole)[0], trained(CartPole, seed=8)[0]
    assert np.array_equal(a.predict(OBS), b.predict(OBS))
    assert not np.array_equal(a.predict(OBS), c.predict(OBS))


def test_save_load_roundtrip(tmp_path):
    a = trained(CartPole)[0]
    a.save(tmp_path)
    b = PPO()
    b.load(tmp_path)
    assert np.array_equal(a.predict(OBS), b.predict(OBS))


def test_progress_callback_and_cooperative_stop():
    calls = []
    _, res = trained(InvertedPendulum, progress=lambda d, t: calls.append((d, t)) or None)
    assert calls[-1] == (res.timesteps, 64)

    c = PPO()
    c.configure(CartPole().get_env_spec(), PARAMS)
    res = c.train(CartPole, TrainConfig(total_timesteps=10_000, seed=0), lambda d, t: True)
    assert res.timesteps < 10_000


def test_predict_before_train_and_registry():
    with pytest.raises(RuntimeError):
        PPO().predict(OBS)
    discover()
    reg = get_registry()
    assert reg.is_compatible("ppo", "cartpole") and reg.is_compatible("ppo", "cartpole_discrete")
