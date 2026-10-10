import numpy as np
import pytest

from rcl.controllers.rl.td3 import TD3
from rcl.core.types import TrainConfig
from rcl.environments.cartpole import CartPole, CartPoleDiscrete
from rcl.environments.inverted_pendulum import InvertedPendulum
from rcl.registry import discover, get_registry

PARAMS = {"buffer_size": 1000, "learning_starts": 16, "batch_size": 16}
CFG = TrainConfig(total_timesteps=96, seed=7)
OBS = np.array([0.0, 0.1, 0.05, -0.2])


def trained(env_cls, seed=7, progress=None):
    c = TD3()
    c.configure(env_cls().get_env_spec(), PARAMS)
    res = c.train(env_cls, CFG.model_copy(update={"seed": seed}), progress)
    return c, res


@pytest.mark.parametrize("env_cls", [CartPole, InvertedPendulum])
def test_train_predict_shapes(env_cls):
    c, res = trained(env_cls)
    spec = env_cls().get_env_spec()
    a = c.predict(env_cls().reset(seed=0)[0])
    assert res.timesteps >= 96
    assert a.shape == spec.action_shape and a.dtype == np.float64


def test_same_seed_same_policy_different_seed_differs():
    a, b, c = trained(CartPole)[0], trained(CartPole)[0], trained(CartPole, seed=8)[0]
    assert np.array_equal(a.predict(OBS), b.predict(OBS))
    assert not np.array_equal(a.predict(OBS), c.predict(OBS))


def test_save_load_roundtrip(tmp_path):
    a = trained(CartPole)[0]
    a.save(tmp_path)
    b = TD3()
    b.load(tmp_path)
    assert np.array_equal(a.predict(OBS), b.predict(OBS))


def test_progress_callback_and_cooperative_stop():
    calls = []
    _, res = trained(InvertedPendulum, progress=lambda d, t: calls.append((d, t)) or None)
    assert calls[-1] == (res.timesteps, 96)

    c = TD3()
    c.configure(CartPole().get_env_spec(), PARAMS)
    res = c.train(CartPole, TrainConfig(total_timesteps=10_000, seed=0), lambda d, t: True)
    assert res.timesteps < 100  # stops at the first opportunity, not at the budget


def test_predict_before_train_discrete_rejected_and_registry():
    with pytest.raises(RuntimeError):
        TD3().predict(OBS)
    with pytest.raises(ValueError):
        TD3().configure(CartPoleDiscrete().get_env_spec(), {})
    discover()
    reg = get_registry()
    assert reg.is_compatible("td3", "cartpole") and not reg.is_compatible(
        "td3", "cartpole_discrete"
    )


def test_exploration_noise_is_a_parameter_and_changes_training():
    quiet, noisy = (
        TD3(),
        TD3(),
    )
    for c, noise in ((quiet, 0.0), (noisy, 0.5)):
        c.configure(CartPole().get_env_spec(), {**PARAMS, "exploration_noise": noise})
        c.train(CartPole, CFG)
    assert not np.array_equal(quiet.predict(OBS), noisy.predict(OBS))
    with pytest.raises(ValueError):
        TD3().configure(CartPole().get_env_spec(), {"exploration_noise": -1.0})


def test_curve_is_thinned_to_about_200_points():
    c = TD3()
    c.configure(CartPole().get_env_spec(), PARAMS)
    res = c.train(CartPole, TrainConfig(total_timesteps=600, seed=0))
    assert 1 < len(res.curve) <= 205  # SB3 ends a rollout every step for TD3 (train_freq=1)
    steps = [s for s, _ in res.curve]
    assert steps == sorted(steps) and all(
        b - a >= 3 for a, b in zip(steps, steps[1:], strict=False)
    )
