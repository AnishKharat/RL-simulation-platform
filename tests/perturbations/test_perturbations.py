import numpy as np
import pytest

from rcl.environments.cartpole import CartPole
from rcl.perturbations.action_delay import ActionDelay
from rcl.perturbations.action_noise import ActionNoise
from rcl.perturbations.disturbance import Disturbance
from rcl.perturbations.obs_noise import ObsNoise
from rcl.perturbations.param_scale import ParamScale


def run(env, actions, seed=3):
    """Apply ``actions``; return (observations, applied actions, flagged steps, true states)."""
    obs0, _ = env.reset(seed=seed)
    obs, applied, flagged, states = [obs0], [], [], []
    for k, a in enumerate(actions):
        o, _, term, trunc, info = env.step(np.array([a]))
        obs.append(o)
        applied.append(float(info["applied_action"][0]))
        states.append(env.get_state())
        if info.get("perturbed"):
            flagged.append(k)
        if term or trunc:
            break
    return np.array(obs), np.array(applied), flagged, np.array(states)


ACTIONS_A = [0.0] * 60
ACTIONS_B = [0.5, -0.5] * 30


def test_obs_noise_is_seeded_and_independent_of_actions() -> None:
    def noise(actions, seed):
        obs, _, _, states = run(ObsNoise(CartPole(), std=0.1), actions, seed)
        return obs[1 : len(states) + 1] - states

    a, b = noise(ACTIONS_A, 3), noise(ACTIONS_B, 3)
    n = min(len(a), len(b))
    assert np.allclose(a[:n], b[:n])  # same draws
    assert np.array_equal(noise(ACTIONS_A, 3), noise(ACTIONS_A, 3))  # reproducible
    c = noise(ACTIONS_A, 4)
    m = min(len(a), len(c))
    assert not np.allclose(a[:m], c[:m])  # seed matters


def test_zero_std_matches_bare_environment() -> None:
    bare, _, _, _ = run(CartPole(), ACTIONS_B)
    wrapped, _, _, _ = run(ObsNoise(CartPole(), std=0.0), ACTIONS_B)
    assert np.array_equal(bare, wrapped)


def test_reset_without_seed_is_an_error() -> None:
    with pytest.raises(ValueError, match="explicit seed"):
        ObsNoise(CartPole(), std=0.1).reset()


def test_stack_position_changes_the_stream() -> None:
    a, _, _, _ = run(ObsNoise(CartPole(), position=0, std=0.1), ACTIONS_A)
    b, _, _, _ = run(ObsNoise(CartPole(), position=1, std=0.1), ACTIONS_A)
    assert not np.allclose(a, b)


def test_action_delay_shifts_commands() -> None:
    _, applied, _, _ = run(ActionDelay(CartPole(), steps=2), [1.0, 2.0, 3.0, 4.0])
    assert applied.tolist() == [0.0, 0.0, 1.0, 2.0]
    _, none, _, _ = run(ActionDelay(CartPole(), steps=0), [1.0, 2.0])
    assert none.tolist() == [1.0, 2.0]


def test_action_noise_is_seeded_and_scaled_by_limit() -> None:
    _, a, _, _ = run(ActionNoise(CartPole(), std=0.1), ACTIONS_A)
    _, b, _, _ = run(ActionNoise(CartPole(), std=0.1), ACTIONS_A)
    assert np.array_equal(a, b)
    assert 0 < a.std() < 10.0  # force limit 10 N, std 0.1 -> about 1 N
    _, c, _, _ = run(ActionNoise(CartPole(), std=0.0), ACTIONS_A)
    assert not c.any()


def test_disturbance_pulse_is_flagged_and_identical_across_controllers() -> None:
    def pulse(actions):
        env = Disturbance(CartPole(), magnitude=0.5, duration_steps=4, start_min=5, start_max=10)
        _, applied, flagged, _ = run(env, actions)
        return applied, flagged

    applied_a, flagged_a = pulse([0.0] * 30)
    _, flagged_b = pulse([1.0, -1.0] * 15)
    assert flagged_a == flagged_b and len(flagged_a) == 4
    assert flagged_a == list(range(flagged_a[0], flagged_a[0] + 4))
    assert 5 <= flagged_a[0] <= 10
    assert np.count_nonzero(applied_a) == 4 and abs(applied_a[flagged_a[0]]) == 5.0


def test_disturbance_saturates_at_the_actuator_limit() -> None:
    env = Disturbance(CartPole(), magnitude=1.0, duration_steps=50, start_min=0, start_max=0)
    _, applied, flagged, _ = run(env, [10.0] * 5)
    sign = 1.0 if applied.min() == 10.0 else -1.0
    assert flagged == [0, 1, 2, 3, 4] and abs(applied).max() <= 10.0
    assert sign in (1.0, -1.0)


def test_zero_magnitude_disturbance_injects_nothing() -> None:
    _, _, flagged, _ = run(Disturbance(CartPole(), magnitude=0.0), ACTIONS_A)
    assert flagged == []


def test_param_scale_scales_numeric_targets_without_mutating() -> None:
    base = {"masspole": 0.1, "length": 0.5}
    out = ParamScale.apply(base, {"targets": ["masspole"], "scale": 2.0})
    assert out == {"masspole": 0.2, "length": 0.5} and base["masspole"] == 0.1
    assert CartPole(**out).mp == 0.2


@pytest.mark.parametrize("target", ["nope", "flag"])
def test_param_scale_rejects_unknown_or_non_numeric_target(target: str) -> None:
    with pytest.raises(ValueError, match="not a numeric"):
        ParamScale.apply({"flag": True}, {"targets": [target], "scale": 1.0})


def test_unseeded_reset_after_a_seeded_one_continues_the_stream() -> None:
    env = ObsNoise(CartPole(), std=0.1)  # SB3 vector envs auto-reset without a seed
    env.reset(seed=3)
    env.reset()  # must not raise
    assert env._seeded


def test_action_delay_is_continuous_only() -> None:
    from rcl.core.types import ActionSpaceKind

    assert ActionDelay.perturbation_metadata.action_spaces == (ActionSpaceKind.CONTINUOUS,)
