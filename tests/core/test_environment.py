from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from rcl.core.environment import BaseEnvironment, EnvWrapper
from rcl.core.types import ActionSpaceKind, EnvMetadata, EnvSpec, LinearModel, TaskSpec


def test_cannot_instantiate_abstract() -> None:
    with pytest.raises(TypeError):
        BaseEnvironment()  # type: ignore[abstract]


class _Dummy(BaseEnvironment):
    env_metadata = EnvMetadata(
        name="dummy",
        version="0.1.0",
        dof=1,
        action_space_kind=ActionSpaceKind.CONTINUOUS,
        physics_backend="none",
    )

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(-1, 1, (1,))
        self.action_space = gym.spaces.Box(-1, 1, (1,))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        super().reset(seed=seed)
        return np.zeros(1), {}

    def step(self, action: Any) -> Any:
        return np.zeros(1), 0.0, False, False, {"applied_action": np.asarray(action, float)}

    def get_env_spec(self) -> EnvSpec:
        return EnvSpec(
            env_id="dummy",
            action_space_kind=ActionSpaceKind.CONTINUOUS,
            obs_shape=(1,),
            state_dim=1,
            dt=0.1,
            action_shape=(1,),
            action_low=(-1.0,),
            action_high=(1.0,),
        )

    def get_state(self) -> np.ndarray:
        return np.zeros(1)

    def get_task_spec(self) -> TaskSpec:
        return TaskSpec(
            setpoint=(0.0,),
            tolerance=(0.1,),
            error_state_indices=(0,),
            step_floor=0.05,
            horizon_steps=5,
        )


class _Lin(_Dummy):
    env_metadata = _Dummy.env_metadata.model_copy(update={"linearizable": True})

    def get_linearization(self) -> LinearModel | None:
        return LinearModel(np.eye(1), np.ones((1, 1)), np.zeros(1), np.zeros(1), 0.1)


@pytest.mark.parametrize("cls", [_Dummy, _Lin])
def test_linearizable_flag_agrees_with_get_linearization(cls: type[_Dummy]) -> None:
    """Every environment's test suite should carry this check (ADR 0014)."""
    env = cls()
    assert env.env_metadata.linearizable == (env.get_linearization() is not None)


def test_simulated_defaults_true_and_hardware_round_trips() -> None:
    assert _Dummy.env_metadata.simulated is True
    hw = _Dummy.env_metadata.model_copy(update={"simulated": False})
    assert EnvMetadata.model_validate_json(hw.model_dump_json()).simulated is False


def test_concrete_subclass_defaults() -> None:
    env = _Dummy()
    assert env.get_linearization() is None
    env.reset(seed=3)
    first = env.np_random.random()
    env.reset(seed=3)
    assert env.np_random.random() == first


class _Offset(EnvWrapper):
    """Perturbs observations only."""

    def step(self, action: Any) -> Any:
        obs, r, term, trunc, info = self.env.step(action)
        return obs + 1.0, r, term, trunc, info


def test_env_wrapper_delegates_and_keeps_true_state() -> None:
    inner = _Dummy()
    w = _Offset(inner)
    assert isinstance(w, BaseEnvironment)
    assert w.env_metadata is _Dummy.env_metadata
    assert w.get_env_spec() == inner.get_env_spec()
    assert w.get_task_spec() == inner.get_task_spec()
    assert w.get_linearization() is None
    w.reset(seed=1)
    obs, *_ = w.step(np.zeros(1))
    assert obs[0] == 1.0  # perturbed observation
    assert w.get_state()[0] == 0.0  # true state untouched


def test_env_wrapper_preserves_applied_action() -> None:
    info = _Offset(_Dummy()).step(np.array([0.5]))[4]
    assert info["applied_action"].shape == _Dummy().get_env_spec().action_shape
    assert info["applied_action"][0] == 0.5


def test_env_wrapper_rejects_plain_gym_env() -> None:
    with pytest.raises(TypeError):
        EnvWrapper(gym.make("CartPole-v1"))  # type: ignore[arg-type]


_BASE: dict[str, Any] = {
    "name": "e",
    "version": "0.1.0",
    "dof": 2,
    "action_space_kind": ActionSpaceKind.CONTINUOUS,
    "physics_backend": "none",
}


def test_metadata_labels_valid_and_default_empty() -> None:
    assert EnvMetadata(**_BASE).state_labels == ()
    md = EnvMetadata(
        **_BASE,
        action_dim=1,
        state_labels=("x", "x_dot", "th", "th_dot"),
        state_units=("m", "m/s", "rad", "rad/s"),
        action_labels=("force",),
        action_units=("N",),
    )
    assert EnvMetadata.model_validate_json(md.model_dump_json()) == md


@pytest.mark.parametrize(
    "kw",
    [
        {"state_labels": ("x", "v"), "state_units": ("m",)},
        {"state_labels": ("x",), "state_units": ("m",)},  # fewer than dof=2
        {"action_labels": ("f",), "action_units": ("N", "N")},
        {"action_labels": ("f", "g"), "action_units": ("N", "N")},  # action_dim=1
    ],
)
def test_metadata_label_validation(kw: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        EnvMetadata(**_BASE, **kw)


def test_task_spec_output_unit_optional() -> None:
    kw: dict[str, Any] = {
        "setpoint": (0.0,),
        "tolerance": (0.1,),
        "error_state_indices": (0,),
        "step_floor": 0.05,
        "horizon_steps": 5,
    }
    assert TaskSpec(**kw).output_unit is None
    assert TaskSpec(**kw, output_unit="rad").output_unit == "rad"
