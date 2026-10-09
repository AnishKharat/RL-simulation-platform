from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from rcl.core.environment import BaseEnvironment
from rcl.core.types import ActionSpaceKind, EnvMetadata, EnvSpec, TaskSpec


def test_cannot_instantiate_abstract() -> None:
    with pytest.raises(TypeError):
        BaseEnvironment()  # type: ignore[abstract]


class _Dummy(BaseEnvironment):
    env_metadata = EnvMetadata(
        name="dummy", dof=1, action_space_kind=ActionSpaceKind.CONTINUOUS, physics_backend="none"
    )

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(-1, 1, (1,))
        self.action_space = gym.spaces.Box(-1, 1, (1,))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        super().reset(seed=seed)
        return np.zeros(1), {}

    def step(self, action: Any) -> Any:
        return np.zeros(1), 0.0, False, False, {}

    def get_env_spec(self) -> EnvSpec:
        return EnvSpec(
            env_id="dummy",
            action_space_kind=ActionSpaceKind.CONTINUOUS,
            obs_shape=(1,),
            state_dim=1,
            dt=0.1,
        )

    def get_state(self) -> np.ndarray:
        return np.zeros(1)

    def get_task_spec(self) -> TaskSpec:
        return TaskSpec(
            setpoint=(0.0,),
            tolerance=(0.1,),
            error_state_index=0,
            step_floor=0.05,
            horizon_steps=5,
        )


def test_concrete_subclass_defaults() -> None:
    env = _Dummy()
    assert env.get_linearization() is None
    env.reset(seed=3)
    first = env.np_random.random()
    env.reset(seed=3)
    assert env.np_random.random() == first
