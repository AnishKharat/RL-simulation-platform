"""Torque-driven inverted pendulum (balance at the upright equilibrium).

State ``[theta, theta_dot]``, theta = 0 upright; positive torque increases theta (so a stabilising
controller applies torque opposite to the lean). Semi-implicit Euler integration. Reward is
``1 - cost`` (higher is better, so falling early is never attractive). Terminated beyond
``THETA_LIMIT``; truncated at ``max_steps`` (= ``TaskSpec.horizon_steps``).
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from rcl.core.environment import BaseEnvironment
from rcl.core.types import ActionSpaceKind, EnvMetadata, EnvSpec, TaskSpec
from rcl.registry import register_environment

THETA_LIMIT = 0.8  # rad; beyond this the pendulum counts as fallen


class InvertedPendulumParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gravity: float = Field(9.81, gt=0)
    mass: float = Field(1.0, gt=0)
    length: float = Field(1.0, gt=0)
    damping: float = Field(0.05, ge=0, description="viscous friction at the pivot")
    max_torque: float = Field(5.0, gt=0)
    dt: float = Field(0.02, gt=0, le=0.1)
    max_steps: int = Field(500, ge=1, le=100000)


@register_environment("inverted_pendulum", params=InvertedPendulumParams)
class InvertedPendulum(BaseEnvironment):
    env_metadata: ClassVar[EnvMetadata] = EnvMetadata(
        name="Inverted Pendulum",
        version="0.1.0",
        description="Balance a pendulum upright with a bounded torque at the pivot.",
        dof=1,
        action_space_kind=ActionSpaceKind.CONTINUOUS,
        physics_backend="analytic",
        state_labels=("theta", "theta_dot"),
        state_units=("rad", "rad/s"),
        action_labels=("torque",),
        action_units=("N*m",),
    )
    metadata = {"render_modes": []}  # noqa: RUF012 - gymnasium convention

    def __init__(
        self,
        gravity: float = 9.81,
        mass: float = 1.0,
        length: float = 1.0,
        damping: float = 0.05,
        max_torque: float = 5.0,
        dt: float = 0.02,
        max_steps: int = 500,
    ) -> None:
        super().__init__()
        self.g, self.m, self.l, self.b = gravity, mass, length, damping
        self.max_torque, self.dt, self.max_steps = max_torque, dt, max_steps
        high = np.array([np.inf] * 2)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float64)
        self.action_space = gym.spaces.Box(-max_torque, max_torque, (1,), np.float64)
        self._state = np.zeros(2)
        self._t = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._state = self.np_random.uniform(-0.1, 0.1, size=2)
        self._t = 0
        return self._state.copy(), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        u = float(
            np.clip(
                np.asarray(action, dtype=np.float64).reshape(-1)[0],
                -self.max_torque,
                self.max_torque,
            )
        )
        th, thd = self._state
        inertia = self.m * self.l**2
        thd = thd + self.dt * (
            (self.m * self.g * self.l * math.sin(th) + u - self.b * thd) / inertia
        )
        th = th + self.dt * thd
        self._state = np.array([th, thd])
        self._t += 1
        cost = (th / THETA_LIMIT) ** 2 + 0.01 * thd**2 + 0.001 * u**2
        terminated = bool(abs(th) > THETA_LIMIT)
        truncated = bool(not terminated and self._t >= self.max_steps)
        info = {"applied_action": np.array([u])}
        return self._state.copy(), 1.0 - cost, terminated, truncated, info

    def get_state(self) -> np.ndarray:
        return self._state.copy()

    def get_env_spec(self) -> EnvSpec:
        return EnvSpec(
            env_id="inverted_pendulum",
            action_space_kind=ActionSpaceKind.CONTINUOUS,
            obs_shape=(2,),
            state_dim=2,
            dt=self.dt,
            action_shape=(1,),
            action_low=(-self.max_torque,),
            action_high=(self.max_torque,),
        )

    def get_task_spec(self) -> TaskSpec:
        # Success: within 0.02 rad for 2 s; velocity unconstrained (1e9).
        return TaskSpec(
            setpoint=(0.0, 0.0),
            tolerance=(0.02, 1e9),
            error_state_indices=(0,),
            wrap_indices=(0,),
            output_unit="rad",
            step_floor=0.05,
            horizon_steps=self.max_steps,
            success_hold_steps=100,
            failure_description=f"pendulum beyond {THETA_LIMIT} rad from upright",
        )
