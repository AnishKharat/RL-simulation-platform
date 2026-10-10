"""CartPole balance (continuous force and discrete push variants).

State ``[x, x_dot, theta, theta_dot]``, theta = 0 upright, positive force pushes the cart to +x
(so it decreases theta). Dynamics and Euler integration follow the classic Barto/Sutton cart-pole.
Reward is +1 per step; terminated when the pole falls or the cart leaves the track; truncated at
``max_steps`` (= ``TaskSpec.horizon_steps``).
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from rcl.core.environment import BaseEnvironment
from rcl.core.types import ActionSpaceKind, EnvMetadata, EnvSpec, LinearModel, TaskSpec
from rcl.registry import register_environment

THETA_LIMIT = 12 * math.pi / 180  # pole counts as fallen beyond 12 degrees
X_LIMIT = 2.4
_LABELS = dict(
    state_labels=("x", "x_dot", "theta", "theta_dot"), state_units=("m", "m/s", "rad", "rad/s")
)


class CartPoleParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gravity: float = Field(9.8, gt=0)
    masscart: float = Field(1.0, gt=0)
    masspole: float = Field(0.1, gt=0)
    length: float = Field(0.5, gt=0, description="half pole length (m)")
    force_mag: float = Field(10.0, gt=0, description="max cart force (N)")
    dt: float = Field(0.02, gt=0, le=0.1)
    max_steps: int = Field(500, ge=1, le=100000)


@register_environment("cartpole", params=CartPoleParams)
class CartPole(BaseEnvironment):
    env_metadata: ClassVar[EnvMetadata] = EnvMetadata(
        name="CartPole (continuous)",
        version="0.1.0",
        description="Balance a pole on a cart by applying a continuous horizontal force.",
        dof=2,
        action_space_kind=ActionSpaceKind.CONTINUOUS,
        physics_backend="analytic",
        linearizable=True,
        action_labels=("force",),
        action_units=("N",),
        **_LABELS,
    )
    metadata = {"render_modes": []}  # noqa: RUF012 - gymnasium convention
    _env_id: ClassVar[str] = "cartpole"

    def __init__(
        self,
        gravity: float = 9.8,
        masscart: float = 1.0,
        masspole: float = 0.1,
        length: float = 0.5,
        force_mag: float = 10.0,
        dt: float = 0.02,
        max_steps: int = 500,
    ) -> None:
        super().__init__()
        self.g, self.mc, self.mp, self.l = gravity, masscart, masspole, length
        self.force_mag, self.dt, self.max_steps = force_mag, dt, max_steps
        high = np.array([np.inf] * 4)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float64)
        self._init_action_space()
        self._state = np.zeros(4)
        self._t = 0

    def _init_action_space(self) -> None:
        self.action_space = gym.spaces.Box(-self.force_mag, self.force_mag, (1,), np.float64)

    def _force(self, action: Any) -> float:
        return float(
            np.clip(
                np.asarray(action, dtype=np.float64).reshape(-1)[0], -self.force_mag, self.force_mag
            )
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._state = self.np_random.uniform(-0.05, 0.05, size=4)
        self._t = 0
        return self._state.copy(), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        force = self._force(action)
        self._state = self._next(self._state, force)
        self._t += 1
        terminated = bool(abs(self._state[0]) > X_LIMIT or abs(self._state[2]) > THETA_LIMIT)
        truncated = bool(not terminated and self._t >= self.max_steps)
        info = {"applied_action": np.asarray(self._applied(force), dtype=np.float64)}
        return self._state.copy(), 1.0, terminated, truncated, info

    def _next(self, s: np.ndarray, force: float) -> np.ndarray:
        """One Euler step of the cart-pole dynamics (pure; also the linearization source)."""
        x, xd, th, thd = s
        total, pml = self.mc + self.mp, self.mp * self.l
        sin, cos = math.sin(th), math.cos(th)
        temp = (force + pml * thd**2 * sin) / total
        thacc = (self.g * sin - cos * temp) / (self.l * (4.0 / 3.0 - self.mp * cos**2 / total))
        xacc = temp - pml * thacc * cos / total
        return np.array(
            [x + self.dt * xd, xd + self.dt * xacc, th + self.dt * thd, thd + self.dt * thacc]
        )

    def get_linearization(self) -> LinearModel | None:
        """Jacobians of the step map at the upright equilibrium (central differences)."""
        eps, z0 = 1e-6, np.zeros(4)
        a_cols = [
            (self._next(eps * e, 0.0) - self._next(-eps * e, 0.0)) / (2 * eps) for e in np.eye(4)
        ]
        b = (self._next(z0, eps) - self._next(z0, -eps)) / (2 * eps)
        return LinearModel(np.stack(a_cols, axis=1), b.reshape(4, 1), z0, np.zeros(1), self.dt)

    def _applied(self, force: float) -> Any:
        return [force]

    def get_state(self) -> np.ndarray:
        return self._state.copy()

    def get_env_spec(self) -> EnvSpec:
        return EnvSpec(
            env_id=self._env_id,
            action_space_kind=self.env_metadata.action_space_kind,
            obs_shape=(4,),
            state_dim=4,
            dt=self.dt,
            **self._action_spec(),
        )

    def _action_spec(self) -> dict[str, Any]:
        return {
            "action_shape": (1,),
            "action_low": (-self.force_mag,),
            "action_high": (self.force_mag,),
        }

    def get_task_spec(self) -> TaskSpec:
        # Success: pole within 0.02 rad for 2 s; cart position/velocities unconstrained (1e9).
        return TaskSpec(
            setpoint=(0.0, 0.0, 0.0, 0.0),
            tolerance=(1e9, 1e9, 0.02, 1e9),
            error_state_indices=(2,),
            wrap_indices=(2,),
            output_unit="rad",
            step_floor=0.05,
            horizon_steps=self.max_steps,
            success_hold_steps=100,
            failure_description="pole beyond 12 degrees or cart beyond 2.4 m",
        )


@register_environment("cartpole_discrete", params=CartPoleParams)
class CartPoleDiscrete(CartPole):
    env_metadata: ClassVar[EnvMetadata] = EnvMetadata(
        name="CartPole (discrete)",
        version="0.1.0",
        description="Balance a pole on a cart by pushing left (0) or right (1) with fixed force.",
        dof=2,
        action_space_kind=ActionSpaceKind.DISCRETE,
        physics_backend="analytic",
        action_labels=("push",),
        action_units=("N",),
        **_LABELS,
    )

    _env_id: ClassVar[str] = "cartpole_discrete"

    def _init_action_space(self) -> None:
        self.action_space = gym.spaces.Discrete(2)

    def _force(self, action: Any) -> float:
        return self.force_mag if int(action) == 1 else -self.force_mag

    def get_linearization(self) -> LinearModel | None:
        return None  # discrete push: no linear input matrix

    def _applied(self, force: float) -> Any:
        return force  # scalar -> shape () = EnvSpec.action_shape for discrete

    def _action_spec(self) -> dict[str, Any]:
        return {"n_actions": 2}
