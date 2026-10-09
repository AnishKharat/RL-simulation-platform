"""Shared spec/data types. Imports nothing from the project (see rcl/core/CLAUDE.md).

Declarative specs are frozen Pydantic models (JSON-serialisable, safe for registry/API use).
Array-carrying types are frozen dataclasses that validate their shape invariants on creation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ActionSpaceKind(StrEnum):
    """Kind of action space an environment exposes / a controller supports."""

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class ControllerFamily(StrEnum):
    CLASSICAL = "classical"
    RL = "rl"


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EnvMetadata(_Spec):
    """Static, class-level description of an environment (drives registry/frontend forms).

    Contract: available without instantiating the environment. It carries no parameter
    schema: the single source of truth is the Pydantic params model given to
    `@register_environment(..., params=Model)`; the registry derives the JSON schema and the
    defaults from it. `version` is the component's own version string; it is recorded in
    `RunRecord.versions` (key `env:<id>`) so a changed robot model is visible in the record.
    """

    name: str
    version: str = Field(min_length=1)
    description: str = ""
    dof: int = Field(ge=1)
    action_space_kind: ActionSpaceKind
    physics_backend: str


class EnvSpec(_Spec):
    """Runtime description of one environment instance, handed to `BaseController.configure`.

    Contract: shapes/bounds are those of the instance's gymnasium spaces; `dt` is the control
    period in seconds. `n_actions` is set iff `action_space_kind` is DISCRETE;
    `action_low/high` are set iff CONTINUOUS, with equal lengths (= prod(action_shape) when
    `action_shape` is given) and low <= high. Violations raise at construction.
    """

    env_id: str
    action_space_kind: ActionSpaceKind
    obs_shape: tuple[int, ...]
    state_dim: int = Field(ge=1)
    dt: float = Field(gt=0)
    action_shape: tuple[int, ...] = ()
    n_actions: int | None = Field(default=None, ge=2)
    action_low: tuple[float, ...] | None = None
    action_high: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def _check_action_fields(self) -> EnvSpec:
        if self.action_space_kind is ActionSpaceKind.DISCRETE:
            if (
                self.n_actions is None
                or self.action_low is not None
                or self.action_high is not None
            ):
                raise ValueError("DISCRETE requires n_actions and forbids action_low/action_high")
            return self
        if self.n_actions is not None or self.action_low is None or self.action_high is None:
            raise ValueError("CONTINUOUS requires action_low/action_high and forbids n_actions")
        if len(self.action_low) != len(self.action_high):
            raise ValueError("action_low and action_high must have equal length")
        if any(lo > hi for lo, hi in zip(self.action_low, self.action_high, strict=True)):
            raise ValueError("action_low must be <= action_high element-wise")
        if self.action_shape and len(self.action_low) != math.prod(self.action_shape):
            raise ValueError("action_low/high length must equal prod(action_shape)")
        return self


class ControllerMetadata(_Spec):
    """Static description of a controller. `supported_action_spaces` is declared, never
    hardcoded elsewhere: the registry uses it to filter compatible environments.
    `requires_training` is False for classical controllers (their `train` is a no-op); the
    experiment manager reads it from here to decide whether to train. `version` is recorded
    in `RunRecord.versions` (key `controller:<id>`)."""

    name: str
    version: str = Field(min_length=1)
    description: str = ""
    family: ControllerFamily
    supported_action_spaces: tuple[ActionSpaceKind, ...] = Field(min_length=1)
    requires_training: bool


class TaskSpec(_Spec):
    """What 'success' means for an environment, in true-state coordinates.

    Success: |state - setpoint| <= tolerance element-wise for `success_hold_steps` consecutive
    steps before the episode ends. `horizon_steps` is the time limit applied to every
    controller (fair-comparison protocol, project_context 6.8). Failure semantics (e.g. pole
    falls) are described in `failure_description` and signalled via `terminated`.

    Metric signal: the scalar output is y = state[error_state_index] with reference
    `y_ref = setpoint[error_state_index]` (a derived property, so it cannot disagree with
    `setpoint`). The 2% settling band is |y - y_ref| <= 0.02 * max(|y_ref - y(0)|, step_floor);
    `step_floor` is the minimum step size used when the commanded step is ~0. `angle_wrap`
    is True when y is an angle (errors are wrapped to (-pi, pi]). The episode time limit is
    T_max = horizon_steps * EnvSpec.dt (there is no separate time-limit field).
    """

    setpoint: tuple[float, ...]
    tolerance: tuple[float, ...]
    error_state_index: int = Field(ge=0)
    step_floor: float = Field(gt=0)
    horizon_steps: int = Field(gt=0)
    success_hold_steps: int = Field(default=1, gt=0)
    angle_wrap: bool = False
    failure_description: str = ""

    @model_validator(mode="after")
    def _check_shapes(self) -> TaskSpec:
        if len(self.tolerance) != len(self.setpoint):
            raise ValueError("tolerance and setpoint must have equal length")
        if self.error_state_index >= len(self.setpoint):
            raise ValueError("error_state_index out of range for setpoint")
        return self

    @property
    def y_ref(self) -> float:
        """Reference of the scalar output: `setpoint[error_state_index]`."""
        return self.setpoint[self.error_state_index]


class TrainConfig(_Spec):
    """Training budget. Classical controllers ignore it. Algorithm hyperparameters (learning
    rate, gamma, network size, ...) are NOT here: they are the controller's own `params`,
    validated by its registry params model."""

    total_timesteps: int = Field(gt=0)
    seed: int = 0


class TrainResult(_Spec):
    """Outcome of `BaseController.train`. Classical controllers return timesteps=0."""

    timesteps: int = Field(ge=0)
    wall_time_s: float = Field(default=0.0, ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)


@dataclass(frozen=True)
class LinearModel:
    """Linearisation z[k+1] = A (z[k]-z_eq) + B (u[k]-u_eq) + z_eq, sampled at `dt`, where z
    is the OBSERVATION vector (what `predict` receives), not the true state.

    Contract: A is (n, n), B is (n, m), x_eq (the observation at equilibrium; the field name is
    kept for brevity) is (n,), u_eq is (m,), all float arrays, n = prod(EnvSpec.obs_shape).
    An environment whose observation differs from its state (noise, partial) expresses the
    model in observation coordinates, so controllers need not know the state-to-obs map.
    """

    A: np.ndarray
    B: np.ndarray
    x_eq: np.ndarray
    u_eq: np.ndarray
    dt: float

    def __post_init__(self) -> None:
        n, m = self.B.shape
        if self.A.shape != (n, n) or self.x_eq.shape != (n,) or self.u_eq.shape != (m,):
            raise ValueError("inconsistent LinearModel shapes")
        if self.dt <= 0:
            raise ValueError("dt must be > 0")


@dataclass(frozen=True)
class EpisodeTrajectory:
    """One recorded episode; the sole input to all metrics (pure functions over this type).

    With T = number of steps taken:
      observations, states: first axis T+1 (index 0 is the post-reset value);
      actions: first axis T; rewards: shape (T,); dt: control period in seconds.
    Exactly one of `terminated` (task failure/end) or `truncated` (time limit) may be True,
    never both. `success` is judged by the environment's TaskSpec. `seed` is the reset seed.
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    states: np.ndarray
    dt: float
    terminated: bool
    truncated: bool
    seed: int
    success: bool

    def __post_init__(self) -> None:
        t = len(self.rewards)
        if self.rewards.ndim != 1:
            raise ValueError("rewards must be 1-D")
        if len(self.actions) != t:
            raise ValueError("actions must have length T")
        if len(self.observations) != t + 1 or len(self.states) != t + 1:
            raise ValueError("observations and states must have length T+1")
        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if self.terminated and self.truncated:
            raise ValueError("terminated and truncated are mutually exclusive")

    @property
    def n_steps(self) -> int:
        return len(self.rewards)

    @property
    def time(self) -> np.ndarray:
        """Time stamps (T+1,) in seconds matching `states`."""
        return np.arange(self.n_steps + 1) * self.dt
