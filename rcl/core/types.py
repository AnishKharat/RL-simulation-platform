"""Shared spec/data types. Imports nothing from the project (see rcl/core/CLAUDE.md).

Declarative specs are frozen Pydantic models (JSON-serialisable, safe for registry/API use).
Array-carrying types are frozen dataclasses that validate their shape invariants on creation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

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

    Compatibility facts (ADR 0014): `action_dim` is the number of action components (1 for a
    single discrete choice); `linearizable` is True iff `get_linearization()` returns a model
    (a test per environment asserts the two agree). The registry compares these with the
    controller's declarations.

    `simulated` is False for hardware-in-the-loop environments (Phase 7 ros_adapter): they have
    no seeded determinism, so the experiment manager refuses multi-seed reproducibility
    claims for them and blocks training on them (see `BaseEnvironment`).

    Labels for plots and tables (all optional, empty = unlabelled): `state_labels` /
    `state_units` name the components of the true state (equal lengths, at least `dof`, since
    the state includes the positions); `action_labels` / `action_units` name the action
    components (length `action_dim`). Example: ("x", "x_dot", "theta", "theta_dot") with
    ("m", "m/s", "rad", "rad/s"). The frontend renders these without robot-specific code;
    an environment test should assert `len(state_labels) == get_env_spec().state_dim`.
    """

    name: str
    version: str = Field(min_length=1)
    description: str = ""
    dof: int = Field(ge=1)
    action_space_kind: ActionSpaceKind
    action_dim: int = Field(default=1, ge=1)
    linearizable: bool = False
    simulated: bool = True
    physics_backend: str
    state_labels: tuple[str, ...] = ()
    state_units: tuple[str, ...] = ()
    action_labels: tuple[str, ...] = ()
    action_units: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_labels(self) -> EnvMetadata:
        if len(self.state_labels) != len(self.state_units):
            raise ValueError("state_labels and state_units must have equal length")
        if self.state_labels and len(self.state_labels) < self.dof:
            raise ValueError("state labels must cover at least dof components")
        if len(self.action_labels) != len(self.action_units):
            raise ValueError("action_labels and action_units must have equal length")
        if self.action_labels and len(self.action_labels) != self.action_dim:
            raise ValueError("action_labels must have action_dim entries")
        return self


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
    in `RunRecord.versions` (key `controller:<id>`).

    Compatibility declarations (ADR 0014), all checked by the registry: the env's action kind
    must be in `supported_action_spaces`; `requires_linearization` needs an env with
    `EnvMetadata.linearizable`; `max_action_dim` (None = unlimited) bounds
    `EnvMetadata.action_dim` (e.g. a scalar PID declares 1)."""

    name: str
    version: str = Field(min_length=1)
    description: str = ""
    family: ControllerFamily
    supported_action_spaces: tuple[ActionSpaceKind, ...] = Field(min_length=1)
    requires_training: bool
    requires_linearization: bool = False
    max_action_dim: int | None = Field(default=None, ge=1)


class PerturbationMetadata(_Spec):
    """Static description of a perturbation (ADR 0021), the basis of API and dashboard forms.

    `stage` is "step" (an in-episode `EnvWrapper`) or "params" (rewrites environment params
    before construction). `magnitude_param` names the scalar parameter that a robustness sweep
    varies, `nominal_magnitude` its no-perturbation value and `magnitude_unit` its unit.
    `action_spaces` lists the environment action kinds the perturbation applies to."""

    name: str
    version: str = Field(min_length=1)
    description: str = ""
    stage: Literal["step", "params"]
    magnitude_param: str = Field(min_length=1)
    magnitude_unit: str = ""
    nominal_magnitude: float
    action_spaces: tuple[ActionSpaceKind, ...] = (
        ActionSpaceKind.CONTINUOUS,
        ActionSpaceKind.DISCRETE,
    )


class TaskSpec(_Spec):
    """What 'success' means for an environment, in true-state coordinates.

    Success: |d_i| <= tolerance[i] for every state component i, for `success_hold_steps`
    consecutive steps, where d_i = state[i] - setpoint[i], wrapped to (-pi, pi] for the indices
    in `wrap_indices` (the same wrap the metrics use). A component that must not constrain
    success gets a large FINITE tolerance (e.g. 1e9): `inf`/`nan` are rejected in `setpoint`
    and `tolerance` so the spec stays JSON-safe; the rollout then ends the episode with
    `terminated` and `truncated` both False (success is not termination).
    `horizon_steps` is the time limit applied to every controller (fair-comparison protocol,
    project_context 6.8). Failure semantics (e.g. pole falls) are
    described in `failure_description` and signalled via `terminated`.

    Metric signal: the output is the vector y = state[error_state_indices] (at least one
    component) with reference `y_ref = setpoint[error_state_indices]` (a derived property, so
    it cannot disagree with `setpoint`). Per-component differences are wrapped to (-pi, pi]
    for the state indices listed in `wrap_indices` (angles) and left as is otherwise; the
    error is the Euclidean norm of the wrapped differences (metrics/DEFINITIONS.md). With one
    index this is the scalar |y - y_ref|. The 2% settling band is
    e <= 0.02 * max(S, step_floor) with S = ||y_ref - y(0)||; `step_floor` is the minimum step
    size used when the commanded step is ~0. `output_unit` optionally names the unit shared by
    all output components (e.g. "rad"); metrics report it as the unit of error-based values.
    The episode time limit is
    T_max = horizon_steps * EnvSpec.dt (there is no separate time-limit field).
    """

    setpoint: tuple[float, ...]
    tolerance: tuple[float, ...]
    error_state_indices: tuple[int, ...] = Field(min_length=1)
    wrap_indices: tuple[int, ...] = ()
    output_unit: str | None = None
    step_floor: float = Field(gt=0)
    horizon_steps: int = Field(gt=0)
    success_hold_steps: int = Field(default=1, gt=0)
    failure_description: str = ""

    @model_validator(mode="after")
    def _check_shapes(self) -> TaskSpec:
        if not all(math.isfinite(v) for v in (*self.setpoint, *self.tolerance)):
            raise ValueError("setpoint and tolerance must be finite (use a large value, not inf)")
        if len(self.tolerance) != len(self.setpoint):
            raise ValueError("tolerance and setpoint must have equal length")
        idx = self.error_state_indices
        if len(set(idx)) != len(idx) or any(not 0 <= i < len(self.setpoint) for i in idx):
            raise ValueError("error_state_indices must be unique and within setpoint range")
        if not set(self.wrap_indices) <= set(idx):
            raise ValueError("wrap_indices must be a subset of error_state_indices")
        return self

    @property
    def y_ref(self) -> tuple[float, ...]:
        """Reference of the output vector: `setpoint[i] for i in error_state_indices`."""
        return tuple(self.setpoint[i] for i in self.error_state_indices)


class TrainConfig(_Spec):
    """Training budget. Classical controllers ignore it. Algorithm hyperparameters (learning
    rate, gamma, network size, ...) are NOT here: they are the controller's own `params`,
    validated by its registry params model. `n_envs` is how many environments the controller
    builds from the `env_factory` for vectorised training."""

    total_timesteps: int = Field(gt=0)
    seed: int = 0
    n_envs: int = Field(default=1, ge=1)


class TrainResult(_Spec):
    """Outcome of `BaseController.train`. Classical controllers return timesteps=0.
    `curve` is the learning curve as `(timestep, mean_episode_return)` pairs, in order."""

    timesteps: int = Field(ge=0)
    wall_time_s: float = Field(default=0.0, ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)
    curve: tuple[tuple[int, float], ...] = ()


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
    `commanded_actions` (optional, shape (T, ...)) is the controller's RAW `predict` output
    before the environment clipped/mapped it, while `actions` are the values actually applied
    (physical units); metrics use `actions`. `perturbation_steps` lists the step indices
    `k` in [0, T) at which a perturbation was injected (empty without perturbations), used by
    recovery-time metrics. `terminated` (task failure only) and `truncated` (time limit) are
    never both True; both are
    False when the episode ended on success (`success_hold_steps` reached). `success` is judged
    by the environment's TaskSpec. `seed` is the reset seed.
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
    commanded_actions: np.ndarray | None = None
    perturbation_steps: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        t = len(self.rewards)
        if self.commanded_actions is not None and len(self.commanded_actions) != t:
            raise ValueError("commanded_actions must have length T")
        if any(not 0 <= k < t for k in self.perturbation_steps):
            raise ValueError("perturbation_steps must be step indices in [0, T)")
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
