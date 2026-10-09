"""BaseEnvironment: the only thing controllers, experiments and metrics know about a robot."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

from rcl.core.types import EnvMetadata, EnvSpec, LinearModel, TaskSpec


class BaseEnvironment(gym.Env[Any, Any], ABC):
    """Gymnasium-compatible environment with benchmarking hooks.

    Contract
    --------
    * Follows the Gymnasium API: `reset(seed=, options=) -> (obs, info)` and
      `step(action) -> (obs, reward, terminated, truncated, info)`, so SB3 works unchanged.
    * Determinism: all randomness comes from `self.np_random`, which is seeded ONLY through
      `reset(seed=...)` (never global RNGs, never a constructor seed). Same seed + same
      actions => identical trajectory. Simulation, not wall-clock, drives time (`dt` fixed).
    * `terminated` = task failure/natural end; `truncated` = the `TaskSpec.horizon_steps`
      time limit. They are never both True.
    * `env_metadata` (class level, `EnvMetadata`) is readable without instantiation and is
      what the registry/frontend use. It is distinct from gymnasium's `metadata` dict
      (render modes/fps), which subclasses still define as usual.
    * `get_state()` returns the TRUE state (not the possibly noisy/partial observation),
      shape `(EnvSpec.state_dim,)`, used by metrics and model-based controllers.
    * `get_linearization()` is an optional capability; the default returns None.
    * Environments import only `rcl.core`; perturbations are implemented as `EnvWrapper`s.
    * `render()`/`close()` follow Gymnasium semantics; `close()` must be idempotent.
    """

    env_metadata: ClassVar[EnvMetadata]

    @abstractmethod
    def get_env_spec(self) -> EnvSpec:
        """Return the runtime `EnvSpec` for this instance (shapes, bounds, dt)."""

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Return the true state, shape `(state_dim,)`, float64, as a copy."""

    @abstractmethod
    def get_task_spec(self) -> TaskSpec:
        """Return the setpoint/tolerance/horizon defining success for this instance."""

    def get_linearization(self) -> LinearModel | None:
        """Return the linearisation about the task equilibrium, or None if unsupported."""
        return None


class EnvWrapper(BaseEnvironment, gym.Wrapper[Any, Any, Any, Any]):
    """Base class for perturbation/noise wrappers around a `BaseEnvironment` (Phase 4).

    Contract
    --------
    * Wraps a `BaseEnvironment` and is itself a `BaseEnvironment`, so controllers, the
      rollout and metrics cannot tell it apart. `reset`/`step`/`render`/`close` follow
      `gymnasium.Wrapper` (delegation); subclasses override `step`/`reset` to perturb.
    * `get_state()` returns the TRUE, UNPERTURBED state of the wrapped environment (sensor
      noise and disturbances act on observations/actions only), so metrics stay comparable
      across perturbation levels.
    * `get_task_spec`, `get_env_spec`, `get_linearization` delegate unchanged.
    * `env_metadata` delegates to the wrapped environment and is available on INSTANCES only
      (a wrapper has no class-level metadata of its own).
    """

    def __init__(self, env: BaseEnvironment) -> None:
        if not isinstance(env, BaseEnvironment):
            raise TypeError("EnvWrapper can only wrap a BaseEnvironment")
        super().__init__(env)
        self._inner = env  # typed handle; gym.Wrapper.env is typed as plain gym.Env

    @property
    def env_metadata(self) -> EnvMetadata:  # type: ignore[override]  # ClassVar -> instance property
        return self._inner.env_metadata

    def get_env_spec(self) -> EnvSpec:
        return self._inner.get_env_spec()

    def get_state(self) -> np.ndarray:
        return self._inner.get_state()

    def get_task_spec(self) -> TaskSpec:
        return self._inner.get_task_spec()

    def get_linearization(self) -> LinearModel | None:
        return self._inner.get_linearization()
