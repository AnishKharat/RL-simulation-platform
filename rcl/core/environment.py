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
    * Environments import only `rcl.core`; perturbations are implemented as wrappers.
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
