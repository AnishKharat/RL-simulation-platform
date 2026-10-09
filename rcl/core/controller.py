"""BaseController: the only thing the experiment manager knows about a control algorithm."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from rcl.core.environment import BaseEnvironment
from rcl.core.types import (
    ControllerMetadata,
    EnvSpec,
    LinearModel,
    TrainConfig,
    TrainResult,
)

# int for discrete action spaces, float array of shape EnvSpec.action_shape for continuous.
Action = int | np.ndarray


class BaseController(ABC):
    """Contract shared by classical (PID, LQR, MPC) and learned (RL) controllers.

    Contract
    --------
    * Lifecycle: `configure(env_spec, params, linear_model)` is called exactly once before
      `predict`, `train`; `params` is validated against the registry params model.
      `linear_model` is the environment's `get_linearization()` result, passed by the
      experiment manager (controllers never see the environment in `configure`). It is in
      OBSERVATION coordinates (see `LinearModel`), so a model-based controller can act on `obs`
      directly. It is None when the environment offers none; a controller whose metadata has
      `requires_linearization` may raise ValueError on None.
      Configuration must fail loudly (ValueError) if `env_spec.action_space_kind` is not in
      `metadata.supported_action_spaces`.
    * `metadata` is class-level (readable without instantiation); the registry uses it to
      build forms and to filter compatible environments. Never hardcode names elsewhere.
    * `metadata.requires_training` is False for classical controllers; their `train` is a
      no-op returning `TrainResult(timesteps=0)`. Training hyperparameters are part of the
      controller `params`, not of `TrainConfig` (which holds only the budget and seed).
    * Controllers see environments only through
      `BaseEnvironment` and `EnvSpec`, never a concrete robot.
    * `reset()` is called by the rollout at the start of every episode (after `env.reset`, before
      the first `predict`) to clear per-episode internal state (PID integrator and previous
      error, RNN hidden state, MPC warm start). Default: no-op. It must not touch learned
      parameters.
    * `predict(obs, deterministic=True)` is a pure function of the controller's state and
      `obs`; any stochasticity must come from RNGs seeded via `configure`/`train` config,
      never from global RNG state.
    * Evaluation is NOT a controller method: the experiments layer owns one shared
      `rollout(env, controller, n_episodes, seed)` (episode i resets the env with `seed + i`,
      steps `predict` until terminated/truncated, returns one `EpisodeTrajectory` each), so
      every controller is evaluated identically (ADR 0013).
    * `save(path)`/`load(path)` round-trip: a loaded controller predicts identically.
    """

    metadata: ClassVar[ControllerMetadata]

    @abstractmethod
    def configure(
        self,
        env_spec: EnvSpec,
        params: dict[str, Any],
        linear_model: LinearModel | None = None,
    ) -> None:
        """Bind to an environment description and hyperparameters (see class contract)."""

    def reset(self) -> None:  # noqa: B027 - intentional optional hook
        """Clear per-episode state; called once per episode by `rollout`. Default: no-op."""

    @abstractmethod
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Action:
        """Map one observation to one action valid for the configured action space."""

    def train(self, env: BaseEnvironment, config: TrainConfig) -> TrainResult:
        """Learn from `env` within `config` budget. Default (classical): no-op."""
        return TrainResult(timesteps=0)

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist learned/tuned state to `path`."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore state written by `save`."""
