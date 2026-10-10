"""Thin Stable-Baselines3 SAC wrapper (continuous action spaces only).

SB3/torch are imported lazily inside methods. Determinism claim: CPU only (ADR 0010). Phase 1
does not normalise observations (shortcut: add VecNormalize state to save/load when needed).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from rcl.core.controller import Action, BaseController
from rcl.core.environment import BaseEnvironment
from rcl.core.types import (
    ActionSpaceKind,
    ControllerFamily,
    ControllerMetadata,
    EnvSpec,
    LinearModel,
    TrainConfig,
    TrainResult,
)
from rcl.registry import register_controller


class SACParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    learning_rate: float = Field(3e-4, gt=0)
    gamma: float = Field(0.99, gt=0, le=1)
    buffer_size: int = Field(10000, ge=1)
    batch_size: int = Field(64, ge=2)
    learning_starts: int = Field(32, ge=0)
    train_freq: int = Field(1, ge=1)
    gradient_steps: int = Field(1, ge=1)
    tau: float = Field(0.005, gt=0, le=1)
    device: Literal["cpu", "cuda"] = "cpu"


@register_controller("sac", params=SACParams)
class SAC(BaseController):
    metadata: ClassVar[ControllerMetadata] = ControllerMetadata(
        name="SAC",
        version="0.1.0",
        description="Soft Actor-Critic (Stable-Baselines3, MLP policy, continuous actions).",
        family=ControllerFamily.RL,
        supported_action_spaces=(ActionSpaceKind.CONTINUOUS,),
        requires_training=True,
    )

    def __init__(self) -> None:
        self._p: SACParams | None = None
        self._model: Any = None

    def configure(
        self,
        env_spec: EnvSpec,
        params: dict[str, Any],
        linear_model: LinearModel | None = None,
    ) -> None:
        if env_spec.action_space_kind not in self.metadata.supported_action_spaces:
            raise ValueError(f"SAC does not support {env_spec.action_space_kind} actions")
        self._p = SACParams(**params)

    def train(
        self,
        env_factory: Callable[[], BaseEnvironment],
        config: TrainConfig,
        progress: Callable[[int, int], bool | None] | None = None,
    ) -> TrainResult:
        from stable_baselines3 import SAC as SB3SAC
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.env_util import make_vec_env

        if self._p is None:
            raise RuntimeError("configure() must be called before train()")
        p, total = self._p, config.total_timesteps
        curve: list[tuple[int, float]] = []

        class Hook(BaseCallback):
            stop = False

            def _on_rollout_end(self) -> None:
                buf = self.model.ep_info_buffer
                if buf:
                    curve.append((self.num_timesteps, float(np.mean([e["r"] for e in buf]))))
                if progress is not None and progress(self.num_timesteps, total):
                    self.stop = True

            def _on_step(self) -> bool:
                return not self.stop

        start = time.perf_counter()
        env = make_vec_env(env_factory, n_envs=config.n_envs)  # wraps each env in Monitor
        try:
            self._model = SB3SAC(
                "MlpPolicy",
                env,
                learning_rate=p.learning_rate,
                gamma=p.gamma,
                buffer_size=p.buffer_size,
                batch_size=p.batch_size,
                learning_starts=p.learning_starts,
                train_freq=p.train_freq,
                gradient_steps=p.gradient_steps,
                tau=p.tau,
                seed=config.seed,
                device=p.device,
            )
            hook = Hook()
            self._model.learn(total_timesteps=total, callback=hook)
            done = int(self._model.num_timesteps)
        finally:
            env.close()
        if progress is not None:
            progress(done, total)
        metrics = {"final_mean_return": curve[-1][1]} if curve else {}
        return TrainResult(
            timesteps=done,
            wall_time_s=time.perf_counter() - start,
            metrics=metrics,
            curve=tuple(curve),
        )

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Action:
        if self._model is None:
            raise RuntimeError("SAC has no policy: call train() or load() first")
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.float64).reshape(self._model.action_space.shape)

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("nothing to save: train() or load() first")
        self._model.save(path / "model.zip")

    def load(self, path: Path) -> None:
        from stable_baselines3 import SAC as SB3SAC

        device = self._p.device if self._p else "cpu"
        self._model = SB3SAC.load(path / "model.zip", device=device)
