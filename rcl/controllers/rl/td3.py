"""Thin Stable-Baselines3 TD3 wrapper (continuous action spaces only).

SB3/torch are imported lazily inside methods. Determinism claim: CPU only (ADR 0010). The training
exploration noise comes from numpy's global RNG, which SB3 seeds from ``TrainConfig.seed`` (hence
one training job at a time). No observation normalisation (shortcut: add VecNormalize state to
save/load when needed).
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


class TD3Params(BaseModel):
    model_config = ConfigDict(extra="forbid")
    learning_rate: float = Field(3e-4, gt=0)
    gamma: float = Field(0.99, gt=0, le=1)
    buffer_size: int = Field(10000, ge=1)
    batch_size: int = Field(64, ge=2)
    learning_starts: int = Field(32, ge=0)
    train_freq: int = Field(1, ge=1)
    gradient_steps: int = Field(1, ge=1)
    tau: float = Field(0.005, gt=0, le=1)
    policy_delay: int = Field(2, ge=1, description="actor/target update every n critic updates")
    target_policy_noise: float = Field(0.2, ge=0, description="smoothing noise on target actions")
    exploration_noise: float = Field(
        0.1, ge=0, description="std of Gaussian action noise during training (fraction of range)"
    )
    device: Literal["cpu", "cuda"] = "cpu"


@register_controller("td3", params=TD3Params)
class TD3(BaseController):
    metadata: ClassVar[ControllerMetadata] = ControllerMetadata(
        name="TD3",
        version="0.1.0",
        description="Twin Delayed DDPG (Stable-Baselines3, MLP policy, continuous actions).",
        family=ControllerFamily.RL,
        supported_action_spaces=(ActionSpaceKind.CONTINUOUS,),
        requires_training=True,
    )

    def __init__(self) -> None:
        self._p: TD3Params | None = None
        self._model: Any = None

    def configure(
        self,
        env_spec: EnvSpec,
        params: dict[str, Any],
        linear_model: LinearModel | None = None,
    ) -> None:
        if env_spec.action_space_kind not in self.metadata.supported_action_spaces:
            raise ValueError(f"TD3 does not support {env_spec.action_space_kind} actions")
        self._p = TD3Params(**params)

    def train(
        self,
        env_factory: Callable[[], BaseEnvironment],
        config: TrainConfig,
        progress: Callable[[int, int], bool | None] | None = None,
    ) -> TrainResult:
        from stable_baselines3 import TD3 as SB3TD3
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.noise import NormalActionNoise

        if self._p is None:
            raise RuntimeError("configure() must be called before train()")
        p, total = self._p, config.total_timesteps
        curve: list[tuple[int, float]] = []
        every, last = max(1, total // 200), [-total]

        class Hook(BaseCallback):
            stop = False

            def _on_rollout_end(self) -> None:
                # Off-policy algorithms end a "rollout" every train_freq steps: keep the curve to
                # about 200 points however long the budget is.
                buf = self.model.ep_info_buffer
                if buf and self.num_timesteps - last[0] >= every:
                    last[0] = self.num_timesteps
                    curve.append((self.num_timesteps, float(np.mean([e["r"] for e in buf]))))
                if progress is not None and progress(self.num_timesteps, total):
                    self.stop = True

            def _on_step(self) -> bool:
                return not self.stop

        start = time.perf_counter()
        env = make_vec_env(env_factory, n_envs=config.n_envs)  # wraps each env in Monitor
        try:
            shape = env.action_space.shape or (1,)  # Box spaces only (configure checked)
            n_act = int(np.prod(shape))
            self._model = SB3TD3(
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
                policy_delay=p.policy_delay,
                target_policy_noise=p.target_policy_noise,
                action_noise=NormalActionNoise(  # TD3 has no built-in exploration
                    np.zeros(n_act), np.full(n_act, p.exploration_noise)
                ),
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
            raise RuntimeError("TD3 has no policy: call train() or load() first")
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.float64).reshape(self._model.action_space.shape)

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("nothing to save: train() or load() first")
        self._model.save(path / "model.zip")

    def load(self, path: Path) -> None:
        from stable_baselines3 import TD3 as SB3TD3

        device = self._p.device if self._p else "cpu"
        self._model = SB3TD3.load(path / "model.zip", device=device)
