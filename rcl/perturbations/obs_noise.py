"""Gaussian sensor noise on the observation the controller sees (not on the true state)."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from rcl.core.environment import BaseEnvironment
from rcl.core.perturbation import StepPerturbation
from rcl.core.types import PerturbationMetadata
from rcl.registry import register_perturbation


class ObsNoiseParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    std: float = Field(
        0.0, ge=0, description="Std of the noise added to each observation component"
    )


@register_perturbation("obs_noise", params=ObsNoiseParams)
class ObsNoise(StepPerturbation):
    perturbation_metadata = PerturbationMetadata(
        name="Observation noise",
        version="0.1.0",
        description="Additive Gaussian noise on each observation component (observation units).",
        stage="step",
        magnitude_param="std",
        magnitude_unit="obs units",
        nominal_magnitude=0.0,
    )

    def __init__(self, env: BaseEnvironment, position: int = 0, std: float = 0.0) -> None:
        super().__init__(env, position)
        self.std = std

    def _noisy(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        return obs + self.rng.normal(0.0, self.std, size=obs.shape)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._seed_rng(seed)
        obs, info = self.env.reset(seed=seed, options=options)
        return self._noisy(obs), info

    def step(self, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._noisy(obs), reward, terminated, truncated, info
