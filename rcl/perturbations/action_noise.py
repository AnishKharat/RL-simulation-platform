"""Gaussian actuator noise added to the clipped command."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from rcl.core.environment import BaseEnvironment
from rcl.core.perturbation import StepPerturbation
from rcl.core.types import ActionSpaceKind, PerturbationMetadata
from rcl.perturbations._action import action_limit, as_command
from rcl.registry import register_perturbation


class ActionNoiseParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    std: float = Field(0.0, ge=0, description="Noise std as a fraction of the actuator limit")


@register_perturbation("action_noise", params=ActionNoiseParams)
class ActionNoise(StepPerturbation):
    perturbation_metadata = PerturbationMetadata(
        name="Action noise",
        version="0.1.0",
        description="Additive Gaussian noise on the command; the environment then clips it.",
        stage="step",
        magnitude_param="std",
        magnitude_unit="fraction of actuator limit",
        nominal_magnitude=0.0,
        action_spaces=(ActionSpaceKind.CONTINUOUS,),
    )

    def __init__(self, env: BaseEnvironment, position: int = 0, std: float = 0.0) -> None:
        super().__init__(env, position)
        self.std = std

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._seed_rng(seed)
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        limit = action_limit(self.action_space)
        noise = self.rng.normal(0.0, self.std, size=limit.shape) * limit
        return self.env.step(as_command(action, self.action_space) + noise)
