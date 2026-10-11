"""Actuation delay: the plant receives the command issued ``steps`` control steps ago."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from rcl.core.environment import BaseEnvironment
from rcl.core.perturbation import StepPerturbation
from rcl.core.types import ActionSpaceKind, PerturbationMetadata
from rcl.registry import register_perturbation


class ActionDelayParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: int = Field(0, ge=0, le=100, description="Delay in control steps")


@register_perturbation("action_delay", params=ActionDelayParams)
class ActionDelay(StepPerturbation):
    perturbation_metadata = PerturbationMetadata(
        name="Action delay",
        version="0.1.0",
        description="Commands reach the plant `steps` control steps late; zero action until then.",
        stage="step",
        magnitude_param="steps",
        magnitude_unit="control steps",
        nominal_magnitude=0.0,
        action_spaces=(ActionSpaceKind.CONTINUOUS,),  # no neutral discrete action exists
    )

    def __init__(self, env: BaseEnvironment, position: int = 0, steps: int = 0) -> None:
        super().__init__(env, position)
        self.steps = steps
        self._queue: list[Any] = []

    def _zero(self) -> np.ndarray:
        return np.zeros(self.action_space.shape or ())

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._seed_rng(seed)
        self._queue = [self._zero() for _ in range(self.steps)]
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        self._queue.append(action)
        return self.env.step(self._queue.pop(0))
