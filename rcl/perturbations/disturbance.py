"""A force/torque pulse on the actuated input channel (ADR 0021: no apply_disturbance hook)."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rcl.core.environment import BaseEnvironment
from rcl.core.perturbation import StepPerturbation
from rcl.core.types import ActionSpaceKind, PerturbationMetadata
from rcl.perturbations._action import action_limit, as_command
from rcl.registry import register_perturbation


class DisturbanceParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    magnitude: float = Field(
        0.0, ge=0, description="Pulse size as a fraction of the actuator limit"
    )
    duration_steps: int = Field(5, ge=1, description="Pulse length in control steps")
    start_min: int = Field(20, ge=0, description="Earliest pulse start step")
    start_max: int = Field(100, ge=0, description="Latest pulse start step")

    @model_validator(mode="after")
    def _window(self) -> DisturbanceParams:
        if self.start_max < self.start_min:
            raise ValueError("start_max must be >= start_min")
        return self


@register_perturbation("disturbance", params=DisturbanceParams)
class Disturbance(StepPerturbation):
    """One pulse per episode: start step and sign are drawn at reset (two draws, always).

    The pulse is added to the clipped command and the environment clips the sum again, so a
    pulse cannot exceed the actuator limit: against a saturated command in the same direction
    it has no effect. Adding it after the environment's own clipping would need an environment
    hook (rejected in ADR 0021). A pulse of magnitude 0 injects nothing.
    """

    perturbation_metadata = PerturbationMetadata(
        name="Disturbance pulse",
        version="0.1.0",
        description="A force/torque pulse through the actuated input, saturating at the limit.",
        stage="step",
        magnitude_param="magnitude",
        magnitude_unit="fraction of actuator limit",
        nominal_magnitude=0.0,
        action_spaces=(ActionSpaceKind.CONTINUOUS,),
    )

    def __init__(
        self,
        env: BaseEnvironment,
        position: int = 0,
        magnitude: float = 0.0,
        duration_steps: int = 5,
        start_min: int = 20,
        start_max: int = 100,
    ) -> None:
        super().__init__(env, position)
        self.magnitude, self.duration = magnitude, duration_steps
        self.start_min, self.start_max = start_min, start_max
        self._start, self._sign, self._k = 0, 1.0, 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._seed_rng(seed)
        self._start = int(self.rng.integers(self.start_min, self.start_max + 1))
        self._sign = 1.0 if self.rng.integers(0, 2) else -1.0
        self._k = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        command = as_command(action, self.action_space)
        active = self.magnitude > 0 and self._start <= self._k < self._start + self.duration
        self._k += 1
        if active:
            command = command + self._sign * self.magnitude * action_limit(self.action_space)
        obs, reward, terminated, truncated, info = self.env.step(command)
        if active:
            info = {**info, "perturbed": True}
        return obs, reward, terminated, truncated, info
