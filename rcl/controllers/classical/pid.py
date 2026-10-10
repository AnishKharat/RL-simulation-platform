"""Scalar PID controller with integral anti-windup and derivative-on-measurement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from rcl.core.controller import Action, BaseController
from rcl.core.types import (
    ActionSpaceKind,
    ControllerFamily,
    ControllerMetadata,
    EnvSpec,
    LinearModel,
)
from rcl.registry import register_controller


class PIDParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kp: float = Field(60.0, description="proportional gain")
    ki: float = Field(0.0, description="integral gain")
    kd: float = Field(5.0, description="derivative gain")
    setpoint: float = 0.0
    obs_index: int = Field(2, ge=0, description="index of the observation component to regulate")
    integral_limit: float = Field(10.0, gt=0, description="anti-windup clamp on the integral term")


@register_controller("pid", params=PIDParams)
class PID(BaseController):
    """u = kp*e + ki*integral(e) + kd*de/dt with e = obs[obs_index] - setpoint.

    Positive gains push the action up when the measurement is above the setpoint (the sign
    that stabilises a pole by chasing its lean); use negative gains for the opposite plant sign.
    The output is clipped to the action bounds from the EnvSpec.
    """

    metadata: ClassVar[ControllerMetadata] = ControllerMetadata(
        name="PID",
        version="0.1.0",
        description="Single-loop PID on one observation component (continuous, scalar action).",
        family=ControllerFamily.CLASSICAL,
        supported_action_spaces=(ActionSpaceKind.CONTINUOUS,),
        requires_training=False,
        max_action_dim=1,
    )

    def __init__(self) -> None:
        self._integral = 0.0
        self._prev: float | None = None

    def configure(
        self,
        env_spec: EnvSpec,
        params: dict[str, Any],
        linear_model: LinearModel | None = None,
    ) -> None:
        if env_spec.action_space_kind not in self.metadata.supported_action_spaces:
            raise ValueError("PID supports continuous action spaces only")
        if (
            env_spec.action_low is None
            or env_spec.action_high is None
            or len(env_spec.action_low) != 1
        ):
            raise ValueError("PID needs a scalar action")
        p = PIDParams(**params)
        if p.obs_index >= int(np.prod(env_spec.obs_shape)):
            raise ValueError(
                f"obs_index {p.obs_index} outside observation of shape {env_spec.obs_shape}"
            )
        self._p, self._dt = p, env_spec.dt
        self._lo, self._hi = env_spec.action_low[0], env_spec.action_high[0]
        self._shape = env_spec.action_shape
        self.reset()

    def reset(self) -> None:
        self._integral, self._prev = 0.0, None

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Action:
        p = self._p
        e = float(np.asarray(obs).reshape(-1)[p.obs_index]) - p.setpoint
        self._integral = float(
            np.clip(self._integral + e * self._dt, -p.integral_limit, p.integral_limit)
        )
        deriv = 0.0 if self._prev is None else (e - self._prev) / self._dt
        self._prev = e
        u = float(np.clip(p.kp * e + p.ki * self._integral + p.kd * deriv, self._lo, self._hi))
        return np.full(self._shape, u)

    def save(self, path: Path) -> None:
        state = {"params": self._p.model_dump(), "dt": self._dt, "lo": self._lo, "hi": self._hi}
        (path / "pid.json").write_text(json.dumps({**state, "shape": list(self._shape)}))

    def load(self, path: Path) -> None:
        d = json.loads((path / "pid.json").read_text())
        self._p, self._dt, self._lo, self._hi = PIDParams(**d["params"]), d["dt"], d["lo"], d["hi"]
        self._shape = tuple(d["shape"])
        self.reset()
