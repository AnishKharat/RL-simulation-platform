"""Discrete-time LQR on the environment's observation-space linear model."""

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


class LQRParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    q_diag: list[float] | None = Field(
        None, description="state cost weights (one per observation component); default all 1"
    )
    r: float = Field(1.0, gt=0, description="control cost weight (identity scaled)")


def solve_dlqr(a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Gain ``K`` (u = -K z) from the discrete Riccati equation, solved by fixed-point iteration.

    shortcut: plain iteration (no scipy), fine for the small systems here; use a QZ/Schur
    solver if the state dimension grows or convergence is slow.
    """
    p = q.copy()
    for _ in range(100_000):
        gain = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
        nxt = q + a.T @ p @ (a - b @ gain)
        if np.max(np.abs(nxt - p)) < 1e-10 * max(1.0, np.max(np.abs(nxt))):
            p = nxt
            break
        p = nxt
    else:
        raise ValueError("LQR Riccati iteration did not converge (is (A, B) stabilizable?)")
    return np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)


@register_controller("lqr", params=LQRParams)
class LQR(BaseController):
    """u = u_eq - K (obs - x_eq), clipped to the action bounds. Needs ``get_linearization``."""

    metadata: ClassVar[ControllerMetadata] = ControllerMetadata(
        name="LQR",
        version="0.1.0",
        description="Linear-quadratic regulator on the environment's linearised model.",
        family=ControllerFamily.CLASSICAL,
        supported_action_spaces=(ActionSpaceKind.CONTINUOUS,),
        requires_training=False,
        requires_linearization=True,
    )

    def configure(
        self,
        env_spec: EnvSpec,
        params: dict[str, Any],
        linear_model: LinearModel | None = None,
    ) -> None:
        if env_spec.action_space_kind not in self.metadata.supported_action_spaces:
            raise ValueError("LQR supports continuous action spaces only")
        if linear_model is None:
            raise ValueError("LQR needs a linearisation; this environment offers none")
        p = LQRParams(**params)
        n, m = linear_model.B.shape
        if p.q_diag is not None and len(p.q_diag) != n:
            raise ValueError(f"q_diag needs {n} entries (one per observation component)")
        if p.q_diag is not None and min(p.q_diag) < 0:
            raise ValueError("q_diag entries must be >= 0")
        q = np.diag(p.q_diag if p.q_diag is not None else np.ones(n))
        self._gain = solve_dlqr(linear_model.A, linear_model.B, q, p.r * np.eye(m))
        self._x_eq, self._u_eq = linear_model.x_eq, linear_model.u_eq
        self._lo = np.asarray(env_spec.action_low)
        self._hi = np.asarray(env_spec.action_high)
        self._shape = env_spec.action_shape

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Action:
        z = np.asarray(obs, dtype=np.float64).reshape(-1) - self._x_eq
        u = np.clip(self._u_eq - self._gain @ z, self._lo, self._hi)
        return np.asarray(u.reshape(self._shape))

    def save(self, path: Path) -> None:
        state = {
            "gain": self._gain.tolist(),
            "x_eq": self._x_eq.tolist(),
            "u_eq": self._u_eq.tolist(),
            "lo": self._lo.tolist(),
            "hi": self._hi.tolist(),
            "shape": list(self._shape),
        }
        (path / "lqr.json").write_text(json.dumps(state))

    def load(self, path: Path) -> None:
        d = json.loads((path / "lqr.json").read_text())
        self._gain = np.asarray(d["gain"])
        self._x_eq, self._u_eq = np.asarray(d["x_eq"]), np.asarray(d["u_eq"])
        self._lo, self._hi = np.asarray(d["lo"]), np.asarray(d["hi"])
        self._shape = tuple(d["shape"])
