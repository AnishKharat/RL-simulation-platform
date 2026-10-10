"""2-DOF planar reaching arm (PyBullet), moving in the horizontal plane (no gravity load).

True state ``[q1, q2, q1_dot, q2_dot, ee_err_x, ee_err_y]`` with ``ee_err = ee - target``
(task-space output in the state, convention C1, so a per-episode target needs no TaskSpec
change). Spec:
docs/architecture/phase3-robots.md. Registered only if PyBullet is installed (ADR 0020).
"""

from __future__ import annotations

import importlib.util
import math
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from rcl.core.environment import BaseEnvironment
from rcl.core.types import ActionSpaceKind, EnvMetadata, EnvSpec, TaskSpec
from rcl.physics.pybullet import PyBulletBackend
from rcl.registry import register_environment

L1, L2 = 0.5, 0.4  # link lengths (m)
TIMESTEP, SUBSTEPS = 0.005, 4  # physics 200 Hz, control dt = 0.02 s
REACH = (0.15, 0.85)  # target radius range (m), inside the reachable annulus [0.1, 0.9]

_AVAILABLE = importlib.util.find_spec("pybullet") is not None


class PlanarArmParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_torque: float = Field(2.0, gt=0, description="torque limit per joint (N*m)")
    max_steps: int = Field(200, ge=1, le=100000)


def _register(cls: type) -> type:
    if not _AVAILABLE:  # optional extra missing: stay out of the registry (ADR 0020)
        return cls
    return register_environment("planar_arm", params=PlanarArmParams)(cls)


def _wrap(a: np.ndarray) -> np.ndarray:
    return np.pi - ((np.pi - a) % (2 * np.pi))


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """End-effector (x, y) for joint angles ``q`` (both about +z, measured from +x)."""
    return np.array(
        [
            L1 * math.cos(q[0]) + L2 * math.cos(q[0] + q[1]),
            L1 * math.sin(q[0]) + L2 * math.sin(q[0] + q[1]),
        ]
    )


@_register
class PlanarArm(BaseEnvironment):
    env_metadata: ClassVar[EnvMetadata] = EnvMetadata(
        name="Planar arm (2-DOF)",
        version="0.1.0",
        description="Two-link arm in the horizontal plane reaching a random target (PyBullet).",
        dof=2,
        action_space_kind=ActionSpaceKind.CONTINUOUS,
        action_dim=2,
        physics_backend="pybullet",
        state_labels=("q1", "q2", "q1_dot", "q2_dot", "ee_err_x", "ee_err_y"),
        state_units=("rad", "rad", "rad/s", "rad/s", "m", "m"),
        action_labels=("tau1", "tau2"),
        action_units=("N*m", "N*m"),
    )
    metadata = {"render_modes": ["rgb_array"]}  # noqa: RUF012 - gymnasium convention

    def __init__(self, max_torque: float = 2.0, max_steps: int = 200) -> None:
        super().__init__()
        self.max_torque, self.max_steps = max_torque, max_steps
        self.dt = TIMESTEP * SUBSTEPS
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (6,), np.float64)
        self.action_space = gym.spaces.Box(-max_torque, max_torque, (2,), np.float64)
        self.render_mode = "rgb_array"
        self._sim = PyBulletBackend(self._build, timestep=TIMESTEP)
        self._base = self._sim.save()
        self.target = np.array([L1, 0.0])
        self._t = 0

    @staticmethod
    def _build(p: Any, cid: int) -> tuple[int, list[int]]:
        def bar(length: float) -> int:
            return int(
                p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[length / 2, 0.02, 0.02],
                    collisionFramePosition=[length / 2, 0, 0],
                    physicsClientId=cid,
                )
            )

        body = p.createMultiBody(
            baseMass=0,  # fixed base at the shoulder
            basePosition=[0, 0, 0.5],
            linkMasses=[1.0, 1.0],
            linkCollisionShapeIndices=[bar(L1), bar(L2)],
            linkVisualShapeIndices=[-1, -1],
            linkPositions=[[0, 0, 0], [L1, 0, 0]],
            linkOrientations=[[0, 0, 0, 1]] * 2,
            linkInertialFramePositions=[[L1 / 2, 0, 0], [L2 / 2, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * 2,
            linkParentIndices=[0, 1],
            linkJointTypes=[p.JOINT_REVOLUTE] * 2,
            linkJointAxis=[[0, 0, 1]] * 2,
            physicsClientId=cid,
        )
        return body, [0, 1]

    def _task_state(self) -> np.ndarray:
        s = self._sim.get_state()  # [q1, q2, qd1, qd2]
        q = _wrap(s[:2])
        return np.array([*q, *s[2:], *(forward_kinematics(q) - self.target)])

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        q0 = self.np_random.uniform(-np.pi, np.pi, size=2)
        radius, angle = self.np_random.uniform(*REACH), self.np_random.uniform(-np.pi, np.pi)
        self.target = np.array([radius * math.cos(angle), radius * math.sin(angle)])
        self._sim.restore(self._base)
        self._sim.set_state(np.array([*q0, 0.0, 0.0]))
        self._sim.apply_joint_torques(np.zeros(2))
        self._t = 0
        return self._task_state(), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        u = np.clip(
            np.asarray(action, dtype=np.float64).reshape(2), -self.max_torque, self.max_torque
        )
        self._sim.apply_joint_torques(u)
        self._sim.step(SUBSTEPS)
        self._t += 1
        z = self._task_state()
        info = {"applied_action": u.copy()}
        if not np.all(np.isfinite(z)):
            return z, 0.0, True, False, info
        err = float(np.linalg.norm(z[4:]))
        reward = 1.0 - (err / 0.9) ** 2 - 0.001 * float(u @ u) - 0.0001 * float(z[2:4] @ z[2:4])
        truncated = self._t >= self.max_steps
        return z, reward, False, truncated, info

    def get_state(self) -> np.ndarray:
        return self._task_state()

    def get_env_spec(self) -> EnvSpec:
        return EnvSpec(
            env_id="planar_arm",
            action_space_kind=ActionSpaceKind.CONTINUOUS,
            obs_shape=(6,),
            state_dim=6,
            dt=self.dt,
            action_shape=(2,),
            action_low=(-self.max_torque,) * 2,
            action_high=(self.max_torque,) * 2,
        )

    def get_task_spec(self) -> TaskSpec:
        return TaskSpec(
            setpoint=(0.0,) * 6,
            tolerance=(1e9, 1e9, 1e9, 1e9, 0.02, 0.02),
            error_state_indices=(4, 5),
            output_unit="m",
            step_floor=0.05,
            horizon_steps=self.max_steps,
            success_hold_steps=25,
            failure_description="none in normal operation (a non-finite state terminates)",
        )

    def render(self) -> np.ndarray | None:
        return self._sim.render(320, 240)

    def close(self) -> None:
        self._sim.close()
