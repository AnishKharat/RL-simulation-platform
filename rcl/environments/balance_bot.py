"""Two-wheeled self-balancing robot (PyBullet), balancing in the sagittal plane.

Both wheels receive the same torque. True state ``[pitch, pitch_rate, x, x_dot]``; pitch = 0 is
upright and positive pitch leans towards +x; positive torque drives the wheels towards +x (so it
catches a forward lean). Specification: docs/architecture/phase3-robots.md. Registered only if
PyBullet is installed (ADR 0020); the model is built in code (no asset files).
"""

from __future__ import annotations

import importlib.util
import math
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from rcl.core.environment import BaseEnvironment
from rcl.core.types import ActionSpaceKind, EnvMetadata, EnvSpec, LinearModel, TaskSpec
from rcl.physics.pybullet import PyBulletBackend
from rcl.registry import register_environment

PITCH_LIMIT = 0.5  # rad; beyond this the robot counts as fallen
WHEEL_RADIUS, AXLE_HALF_WIDTH = 0.05, 0.12
TIMESTEP, SUBSTEPS = 0.005, 4  # physics 200 Hz, control dt = 0.02 s

_AVAILABLE = importlib.util.find_spec("pybullet") is not None


class BalanceBotParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_torque: float = Field(1.0, gt=0, description="wheel torque limit (N*m, both wheels)")
    chassis_mass: float = Field(1.0, gt=0)
    com_height: float = Field(0.15, gt=0, le=0.5, description="chassis COM above the axle (m)")
    max_steps: int = Field(500, ge=1, le=100000)


def _register(cls: type) -> type:
    if not _AVAILABLE:  # optional extra missing: stay out of the registry (ADR 0020)
        return cls
    return register_environment("balance_bot", params=BalanceBotParams)(cls)


@_register
class BalanceBot(BaseEnvironment):
    env_metadata: ClassVar[EnvMetadata] = EnvMetadata(
        name="Self-balancing robot",
        version="0.1.0",
        description="Two-wheeled inverted pendulum (PyBullet); one symmetric wheel torque.",
        dof=2,
        action_space_kind=ActionSpaceKind.CONTINUOUS,
        linearizable=True,
        physics_backend="pybullet",
        state_labels=("pitch", "pitch_rate", "x", "x_dot"),
        state_units=("rad", "rad/s", "m", "m/s"),
        action_labels=("wheel_torque",),
        action_units=("N*m",),
    )
    metadata = {"render_modes": ["rgb_array"]}  # noqa: RUF012 - gymnasium convention

    def __init__(
        self,
        max_torque: float = 1.0,
        chassis_mass: float = 1.0,
        com_height: float = 0.15,
        max_steps: int = 500,
    ) -> None:
        super().__init__()
        self.max_torque, self.max_steps = max_torque, max_steps
        self._mass, self._com = chassis_mass, com_height
        self.dt = TIMESTEP * SUBSTEPS
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (4,), np.float64)
        self.action_space = gym.spaces.Box(-max_torque, max_torque, (1,), np.float64)
        self.render_mode = "rgb_array"
        self._sim = PyBulletBackend(self._build, timestep=TIMESTEP, floating_base=True)
        self._base = self._sim.save()  # canonical world snapshot (linearisation, C2)
        self._t = 0

    def _build(self, p: Any, cid: int) -> tuple[int, list[int]]:
        plane = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=cid)
        p.createMultiBody(0, plane, physicsClientId=cid)
        box = p.createCollisionShape(  # chassis box centred on the COM, above the axle
            p.GEOM_BOX,
            halfExtents=[0.05, 0.08, self._com],
            collisionFramePosition=[0, 0, self._com],
            physicsClientId=cid,
        )
        wheel = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=WHEEL_RADIUS, height=0.02, physicsClientId=cid
        )
        axis_y = p.getQuaternionFromEuler([math.pi / 2, 0, 0])  # cylinder axis z -> y
        body = p.createMultiBody(
            baseMass=self._mass,
            baseCollisionShapeIndex=box,
            basePosition=[0, 0, WHEEL_RADIUS],
            baseInertialFramePosition=[0, 0, self._com],  # base frame on the axle
            linkMasses=[0.1, 0.1],
            linkCollisionShapeIndices=[wheel, wheel],
            linkVisualShapeIndices=[-1, -1],
            linkPositions=[[0, AXLE_HALF_WIDTH, 0], [0, -AXLE_HALF_WIDTH, 0]],
            linkOrientations=[axis_y, axis_y],
            linkInertialFramePositions=[[0, 0, 0]] * 2,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * 2,
            linkParentIndices=[0, 0],
            linkJointTypes=[p.JOINT_REVOLUTE] * 2,
            linkJointAxis=[[0, 0, 1]] * 2,  # wheel frame z = world y (axle)
            physicsClientId=cid,
        )
        for link in (0, 1):
            p.changeDynamics(body, link, lateralFriction=1.5, physicsClientId=cid)
        p.changeDynamics(body, -1, lateralFriction=0.5, physicsClientId=cid)
        return body, [0, 1]

    # --- state mapping (4-d task state <-> full simulator state) -----------------------------
    # PyBullet reports/resets the base at its centre of mass (COM), which sits ``com`` above the
    # axle; x / x_dot are those of the axle, so convert between the two frames.
    def _full(self, z: np.ndarray) -> np.ndarray:
        pitch, rate, x, xd = (float(v) for v in z)
        q = self._sim.bullet.getQuaternionFromEuler([0.0, pitch, 0.0])
        s, c, h = math.sin(pitch), math.cos(pitch), self._com
        com_pos = [x + h * s, 0.0, WHEEL_RADIUS + h * c]
        com_vel = [xd + h * c * rate, 0.0, -h * s * rate]
        wheel = xd / WHEEL_RADIUS - rate  # rolling, relative to the chassis
        # wheel joints spin about the link's z, which maps to -y for this mounting
        return np.array([*com_pos, *q, *com_vel, 0.0, rate, 0.0, 0.0, 0.0, -wheel, -wheel])

    def _task_state(self) -> np.ndarray:
        s = self._sim.get_state()
        pitch = self._sim.bullet.getEulerFromQuaternion(s[3:7])[1]
        rate, h = s[11], self._com
        x = s[0] - h * math.sin(pitch)
        xd = s[7] - h * math.cos(pitch) * rate
        return np.array([pitch, rate, x, xd])

    def _torque(self, u: float) -> None:
        self._sim.apply_joint_torques(np.array([-u, -u]))  # joint axis is -y: flip sign

    # --- gymnasium API -----------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        z0 = np.array(
            [
                self.np_random.uniform(-0.1, 0.1),
                *self.np_random.uniform(-0.05, 0.05, size=3),
            ]
        )
        self._sim.restore(self._base)
        self._sim.set_state(self._full(z0))
        self._torque(0.0)
        self._t = 0
        return self._task_state(), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        u = float(
            np.clip(
                np.asarray(action, dtype=np.float64).reshape(-1)[0],
                -self.max_torque,
                self.max_torque,
            )
        )
        self._torque(u)
        self._sim.step(SUBSTEPS)
        self._t += 1
        z = self._task_state()
        if not np.all(np.isfinite(z)):
            return z, 0.0, True, False, {"applied_action": np.array([u])}
        cost = (z[0] / PITCH_LIMIT) ** 2 + 0.01 * z[2] ** 2 + 0.001 * u**2
        terminated = bool(abs(z[0]) > PITCH_LIMIT)
        truncated = bool(not terminated and self._t >= self.max_steps)
        return z, 1.0 - cost, terminated, truncated, {"applied_action": np.array([u])}

    def get_state(self) -> np.ndarray:
        return self._task_state()

    def get_env_spec(self) -> EnvSpec:
        return EnvSpec(
            env_id="balance_bot",
            action_space_kind=ActionSpaceKind.CONTINUOUS,
            obs_shape=(4,),
            state_dim=4,
            dt=self.dt,
            action_shape=(1,),
            action_low=(-self.max_torque,),
            action_high=(self.max_torque,),
        )

    def get_task_spec(self) -> TaskSpec:
        return TaskSpec(
            setpoint=(0.0, 0.0, 0.0, 0.0),
            tolerance=(0.02, 1e9, 1e9, 1e9),
            error_state_indices=(0,),
            wrap_indices=(0,),
            output_unit="rad",
            step_floor=0.05,
            horizon_steps=self.max_steps,
            success_hold_steps=100,
            failure_description=f"pitch beyond {PITCH_LIMIT} rad (fallen)",
        )

    def get_linearization(self) -> LinearModel | None:
        """Central differences of the control-step map about upright (C2): every evaluation
        restores the canonical world, writes the perturbed state, applies u and steps."""

        def step_map(z: np.ndarray, u: float) -> np.ndarray:
            self._sim.restore(self._base)
            self._sim.set_state(self._full(z))
            self._torque(u)
            self._sim.step(SUBSTEPS)
            return self._task_state()

        eps, z0, current = 1e-4, np.zeros(4), self._sim.save()
        try:
            cols = [
                (step_map(eps * e, 0.0) - step_map(-eps * e, 0.0)) / (2 * eps) for e in np.eye(4)
            ]
            b = (step_map(z0, eps) - step_map(z0, -eps)) / (2 * eps)
        finally:
            self._sim.restore(current)  # leave no trace in the running episode
        return LinearModel(np.stack(cols, axis=1), b.reshape(4, 1), z0, np.zeros(1), self.dt)

    def render(self) -> np.ndarray | None:
        return self._sim.render(320, 240)

    def close(self) -> None:
        self._sim.close()
