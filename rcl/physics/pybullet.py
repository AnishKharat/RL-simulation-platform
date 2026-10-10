"""PyBullet implementation of ``rcl.core.physics.PhysicsBackend`` (ADR 0019).

Headless (``DIRECT``), fixed timestep, one private client per instance. PyBullet is imported
lazily so this module imports fine without the optional ``physics`` extra (ADR 0020).
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

# The environment's world builder: (pybullet module, client id) -> (robot body id, actuated joints)
WorldBuilder = Callable[[Any, int], tuple[int, list[int]]]


class PyBulletBackend:
    """A PyBullet world for one environment instance.

    ``build`` creates the bodies (planes, robot) through the given pybullet module and client id
    and returns the robot body id and its actuated (torque-controlled) joint ids, in action order.
    State layout from ``get_state``: ``[q, q_dot]`` over the actuated joints, preceded by
    ``[base_pos(3), base_quat_xyzw(4), base_lin_vel(3), base_ang_vel(3)]`` if ``floating_base``.
    Engine settings are applied identically on every ``reset`` (they change trajectories; spike).
    """

    name = "pybullet"

    def __init__(
        self,
        build: WorldBuilder,
        *,
        timestep: float = 1 / 240,
        gravity: Sequence[float] = (0.0, 0.0, -9.81),
        floating_base: bool = False,
    ) -> None:
        import pybullet

        self.bullet: Any = pybullet
        self.timestep = timestep
        self.floating_base = floating_base
        self._build, self._gravity = build, tuple(gravity)
        self.client_id: int = pybullet.connect(pybullet.DIRECT)
        self._closed = False
        self.reset()

    # --- world -------------------------------------------------------------------------------
    def version(self) -> str:
        return importlib.metadata.version("pybullet")

    def reset(self) -> None:
        p, cid = self.bullet, self.client_id
        p.resetSimulation(physicsClientId=cid)
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1, physicsClientId=cid)
        p.setTimeStep(self.timestep, physicsClientId=cid)
        p.setGravity(*self._gravity, physicsClientId=cid)
        self.body, self.joints = self._build(p, cid)
        # default velocity motors would fight the torques: switch them off
        if self.joints:
            p.setJointMotorControlArray(
                self.body,
                self.joints,
                p.VELOCITY_CONTROL,
                forces=[0.0] * len(self.joints),
                physicsClientId=cid,
            )
        self._torques = np.zeros(len(self.joints))

    def step(self, n: int = 1) -> None:
        p = self.bullet
        for _ in range(n):  # re-apply each step: torque commands are consumed by stepSimulation
            if not self.joints:
                p.stepSimulation(physicsClientId=self.client_id)
                continue
            p.setJointMotorControlArray(
                self.body,
                self.joints,
                p.TORQUE_CONTROL,
                forces=self._torques.tolist(),
                physicsClientId=self.client_id,
            )
            p.stepSimulation(physicsClientId=self.client_id)

    # --- state -------------------------------------------------------------------------------
    def get_state(self) -> np.ndarray:
        p, cid = self.bullet, self.client_id
        js = p.getJointStates(self.body, self.joints, physicsClientId=cid) if self.joints else []
        joint = [s[0] for s in js] + [s[1] for s in js]
        if not self.floating_base:
            return np.array(joint, dtype=np.float64)
        pos, orn = p.getBasePositionAndOrientation(self.body, physicsClientId=cid)
        lin, ang = p.getBaseVelocity(self.body, physicsClientId=cid)
        return np.array([*pos, *orn, *lin, *ang, *joint], dtype=np.float64)

    def set_state(self, state: np.ndarray) -> None:
        p, cid, s = self.bullet, self.client_id, np.asarray(state, dtype=np.float64)
        expected = 2 * len(self.joints) + (13 if self.floating_base else 0)
        if s.shape != (expected,):
            raise ValueError(f"state must have shape ({expected},), got {s.shape}")
        if self.floating_base:
            p.resetBasePositionAndOrientation(self.body, s[0:3], s[3:7], physicsClientId=cid)
            p.resetBaseVelocity(self.body, s[7:10], s[10:13], physicsClientId=cid)
            s = s[13:]
        n = len(self.joints)
        for k, j in enumerate(self.joints):
            p.resetJointState(self.body, j, s[k], s[n + k], physicsClientId=cid)

    def save(self) -> int:
        return int(self.bullet.saveState(physicsClientId=self.client_id))

    def restore(self, handle: int) -> None:
        self.bullet.restoreState(stateId=handle, physicsClientId=self.client_id)

    # --- actuation / output ------------------------------------------------------------------
    def apply_joint_torques(self, torques: np.ndarray) -> None:
        t = np.asarray(torques, dtype=np.float64).reshape(-1)
        if t.shape != (len(self.joints),):
            raise ValueError(f"need {len(self.joints)} torques, got {t.shape[0]}")
        self._torques = t.copy()

    def render(self, width: int, height: int) -> np.ndarray | None:
        p = self.bullet
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0.3], distance=2.0, yaw=45, pitch=-30, roll=0, upAxisIndex=2
        )
        proj = p.computeProjectionMatrixFOV(fov=60, aspect=width / height, nearVal=0.05, farVal=20)
        _, _, rgba, _, _ = p.getCameraImage(
            width,
            height,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client_id,
        )
        return np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3].copy()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.bullet.disconnect(physicsClientId=self.client_id)
