"""THROWAWAY SPIKE (architecture review 4, ADR 0019): is PyBullet deterministic enough for
ADR 0010's reproducibility claim, and does a manual set_state restore an exact state?

Not part of the package; never imported by rcl.
Run: python spikes/pybullet_determinism/determinism.py
Scenario: r2d2 (4 driven wheels + contacts) on a plane, headless DIRECT mode, fixed 1/240 s step,
random initial joint state and random wheel torques from a seeded numpy RNG.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys

import numpy as np
import pybullet as p
import pybullet_data

STEPS = 600


def _world(cid: int, flag: bool) -> tuple[int, list[int]]:
    p.resetSimulation(physicsClientId=cid)
    if flag:
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1, physicsClientId=cid)
    p.setTimeStep(1 / 240, physicsClientId=cid)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    p.loadURDF("plane.urdf", physicsClientId=cid)
    robot = p.loadURDF("r2d2.urdf", [0, 0, 0.5], physicsClientId=cid)
    joints = [
        j
        for j in range(p.getNumJoints(robot, physicsClientId=cid))
        if p.getJointInfo(robot, j, physicsClientId=cid)[2] == p.JOINT_REVOLUTE
    ]
    # disable default velocity motors so torque control is exact
    p.setJointMotorControlArray(
        robot, joints, p.VELOCITY_CONTROL, forces=[0] * len(joints), physicsClientId=cid
    )
    return robot, joints


def _state(cid: int, robot: int, joints: list[int]) -> np.ndarray:
    pos, orn = p.getBasePositionAndOrientation(robot, physicsClientId=cid)
    lin, ang = p.getBaseVelocity(robot, physicsClientId=cid)
    js = p.getJointStates(robot, joints, physicsClientId=cid)
    return np.array([*pos, *orn, *lin, *ang, *[s[0] for s in js], *[s[1] for s in js]])


def _set_state(cid: int, robot: int, joints: list[int], s: np.ndarray) -> None:
    n = len(joints)
    p.resetBasePositionAndOrientation(robot, s[0:3], s[3:7], physicsClientId=cid)
    p.resetBaseVelocity(robot, s[7:10], s[10:13], physicsClientId=cid)
    for k, j in enumerate(joints):
        p.resetJointState(robot, j, s[13 + k], s[13 + n + k], physicsClientId=cid)


def run(seed: int, cid: int | None = None, flag: bool = False) -> np.ndarray:
    """State sequence (STEPS+1, d). A new DIRECT client unless ``cid`` is given (reuse)."""
    own = cid is None
    cid = p.connect(p.DIRECT) if own else cid
    robot, joints = _world(cid, flag)
    rng = np.random.default_rng(seed)
    for j in joints:
        p.resetJointState(robot, j, rng.uniform(-0.2, 0.2), 0.0, physicsClientId=cid)
    out = [_state(cid, robot, joints)]
    for _ in range(STEPS):
        tau = rng.uniform(-5, 5, len(joints))
        p.setJointMotorControlArray(
            robot, joints, p.TORQUE_CONTROL, forces=tau, physicsClientId=cid
        )
        p.stepSimulation(physicsClientId=cid)
        out.append(_state(cid, robot, joints))
    if own:
        p.disconnect(cid)
    return np.array(out)


def restore_check(manual: bool, after_restore: bool = False) -> float:
    """Max |diff| between continuing from step 300 and re-running the tail after restoring the
    step-300 state, either manually (PhysicsBackend.set_state style) or with saveState."""
    cid = p.connect(p.DIRECT)
    robot, joints = _world(cid, False)
    rng = np.random.default_rng(7)
    taus = rng.uniform(-5, 5, (STEPS, len(joints)))
    for t in range(300):
        p.setJointMotorControlArray(
            robot, joints, p.TORQUE_CONTROL, forces=taus[t], physicsClientId=cid
        )
        p.stepSimulation(physicsClientId=cid)
    snap, sid = _state(cid, robot, joints), p.saveState(physicsClientId=cid)
    first = []
    for t in range(300, STEPS):
        p.setJointMotorControlArray(
            robot, joints, p.TORQUE_CONTROL, forces=taus[t], physicsClientId=cid
        )
        p.stepSimulation(physicsClientId=cid)
        first.append(_state(cid, robot, joints))
    if after_restore:  # exact snapshot first, then a manual set of the SAME values
        p.restoreState(stateId=sid, physicsClientId=cid)
        _set_state(cid, robot, joints, snap)
    elif manual:
        _set_state(cid, robot, joints, snap)
    else:
        p.restoreState(stateId=sid, physicsClientId=cid)
    second = []
    for t in range(300, STEPS):
        p.setJointMotorControlArray(
            robot, joints, p.TORQUE_CONTROL, forces=taus[t], physicsClientId=cid
        )
        p.stepSimulation(physicsClientId=cid)
        second.append(_state(cid, robot, joints))
    p.disconnect(cid)
    return float(np.max(np.abs(np.array(first) - np.array(second))))


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


if __name__ == "__main__":
    if sys.argv[1:] == ["--digest"]:  # child process for the cross-process check
        print(digest(run(0)))
        sys.exit(0)
    a, b = run(0), run(0)
    cid = p.connect(p.DIRECT)
    r1, r2 = run(0, cid), run(0, cid)  # one client, resetSimulation between runs
    p.disconnect(cid)
    child = subprocess.run([sys.executable, __file__, "--digest"], capture_output=True, text=True)
    results = {
        "fresh client x2 (same seed) identical": np.array_equal(a, b),
        "reused client + resetSimulation identical": np.array_equal(r1, r2)
        and np.array_equal(r1, a),
        "deterministicOverlappingPairs=1 identical": np.array_equal(
            run(0, flag=True), run(0, flag=True)
        ),
        "flag changes the trajectory": not np.array_equal(a, run(0, flag=True)),
        "separate process identical": child.stdout.strip() == digest(a),
        "different seed differs": not np.array_equal(a, run(1)),
        "max |diff| after manual set_state restore": restore_check(manual=True),
        "max |diff| after saveState/restoreState": restore_check(manual=False),
        "max |diff| restoreState then manual set of same values": restore_check(True, True),
        "manual restore is itself repeatable": restore_check(True) == restore_check(True),
    }
    print(f"pybullet API {p.getAPIVersion()} | numpy {np.__version__} | {STEPS} steps")
    for k, v in results.items():
        print(f"{k}: {v}")
