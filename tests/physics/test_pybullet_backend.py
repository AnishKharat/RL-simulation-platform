import numpy as np
import pytest

pybullet = pytest.importorskip("pybullet")

from rcl.core.physics import PhysicsBackend  # noqa: E402
from rcl.physics.pybullet import PyBulletBackend  # noqa: E402


def two_link(p, cid):
    """Fixed-base double pendulum hanging under gravity (two revolute joints about x)."""
    link = p.createCollisionShape(p.GEOM_CAPSULE, radius=0.03, height=0.4, physicsClientId=cid)
    body = p.createMultiBody(
        baseMass=0,
        basePosition=[0, 0, 1.5],
        linkMasses=[1.0, 1.0],
        linkCollisionShapeIndices=[link, link],
        linkVisualShapeIndices=[-1, -1],
        linkPositions=[[0, 0, 0], [0, 0, -0.5]],
        linkOrientations=[[0, 0, 0, 1]] * 2,
        linkInertialFramePositions=[[0, 0, -0.25]] * 2,
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * 2,
        linkParentIndices=[0, 1],
        linkJointTypes=[p.JOINT_REVOLUTE] * 2,
        linkJointAxis=[[1, 0, 0]] * 2,
        physicsClientId=cid,
    )
    return body, [0, 1]


@pytest.fixture
def backend():
    b = PyBulletBackend(two_link)
    yield b
    b.close()


def rollout(b, steps=200, torque=(0.5, -0.3)):
    b.reset()
    b.set_state(np.array([0.3, -0.2, 0.0, 0.0]))
    b.apply_joint_torques(np.array(torque))
    out = []
    for _ in range(steps):
        b.step()
        out.append(b.get_state())
    return np.array(out)


def test_satisfies_protocol_and_reports_version(backend):
    assert isinstance(backend, PhysicsBackend)
    assert backend.name == "pybullet" and backend.version().startswith("3.")
    assert backend.timestep == pytest.approx(1 / 240)


def test_state_layout_roundtrip_and_validation(backend):
    s = np.array([0.1, -0.2, 0.3, -0.4])
    backend.set_state(s)
    assert backend.get_state().shape == (4,) and np.allclose(backend.get_state(), s)
    with pytest.raises(ValueError):
        backend.set_state(np.zeros(3))
    with pytest.raises(ValueError):
        backend.apply_joint_torques(np.zeros(3))


def test_stepping_is_deterministic_across_instances_and_torques_matter(backend):
    other = PyBulletBackend(two_link)
    try:
        assert np.array_equal(rollout(backend), rollout(other))
    finally:
        other.close()
    assert not np.array_equal(rollout(backend), rollout(backend, torque=(0.0, 0.0)))
    assert np.array_equal(rollout(backend), rollout(backend))  # reset() restores the world


def test_save_restore_gives_the_exact_continuation(backend):
    rollout(backend, steps=50)
    handle = backend.save()
    first = [backend.step() or backend.get_state() for _ in range(100)]
    backend.restore(handle)
    second = [backend.step() or backend.get_state() for _ in range(100)]
    assert np.array_equal(np.array(first), np.array(second))


def test_gravity_moves_an_unactuated_link(backend):
    backend.set_state(np.array([0.5, 0.0, 0.0, 0.0]))
    backend.step(120)
    assert abs(backend.get_state()[2]) > 0.1  # it swings


def test_render_shape_without_changing_state_and_idempotent_close(backend):
    backend.set_state(np.array([0.2, 0.1, 0.0, 0.0]))
    before = backend.get_state()
    img = backend.render(64, 48)
    assert img.shape == (48, 64, 3) and img.dtype == np.uint8
    assert np.array_equal(before, backend.get_state())
    backend.close()
    backend.close()


def test_floating_base_layout():
    def free_box(p, cid):
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1] * 3, physicsClientId=cid)
        body = p.createMultiBody(
            baseMass=1.0, baseCollisionShapeIndex=shape, basePosition=[0, 0, 1], physicsClientId=cid
        )
        return body, []

    b = PyBulletBackend(free_box, floating_base=True)
    try:
        s = b.get_state()
        assert s.shape == (13,) and s[2] == pytest.approx(1.0)
        b.step(48)
        assert b.get_state()[2] < 1.0  # falls under gravity
    finally:
        b.close()
