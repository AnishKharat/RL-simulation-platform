from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

pytest.importorskip("pybullet")

from rcl.core.controller import BaseController  # noqa: E402
from rcl.core.types import (  # noqa: E402
    ActionSpaceKind,
    ControllerFamily,
    ControllerMetadata,
    EnvSpec,
)
from rcl.environments.planar_arm import L1, L2, PlanarArm, forward_kinematics  # noqa: E402
from rcl.experiments import rollout  # noqa: E402
from rcl.registry import discover, get_registry  # noqa: E402


class JointPD(BaseController):
    """Test-only controller: PD in joint space towards an IK solution of the target."""

    metadata: ClassVar[ControllerMetadata] = ControllerMetadata(
        name="pd", version="0", family=ControllerFamily.CLASSICAL,
        supported_action_spaces=(ActionSpaceKind.CONTINUOUS,), requires_training=False,
    )  # fmt: skip

    def configure(self, env_spec: EnvSpec, params: dict[str, Any], linear_model: Any = None):
        self.env = params["env"]

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        x, y = self.env.target
        c2 = (x * x + y * y - L1**2 - L2**2) / (2 * L1 * L2)
        q2 = np.arccos(np.clip(c2, -1, 1))
        q1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
        err = (np.array([q1, q2]) - obs[:2] + np.pi) % (2 * np.pi) - np.pi
        return np.clip(20.0 * err - 4.0 * obs[2:4], -2.0, 2.0)

    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...


@pytest.fixture
def env():
    e = PlanarArm()
    yield e
    e.close()


def test_registered_spec_labels_and_not_linearizable(env):
    discover()
    assert get_registry().get("environment", "planar_arm").cls is PlanarArm
    obs, _ = env.reset(seed=0)
    spec, meta = env.get_env_spec(), PlanarArm.env_metadata
    assert obs.shape == spec.obs_shape == (6,) and len(meta.state_labels) == spec.state_dim
    assert spec.action_shape == (2,) and meta.action_dim == 2 and env.get_linearization() is None
    _, _, term, trunc, info = env.step(np.array([0.5, -0.5]))
    assert info["applied_action"].shape == (2,) and not term and not trunc


def test_registry_hides_scalar_and_linearisation_controllers():
    discover()
    ids = [c.id for c in get_registry().compatible_controllers("planar_arm")]
    assert "ppo" in ids and "sac" in ids and "pid" not in ids and "lqr" not in ids


def test_end_effector_error_and_seeded_targets(env):
    obs, _ = env.reset(seed=4)
    assert np.allclose(obs[4:], forward_kinematics(obs[:2]) - env.target)
    assert 0.15 <= np.linalg.norm(env.target) <= 0.85
    t4 = env.target.copy()
    env.reset(seed=4)
    assert np.array_equal(env.target, t4)
    env.reset(seed=5)
    assert not np.array_equal(env.target, t4)


def test_seeded_determinism_and_torque_clipping(env):
    def run(seed):
        env.reset(seed=seed)
        return np.array([env.step(np.array([5.0, -0.3]))[0] for _ in range(20)])

    assert np.array_equal(run(1), run(1))
    env.reset(seed=1)
    assert np.array_equal(env.step(np.array([5.0, -9.0]))[4]["applied_action"], [2.0, -2.0])


def test_no_gravity_load_and_horizon_truncation():
    e = PlanarArm(max_steps=5)
    try:
        obs0, _ = e.reset(seed=2)
        for _ in range(5):
            obs, _, term, trunc, _ = e.step(np.zeros(2))
        assert np.allclose(obs[:2], obs0[:2], atol=1e-6)  # horizontal plane: nothing moves
        assert trunc and not term
    finally:
        e.close()


def test_success_is_reached_by_a_reaching_controller(env):
    c = JointPD()
    c.configure(env.get_env_spec(), {"env": env})
    trajs = rollout(env, c, 3, seed=0)
    assert all(t.success for t in trajs)
    assert all(np.linalg.norm(t.states[-1, 4:]) < 0.02 for t in trajs)
