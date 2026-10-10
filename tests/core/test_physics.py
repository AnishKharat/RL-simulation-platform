"""The PhysicsBackend stub is a structural contract only (ADR 0019)."""

import numpy as np

from rcl.core.physics import PhysicsBackend


class _Fake:
    name, timestep = "fake", 0.01

    def __init__(self) -> None:
        self._s = np.zeros(2)

    def version(self) -> str:
        return "0"

    def reset(self) -> None:
        self._s = np.zeros(2)

    def step(self, n: int = 1) -> None:
        self._s = self._s + n * self.timestep

    def get_state(self) -> np.ndarray:
        return self._s.copy()

    def set_state(self, state: np.ndarray) -> None:
        self._s = np.array(state, dtype=np.float64)

    def apply_joint_torques(self, torques: np.ndarray) -> None: ...

    def render(self, width: int, height: int) -> np.ndarray | None:
        return None

    def close(self) -> None: ...


def test_a_conforming_class_satisfies_the_protocol_and_others_do_not():
    assert isinstance(_Fake(), PhysicsBackend)
    assert not isinstance(object(), PhysicsBackend)


def test_core_physics_imports_no_simulator():
    import sys

    import rcl.core.physics  # noqa: F401

    assert "pybullet" not in sys.modules and "mujoco" not in sys.modules
