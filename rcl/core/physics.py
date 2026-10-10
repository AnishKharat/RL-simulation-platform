"""PhysicsBackend: the contract a rigid-body simulator offers to the environments that use it.

STUB (architecture review 4, ADR 0019): interface only, no implementation and no caller yet.
Adding this module changes no existing interface or behaviour.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PhysicsBackend(Protocol):
    """A simulated world for ONE environment instance (PyBullet now, MuJoCo later).

    Contract
    --------
    * Ownership: the environment creates its backend (with its own model files and
      parameters), drives it, and closes it in ``close()``. Controllers, the experiment
      manager and metrics never see a backend; they only see ``BaseEnvironment``.
    * Time: ``timestep`` is the fixed physics step in seconds. The environment's control period
      (``EnvSpec.dt``) is an integer number of physics steps, chosen by the environment.
      No real-time stepping, ever.
    * No randomness: a backend has no RNG. Environments sample initial conditions from their
      own ``np_random`` (seeded only by ``reset(seed=...)``, ADR 0010) and apply them with
      ``set_state``. Determinism claim: same ``set_state`` + same ``apply_joint_torques``
      sequence + same ``step`` counts => identical ``get_state`` sequence, on the same machine,
      library version and headless mode (see ``spikes/pybullet_determinism``).
    * State: ``get_state`` returns the full generalised state as a float64 copy (joint positions,
      joint velocities and, for floating-base robots, base pose and twist) in a fixed order the
      environment documents. ``set_state`` writes such a vector AND clears the simulator's
      internal caches (contacts, solver warm start): deterministic as a procedure, but NOT an
      exact continuation of the run the vector was read from (spike: diverges with contacts).
      Exact continuation needs ``save`` / ``restore`` (native snapshots, internal caches
      included). Finite-difference linearisation evaluates every perturbation the same way:
      ``restore(base)``, ``set_state(x +/- eps)``, then step.
    * Actuation: ``apply_joint_torques`` sets the torque of every actuated joint (length =
      actuated joint count) and holds it until the next call. Environments map their action
      space onto it (clipping and scaling are the environment's job, reported in
      ``info["applied_action"]``).
    * ``render`` returns an RGB array (height, width, 3) uint8, or None if unsupported; it must
      not change the simulation state. ``close`` is idempotent.
    """

    name: str
    timestep: float

    def version(self) -> str:
        """Simulator library version, recorded in ``RunRecord.versions``."""
        ...

    def reset(self) -> None:
        """Rebuild the world in its canonical (model-file) configuration. Uses no RNG."""
        ...

    def step(self, n: int = 1) -> None:
        """Advance ``n`` physics steps of ``timestep`` seconds."""
        ...

    def get_state(self) -> np.ndarray:
        """Full generalised state, float64 copy, in the environment-documented order."""
        ...

    def set_state(self, state: np.ndarray) -> None:
        """Write a ``get_state``-layout vector; clears internal caches (not an exact restore)."""
        ...

    def save(self) -> int:
        """Exact in-memory snapshot of the whole simulation; returns a handle for ``restore``."""
        ...

    def restore(self, handle: int) -> None:
        """Return exactly to a ``save`` snapshot (same continuation, bit for bit)."""
        ...

    def apply_joint_torques(self, torques: np.ndarray) -> None:
        """Set and hold the torque of every actuated joint (N*m, or N for prismatic joints)."""
        ...

    def render(self, width: int, height: int) -> np.ndarray | None:
        """RGB frame (height, width, 3) uint8, or None if this backend cannot render."""
        ...

    def close(self) -> None:
        """Release the simulator. Idempotent."""
        ...
