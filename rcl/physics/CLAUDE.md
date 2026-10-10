# rcl/physics — local rules

- `PhysicsBackend` implementations only (contract: `rcl/core/physics.py`, ADR 0019). One simulator per module.
- Import only `rcl.core` (+ the simulator). Import the simulator lazily (inside methods/constructors), never at module top level (ADR 0020).
- No RNG, no global state; every backend instance owns its own simulator client and closes it idempotently.
- Headless only in benchmarks; never real-time stepping.
- Tests: `pytest.importorskip(<simulator>)`; cover reset/step/get/set_state, save/restore exactness, torque application, close idempotency, determinism.
