# Project Status

> Read this FIRST every session. Keep under ~20 lines. Overwrite, don't append history (git log is the history).

- **Current phase:** 3 (PyBullet robots): architecture review iteration 4 done as docs/stubs/spike; **architecture IN REVIEW** (not frozen), no implementation yet
- **Last completed:** ADR 0019 (PhysicsBackend protocol stub in `rcl/core/physics.py`, proposed `rcl.physics` layer), ADR 0020 (optional `physics` extra, conditional registration, CI job), `docs/architecture/multi-dof-audit.md`, `docs/architecture/phase3-robots.md`, `spikes/pybullet_determinism/` (PyBullet 3.2.7 built from source; seeded runs bit-identical; manual set_state not an exact continuation)
- **Next task:** user decides the 4 open questions (physics-layer rule change, arm model, URDF assets, ADR 0020) and approves refreeze; then Phase 3 implementation in the proposed order
- **Blockers / open questions:** see `docs/OPEN_QUESTIONS.md` (4 open)
- **Checks:** pytest 248 pass | ruff clean | mypy clean | lint-imports 9 kept
- **Last updated:** 2026-10-10 by orchestrator
