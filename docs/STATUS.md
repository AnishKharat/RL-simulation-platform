# Project Status

> Read this FIRST every session. Keep under ~20 lines. Overwrite, don't append history (git log is the history).

- **Current phase:** 3 (PyBullet robots), implementation in progress; architecture FROZEN (re-frozen 2026-10-10 after review 4)
- **Last completed:** `rcl.physics` layer + PyBullet backend (ADR 0019), optional `physics` extra + conditional registration + Linux/3.11 CI job (ADR 0020), `balance_bot` (finite-difference linearisation; LQR/PID balance it), 2-DOF `planar_arm`, test that every compatible existing controller runs unchanged, PyBullet version in run records, trajectory signal selection, Phase 3 benchmarks in `docs/benchmarks/phase3/`
- **Env:** project `.venv` (`--system-site-packages`); PyBullet 3.2.7 lives only in `.venv` (built from source); run checks with `.venv/Scripts/python`
- **Next task:** answer the user's force-push and arm-controller questions (restatement requested: not found in the session), then tag `phase-3-complete` after user confirmation
- **Blockers / open questions:** the two user questions above. Rendering done: `python -m rcl.benchmark.gif` (optional `render` extra)
- **Checks:** pytest 283 pass (venv, with PyBullet) | ruff clean | mypy clean | lint-imports 11 kept | GitHub CI green incl. physics job
- **Last updated:** 2026-10-10 by orchestrator
