# rcl/core — local rules

- Interfaces and types ONLY: `BaseEnvironment`, `BaseController`, `EnvSpec`, `TaskSpec`, `EpisodeTrajectory`, config schemas.
- Imports nothing from the `rcl` package. No heavy deps (no torch, SB3, pybullet).
- Every interface has a docstring stating its contract.
- Interface changes need an ADR and a note in `docs/architecture/`. Once architecture is FROZEN, raise via `docs/OPEN_QUESTIONS.md` instead of editing.
- Pydantic for configs/schemas; type hints on all public functions.
