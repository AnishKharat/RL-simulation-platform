# Project Status

> Read this FIRST every session. Keep under ~20 lines. Overwrite, don't append history (git log is the history).

- **Current phase:** 4 (Robustness) in progress on `main`. `phase-3-complete` stays as tagged. Do NOT tag `phase-4-complete` until the user confirms
- **Architecture:** FROZEN again after review iteration 5 (2026-10-11). ADR 0021 Accepted: registry kind `perturbation`, `StepPerturbation`/`ParamPerturbation` in core, disturbance via the actuator channel (no `apply_disturbance`). Design: `docs/architecture/phase4-robustness.md`
- **Stage 0 and Stage 1 done and merged:** perturbation kind + import-linter contract, five perturbations (`obs_noise`, `action_noise`, `action_delay`, `disturbance`, `param_scale`), manager builds train/eval stacks, rollout records `perturbation_steps`, spec-only `config_hash` test
- **Next (Stage 2):** robustness metrics (failure_count, recovery_time, degradation curves) as pure functions matching `rcl/metrics/DEFINITIONS.md`. Then Stage 3 suite runner, 4 API + dashboard, 5 modularity proof, 6 benchmark (balance_bot, CartPole; skip planar_arm), then one opus review
- **To document at Stage 6:** the disturbance is applied before the environment's clipping and saturates at the actuator limit (docs/benchmarks/phase4/README.md)
- **Known limits:** `obs_noise` std is in absolute observation units; `param_scale` is one value per run; no PyBullet run with perturbations yet; CI status not checked (no `gh` here)
- **Phase 7:** marked out of scope for now (simulation-only)
- **Env:** project `.venv`; checks via `.venv/Scripts/python`
- **Checks:** pytest 0 pass | ruff clean | mypy clean | lint-imports 12 kept
- **Last updated:** 2026-10-11 by orchestrator
