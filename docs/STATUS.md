# Project Status

> Read this FIRST every session. Keep under ~20 lines. Overwrite, don't append history (git log is the history).

- **Current phase:** 4 (Robustness) in progress on `main`. `phase-3-complete` stays as tagged. Do NOT tag `phase-4-complete` until the user confirms
- **Architecture:** FROZEN again after review iteration 5 (2026-10-11). ADR 0021 Accepted: registry kind `perturbation`, `StepPerturbation`/`ParamPerturbation` in core, disturbance via the actuator channel (no `apply_disturbance`). Design: `docs/architecture/phase4-robustness.md`
- **Stage 0 and Stage 1 done and merged:** perturbation kind + import-linter contract, five perturbations (`obs_noise`, `action_noise`, `action_delay`, `disturbance`, `param_scale`), manager builds train/eval stacks, rollout records `perturbation_steps`, spec-only `config_hash` test
- **Stage 2 done:** `failure_count`, `recovery_time` (in the manager metric list) and pure degradation helpers (`breakdown_magnitude`, `relative_curve`, `mean_by_magnitude`) in `rcl/metrics/robustness.py`; cross-seed CIs for curves belong to Stage 3 analytics
- **Stage 3 done:** `execute_sweep` (train once, evaluate under each level stack), `BenchmarkSpec.perturbation` (`RobustnessSpec`: id, params, levels; nominal level auto-added), one experiment per level in the suite, `suite_metric_rows` now carries `perturbations`, `rcl.analytics.degradation_curves` (per-level CI, relative, breakdown). In-memory `run_benchmark` rejects sweeps (stored runs only)
- **Stages 4-6 done:** `/perturbations`, `/benchmarks/{id}/robustness`, Robustness dashboard page, perturbation modularity proof, six sweeps in `docs/benchmarks/phase4/` with an honest README. Opus review done; fixes: train-stage perturbations with SB3 auto-reset, `action_delay` continuous-only, params field on the page, breakdown only for success_rate, nominal run owns the sweep checkpoint, 422 for an unregistered sweep perturbation
- **Next:** user decision on `recovery_time` band (docs/OPEN_QUESTIONS.md), then user confirms Phase 4 exit before tagging `phase-4-complete`
- **Known limits:** `obs_noise` std is in absolute observation units; `param_scale` is one value per run; no PyBullet run with perturbations yet; CI status not checked (no `gh` here)
- **Phase 7:** marked out of scope for now (simulation-only)
- **Env:** project `.venv`; checks via `.venv/Scripts/python`
- **Checks:** pytest 345 pass | ruff clean | mypy clean | lint-imports 12 kept
- **Last updated:** 2026-10-11 by orchestrator
