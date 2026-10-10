# Architecture Status

**State: FROZEN**  <!-- DRAFT | IN REVIEW | FROZEN -->

Implementation (anything beyond docs, stubs, spikes) is allowed only when this says FROZEN.

## Review iterations (min. 3 on Phases 1–3)

| # | Date | Focus | Changes / ADRs | Done |
|---|------|-------|----------------|------|
| 1 | 2026-10-10 | Interfaces, registry, config schema consistency | Registry/env_metadata fix, controller `reset()`, rollout moved to experiments (ADR 0013), TaskSpec cleanup, requires_training in metadata, single param-schema source, seed/config_hash, EnvSpec validators, EnvWrapper | ☑    |
| 2 | 2026-10-10 | Extensibility: Phases 3/4/7 fit, compatibility, training, determinism | ADRs 0014-0015; configure(linear_model), train(env_factory), multi-output TaskSpec, T/N indexing, terminated=failure, schema_version+perturbations, perturbations/ros_adapter in contracts, labels/units, param-schema subset | ☑    |
| 3 | 2026-10-10 | Phase 1 paper dry-run + global consistency | ADR 0016 + api.md; train progress hook, info[applied_action], indirect-import contracts, CI extras, registry kind/is_compatible checks, seed recipe (train/eval), rollout pinned, wrap-aware success, MetricValue storage, create_all until Phase 5 | ☑    |
| 4 | 2026-10-10 | Phase 3 readiness: PhysicsBackend, multi-DOF shapes, controllers on new robots, PyBullet | ADR 0019 (PhysicsBackend stub in core; proposed `rcl.physics` layer = rule change), ADR 0020 (optional extra, conditional registration, CI); `multi-dof-audit.md`, `phase3-robots.md`; spike: deterministic, `set_state` not exact (save/restore added). User decisions: ADRs 0019/0020 accepted, 2-DOF planar arm, assets in sibling folder | ☑    |
| 5 | 2026-10-11 | Phase 4 readiness: perturbation wrappers, mass hook, robustness metrics, fairness, API | ADR 0021, `phase4-robustness.md`. User decisions: ADR approved; disturbance via actuator channel (no `apply_disturbance`); import-linter contract for perturbations; config_hash from spec inputs only | ☑ |

## Freeze checklist
- ☑ Component, sequence and data-model diagrams in `docs/architecture/`
- ☑ `core/` interface stubs reviewed
- ☑ Registry design doc
- ☑ Experiment config schema
- ☑ Metric definitions (`rcl/metrics/DEFINITIONS.md`)
- ☑ Spikes passed: SB3 custom env, FastAPI background job, registry auto-discovery
- ☑ ADR for every major choice
- ☑ Import-linter rules in place

Frozen on: 2026-10-10 (approved by user). Re-opened for review iteration 4 on 2026-10-10 and re-frozen on 2026-10-10 (approved by user).
Re-opened for review iteration 5 on 2026-10-11 and re-frozen on 2026-10-11 (approved by user).
