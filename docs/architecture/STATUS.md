# Architecture Status

**State: IN REVIEW**  <!-- DRAFT | IN REVIEW | FROZEN -->

Implementation (anything beyond docs, stubs, spikes) is allowed only when this says FROZEN.

## Review iterations (min. 3 on Phases 1–3)

| # | Date | Focus | Changes / ADRs | Done |
|---|------|-------|----------------|------|
| 1 | 2026-10-10 | Interfaces, registry, config schema consistency | Registry/env_metadata fix, controller `reset()`, rollout moved to experiments (ADR 0013), TaskSpec cleanup, requires_training in metadata, single param-schema source, seed/config_hash, EnvSpec validators, EnvWrapper | ☑    |
| 2 | 2026-10-10 | Extensibility: Phases 3/4/7 fit, compatibility, training, determinism | ADRs 0014-0015; configure(linear_model), train(env_factory), multi-output TaskSpec, T/N indexing, terminated=failure, schema_version+perturbations, perturbations/ros_adapter in contracts, labels/units, param-schema subset | ☑    |
| 3 | 2026-10-10 | Phase 1 paper dry-run + global consistency | ADR 0016 + api.md; train progress hook, info[applied_action], indirect-import contracts, CI extras, registry kind/is_compatible checks, seed recipe (train/eval), rollout pinned, wrap-aware success, MetricValue storage, create_all until Phase 5 | ☑    |

## Freeze checklist
- ☑ Component, sequence and data-model diagrams in `docs/architecture/`
- ☑ `core/` interface stubs reviewed
- ☑ Registry design doc
- ☑ Experiment config schema
- ☑ Metric definitions (`rcl/metrics/DEFINITIONS.md`)
- ☑ Spikes passed: SB3 custom env, FastAPI background job, registry auto-discovery
- ☑ ADR for every major choice
- ☑ Import-linter rules in place

Frozen on: —
