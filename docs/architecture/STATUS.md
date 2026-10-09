# Architecture Status

**State: IN REVIEW**  <!-- DRAFT | IN REVIEW | FROZEN -->

Implementation (anything beyond docs, stubs, spikes) is allowed only when this says FROZEN.

## Review iterations (min. 3 on Phases 1–3)

| # | Date | Focus | Changes / ADRs | Done |
|---|------|-------|----------------|------|
| 1 | 2026-10-10 | Interfaces, registry, config schema consistency | Registry/env_metadata fix, controller `reset()`, rollout moved to experiments (ADR 0013), TaskSpec cleanup, requires_training in metadata, single param-schema source, seed/config_hash, EnvSpec validators, EnvWrapper | ☑    |
| 2 |      |       |                | ☐    |
| 3 |      |       |                | ☐    |

## Freeze checklist
- ☐ Component, sequence and data-model diagrams in `docs/architecture/`
- ☐ `core/` interface stubs reviewed
- ☐ Registry design doc
- ☐ Experiment config schema
- ☐ Metric definitions (`rcl/metrics/DEFINITIONS.md`)
- ☐ Spikes passed: SB3 custom env, FastAPI background job, registry auto-discovery
- ☐ ADR for every major choice
- ☐ Import-linter rules in place

Frozen on: —
