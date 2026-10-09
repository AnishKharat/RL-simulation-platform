# 0009. Experiment vs Run data model

- Status: Proposed
- Date: 2026-10-10

## Context
A multi-seed benchmark needs many executions of the same definition. `docs/architecture/data-model.md` separates
the reusable definition from one execution but leaves the multi-seed shape open ("decision to confirm").

## Decision (proposed)
- `EXPERIMENT` = reusable definition: `env_id`, `controller_id`, verbatim validated `config`, `config_hash`
  (sha256 of canonical JSON), optional `suite_id`.
- `RUN` = one execution of an experiment: `status` (`queued | running | done | failed`), `seed`, `derived_seeds`,
  `versions`, `code_version`, `error`/`error_stage`, timestamps.
- Children hang off `RUN`: `EPISODE`, `METRIC_RESULT`, `ARTIFACT`.
- Phase 1 creates exactly one Run per Experiment.
- Multi-seed benchmarks (Phase 2): preferred is one Run per seed under a shared Experiment, so aggregation is
  `GROUP BY experiment_id`. Because `ExperimentConfig.seed` is part of the config (ADR 0004), this needs either
  config_hash computed with the seed excluded or the seed overridden per Run; to be settled before Phase 2.
- `BENCHMARK_SUITE` / `BENCHMARK_RUN` / `AGGREGATE_RESULT` arrive in Phase 2.

## Consequences
- Re-running an experiment (retry, rerun after a code change) adds a Run, preserving history.
- Aggregation queries stay simple under the preferred option.
- Open: how `config.seed` and `config_hash` interact with per-Run seeds (above).

## Alternatives
- One Experiment per (controller, env, seed) cell: no run-level grouping, so aggregation needs a suite join and
  config duplication.
- Single merged table: loses the definition/execution distinction and retry history.
