# API surface (Phase 1)

Status: proposed in architecture review 3 (ADR 0016). Code: `rcl/api/` (thin; no domain logic). The frontend talks only to these endpoints (ADR 0006). All bodies are JSON. Models are Pydantic classes in `rcl/api`; the schemas below name their fields.

## Identifiers: experiment vs run

An **Experiment** is the reusable definition (config), a **Run** is one execution of it (ADR 0009). Phase 1 creates exactly one Run per Experiment, but the API always returns both ids and always puts status, progress and results on the **run**. Clients poll `GET /runs/{run_id}`, never the experiment, for status. `GET /experiments/{id}` returns the definition plus a summary of its runs.

## Endpoints

| Method and path | Request | Response | Notes |
|---|---|---|---|
| `GET /environments` | none | `list[EntryInfo]` | `Entry.info()`; sorted by id |
| `GET /controllers` | none | `list[EntryInfo]` | same shape |
| `GET /controllers/compatible?env_id=<id>` | query `env_id` | `list[EntryInfo]` | `Registry.compatible_controllers` (ADR 0014); 404 if `env_id` unknown |
| `GET /metrics` | none | `list[MetricInfo]` | metrics the experiment manager computes (fixed list in Phase 1) |
| `POST /experiments` | `ExperimentConfig` | 202 `CreatedExperiment` | validates, resolves params, rejects incompatible pairs, queues |
| `GET /experiments/{experiment_id}` | none | `ExperimentDetail` | definition + `runs: list[RunSummary]`; 404 if unknown |
| `GET /runs/{run_id}` | none | `RunDetail` | status, progress, stage, error, and when `done` the metrics and episodes; 404 if unknown |
| `GET /runs/{run_id}/trajectory/{episode}` | path `episode` = 0-based index | `TrajectoryOut` | 404 if run or episode is unknown; 409 if the run is not `done` |

## Schemas

`EntryInfo`: `{id: str, kind: "environment" | "controller", metadata: EnvMetadata | ControllerMetadata (JSON dump), params_schema: JSON Schema object ({} when the entry has no params)}`. The frontend renders forms from `params_schema` and labels from `metadata` (name, description, `state_labels`, ...).

`MetricInfo`: `{name: str, description: str, unit: str | null, higher_is_better: bool, companions: [{name: str, unit: str | null}]}`. `unit` is null when the unit is the environment's `TaskSpec.output_unit`; the concrete unit of a result is always in `MetricOut.unit`. Definitions: `rcl/metrics/DEFINITIONS.md` (`MetricValue`). Phase 1 returns `return` and `angle_error` (companion `max_abs_angle_error`).

`CreatedExperiment`: `{experiment_id: str, run_ids: list[str], status: "queued"}` (Phase 1: one run id).

`ExperimentDetail`: `{experiment_id, name, env_id, controller_id, config: ExperimentConfig (resolved), config_hash, created_at, runs: list[RunSummary]}`.

`RunSummary`: `{run_id, experiment_id, status: "queued" | "running" | "done" | "failed", stage: str | null, progress: float (0..1), seed: int, queued_at, started_at | null, finished_at | null}`. `stage` is one of `training`, `evaluating`, `computing_metrics`, `saving` while running, else null. `progress` is written by the job manager from the `progress` callback of `BaseController.train` and the rollout episode count (ADR 0007, ADR 0015).

`RunDetail`: `RunSummary` plus `error: {type, message, stage} | null` (when `failed`), `versions`, `derived_seeds`, `code_version`, `task_spec`, and, when `done`, `episodes: list[{index, seed, terminated, truncated, success, n_steps}]`, `metrics: list[MetricOut]` and `train_curve: list[[timestep, mean_return]]` (empty for classical controllers).

`MetricOut`: `{metric_name: str, episode_index: int | null, value: float | null, unit: str, status: "ok" | "censored" | "undefined"}`; one entry per `METRIC_RESULT` row, so companion values appear under their own `metric_name`; `episode_index` null is a run-level value; `value` null means NaN (`undefined`).

`TrajectoryOut`: `{episode, seed, dt, success, terminated, truncated, time: list[float] (T+1), states: list[list[float]] (T+1, state_dim), actions: list[list[float]] (T, action dim), rewards: list[float] (T), state_labels: list[str], state_units: list[str], action_labels: list[str], action_units: list[str]}`. The labels and units are copied from the environment's `env_metadata` (empty lists if the environment defines none), so plots label axes without environment knowledge.

## Errors

- **422** invalid body or params, and incompatible pairs: `{"detail": [{"loc": [...], "msg": str, "type": str}, ...]}`. For params errors the items are `ParamValidationError.errors` with `loc` prefixed by `["body", "environment" | "controller", "params"]`; an unknown id uses `loc ["body", "environment" | "controller", "id"]`, type `unknown_id`; an incompatible pair uses `loc ["body", "controller", "id"]`, type `incompatible`. Nothing is queued on 422 (see `sequence-run-experiment.md`).
- **404** `{"detail": str}` for unknown ids.
- **409** `{"detail": str}` for a trajectory of a run that is not `done`.
- A failed run is not an HTTP error: `GET /runs/{id}` returns 200 with `status: "failed"` and `error`.

## Frontend notes

- The frontend builds the `POST /experiments` JSON from the schemas it fetched (`params_schema`, `metadata`) and plain HTTP only. It imports nothing from `rcl` (enforced by the `frontend` import-linter contract), so it cannot reuse `ExperimentConfig`; it assembles `{name, seed, environment: {id, params}, controller: {id, params}, evaluation: {episodes}, training?: {total_timesteps}}` itself. `training` is sent only when the selected controller's `metadata.requires_training` is true. Field names are fixed by this doc, not by schema discovery.
- Form renderer: the registry restricts params models to a JSON-schema subset (`registry.md` section 3), so the renderer must handle exactly: scalar types (`string`, `number`, `integer`, `boolean`) with `minimum`/`maximum`/`exclusiveMinimum`/`exclusiveMaximum`; `enum` (Python `Enum` and `Literal`; `$ref` to a `$defs` entry that only holds an `enum` must be dereferenced); `anyOf` with a `null` branch (an optional field: render the inner widget plus a "none" toggle); arrays of numbers or strings, in both forms `items` (variable-length list) and `prefixItems` with `minItems`/`maxItems` (fixed-length tuple); and `default` (pre-fill each widget; send only values the user changed or all of them, both are valid because the server resolves defaults). `title` and `description` are labels and help text. Anything else is a registry bug, not a frontend case.
- Server-side validation is authoritative: map each 422 item `loc` (after the `body` prefix) to the offending widget and show `msg` next to it.

## Phase 2 additions: benchmarks

Additive; nothing above changes. A **suite** is a benchmark definition (spec verbatim, ADR 0009: one Experiment per environment-controller cell, one Run per seed); a **benchmark run** is one execution of it (poll this for status and progress).

| Method and path | Request | Response | Notes |
|---|---|---|---|
| `POST /benchmarks` | `BenchmarkSpec` `{name, environments: [{id, params}], controllers: [{id, params}], seeds: [int], episodes, training?: {total_timesteps}}` | 202 `{suite_id, benchmark_run_id, status: "queued", skipped: [{env_id, controller_id, reason}]}` | Validates the whole grid first (422 with `loc` under `["body", "environments"\|"controllers", index, ...]` or `["body", "training"]`). Incompatible pairs are skipped and reported, not errors; 422 if none is runnable. Same seeds, episodes, training budget and environment definition for every cell (fairness). |
| `GET /benchmarks` | none | `list[{suite_id, name, created_at}]` | newest first |
| `GET /benchmarks/{suite_id}` | none | `{suite_id, name, created_at, spec, benchmark_runs: [{benchmark_run_id, status, progress}], experiments: [{experiment_id, env_id, controller_id, config_hash, runs: [{run_id, seed, status}]}]}` | 404 if unknown. A failed run does not stop the benchmark; the benchmark run ends `failed` if any run failed. |
| `GET /benchmarks/{suite_id}/summary?env_id=` | optional `env_id` | `list[SummaryOut]` | Cross-seed statistics over finished runs. `SummaryOut`: `{env_id, controller_id, metric_name, unit, n_valid, n_total, n_censored, mean, std, ci_low, ci_high, ci_method: "t"\|"bootstrap", low_n}`; NaN is sent as `null`. Interval rules: `rcl/metrics/DEFINITIONS.md`. |
| `GET /benchmarks/{suite_id}/paired?env_id=&metric=&a=&b=` | query | `SummaryOut` with `controller_id "a-b"` | Per-seed differences a - b over common valid seeds. |

`GET /metrics` entries gained `ci_method` (`"t"` or `"bootstrap"`). Optional training tracking: set `RCL_MLFLOW_URI` to log training curves and checkpoints of trained runs to MLflow (the database remains the source of truth).
