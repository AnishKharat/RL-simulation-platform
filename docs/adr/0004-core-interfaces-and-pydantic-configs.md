# 0004. core holds interfaces and Pydantic configs only

- Status: Accepted
- Date: 2026-10-10

## Context
Every other package depends on `core`, so anything placed there is coupled to the whole project. We also need
configs that validate, serialize to YAML/JSON/DB and generate JSON schema for forms.

## Decision
`rcl.core` contains only: abstract interfaces (`BaseEnvironment`, `BaseController`), shared types, and the
experiment config schema. It imports nothing from `rcl` (third-party `gymnasium`, `numpy`, `pydantic` are fine).
- Declarative specs (`EnvMetadata`, `EnvSpec`, `ControllerMetadata`, `TaskSpec`, `TrainConfig`, `TrainResult`) are
  frozen Pydantic models with `extra="forbid"`.
- Array-carrying types (`LinearModel`, `EpisodeTrajectory`) are frozen dataclasses that validate shape invariants
  in `__post_init__` (Pydantic is a poor fit for ndarray payloads).
- `ExperimentConfig` (`name`, `seed`, `environment`, `controller`, optional `training`, `evaluation`) is the single
  source of truth for a run; `ComponentConfig.params` stays a plain dict and is validated later against the
  registry entry's param model, because core cannot know concrete schemas.
- `RunRecord` (seed, config, library versions, code version) lives in core as the reproducibility block (ADR 0010).
- No stores, no registry, no behaviour beyond validation.

## Consequences
- Configs round-trip through `model_dump(mode="json")` verbatim into the DB and API.
- Param validation is two-stage (structure in core, content in registry); a bad `params` dict is caught at
  experiment start, not at config load.
- Rejecting unknown fields (`extra="forbid"`) catches typos but makes config evolution an explicit schema change.

## Alternatives
- Dataclasses everywhere: no JSON schema or coercion for free.
- Pydantic for trajectories: copying/validation overhead on large arrays.
- Typed per-component config unions in core: would make core know every robot.

## Revisions (architecture review 1)
- `EnvSpec` validates its action fields at construction: DISCRETE needs `n_actions` and no bounds; CONTINUOUS needs equal-length `action_low/high` (low <= high, length = prod(action_shape) when given) and no `n_actions`.
