# 0016. API surface

- Status: Accepted (review 3; proposed in review 2)
- Date: 2026-10-10

## Context
ADR 0006 fixed FastAPI and an HTTP-only frontend, but the endpoint set, schemas and error shape were scattered across
the sequence diagram, `registry.md` and the roadmap, and used "experiment id" and "run id" interchangeably.

## Decision
`docs/architecture/api.md` is the contract. In short:
- Registry listing: `GET /environments`, `GET /controllers`, `GET /controllers/compatible?env_id=` (replaces the
  path-style compatible-controllers route), all returning `Entry.info()`; `GET /metrics` returns the metric catalog.
- `POST /experiments` takes an `ExperimentConfig` and returns 202 `{experiment_id, run_ids, status}`.
- Status, progress, stage, errors and results live on the run: `GET /runs/{run_id}`; `GET /experiments/{id}` is the
  definition plus run summaries. Trajectories: `GET /runs/{run_id}/trajectory/{episode}`, with state/action labels and
  units from `env_metadata`.
- Validation errors are 422 with `{"detail": [{loc, msg, type}]}`, reusing `ParamValidationError.errors`.

## Consequences
- The frontend needs no `rcl` imports: forms come from `params_schema`, axes from labels/units, metrics from `/metrics`.
- Polling targets one resource (the run), so a multi-seed Experiment (Phase 2) needs no API change.
- Schemas are part of the architecture; changing them after the freeze goes through `docs/OPEN_QUESTIONS.md`.

## Alternatives
- Poll the experiment: ambiguous once an Experiment has several Runs.
- Nest runs under experiments in the URL (`/experiments/{id}/runs/{rid}`): longer paths, no extra information, since
  run ids are globally unique.
