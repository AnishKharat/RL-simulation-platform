# 0008. SQLAlchemy with SQLite then Postgres; DB vs MLflow roles

- Status: Accepted for SQLAlchemy/SQLite/Alembic/artifacts; Proposed for the MLflow role (Phase 2)
- Date: 2026-10-10

## Context
We need a catalog of experiments, runs, metrics and benchmarks that is queryable for comparisons, easy to run
locally, and movable to a server database. Training curves and checkpoints are large and fit experiment trackers
better. See `docs/architecture/data-model.md`.

## Decision
- SQLAlchemy models with Alembic migrations in `rcl.persistence`. SQLite for development; PostgreSQL later by
  changing the connection URL only (avoid dialect-specific types; JSON columns stay portable).
- The database is the source of truth for catalog, status, verbatim config, seeds, versions, per-episode and
  per-run metric values, benchmarks and aggregates. It must answer every comparison query without MLflow running.
- Trajectories are not DB rows: one compressed `.npz` per episode (arrays only, no pickle) in an artifact store at
  `artifacts/<run_id>/episodes/<index>.npz` (local dir now, object-store URI later), with an `ARTIFACT` row holding
  `uri`, `sha256`, size. `artifacts/` is git-ignored.
- Proposed: MLflow is an optional tracking sink from Phase 2 for training curves and checkpoints;
  `ARTIFACT.mlflow_run_id` links the two. Checkpoints and datasets are never committed.

## Consequences
- Plain SQL covers comparisons; Postgres migration is a config change if SQLite-only features are avoided.
- Two stores to keep consistent once MLflow arrives; the DB wins on disagreement.
- Model checkpoints are pickle-based zips: never load untrusted ones.
- `AGGREGATE_RESULT` is a recomputable cache, introduced in Phase 2.

## Alternatives
- MLflow as the only store: weak relational queries and benchmark modelling.
- Files/Parquet only: no transactions or status tracking.
- Postgres from day one: setup cost for a single-user MVP.
- Trajectories as DB blobs: bloat and slow bulk reads.
