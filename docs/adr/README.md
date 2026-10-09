# Architecture Decision Records

One short ADR per major decision. Copy `0000-template.md`, number sequentially, never rewrite history:
supersede with a new ADR instead. "Proposed" means not yet settled in architecture review.

| # | Title | Status |
|---|-------|--------|
| 0001 | Gymnasium API as environment base | Accepted |
| 0002 | Environment/controller independence and declared compatibility | Accepted |
| 0003 | Decorator registry, pkgutil discovery, registry import rule | Accepted |
| 0004 | core holds interfaces and Pydantic configs only | Accepted |
| 0005 | Metrics as pure functions over EpisodeTrajectory | Accepted |
| 0006 | FastAPI layer from Phase 1; frontend HTTP-only | Accepted |
| 0007 | Background jobs via ThreadPoolExecutor job manager | Accepted |
| 0008 | SQLAlchemy with SQLite then Postgres; DB vs MLflow roles | Accepted (MLflow part Proposed) |
| 0009 | Experiment vs Run data model | Proposed |
| 0010 | Reproducibility record | Accepted |
| 0011 | import-linter enforcement | Accepted |
| 0012 | `env_metadata` naming and `get_env_spec` | Accepted |
| 0013 | Rollout lives in the experiments layer | Accepted |
| 0014 | Declared compatibility | Accepted |
| 0015 | Training env factory and checkpoint directory | Accepted |
| 0016 | API surface | Accepted |
