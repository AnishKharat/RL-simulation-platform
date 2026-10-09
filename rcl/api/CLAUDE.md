# rcl/api — local rules

- Thin FastAPI layer: validate, delegate to `experiments`/`analytics`/`registry`/`persistence`, return. No business logic.
- No hardcoded environment or controller names; everything comes from registry metadata.
- Long runs are background jobs with a status field (queued/running/done/failed); never block a request on training.
- Pydantic request/response schemas for every endpoint; frontend consumes only these (HTTP).
- Tests with FastAPI `TestClient`.
