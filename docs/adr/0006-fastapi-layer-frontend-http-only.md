# 0006. FastAPI layer from Phase 1; frontend HTTP-only

- Status: Accepted
- Date: 2026-10-10

## Context
The Streamlit dashboard could import the experiment manager directly and be done quickly, but that welds UI to
backend, blocks other clients (CLI, notebooks, a future web UI) and makes long-running jobs awkward.

## Decision
- A FastAPI service (`rcl.api`) exists from Phase 1 and is the only entry to the backend. The frontend
  (`rcl.frontend`, Streamlit) talks to it over HTTP only and imports no other `rcl` package (import-linter,
  ADR 0011).
- The API imports only `experiments`, `analytics`, `registry`, `persistence`; it calls `discover()` at startup and
  never names a concrete robot or controller.
- Registry-driven endpoints: `GET /environments`, `GET /controllers`,
  `GET /environments/{id}/compatible-controllers`; forms are rendered from each entry's `params_schema`.
- Long operations return `202` with a job id and are polled (ADR 0007). Request/response schemas are Pydantic.
- Schemas needed by the frontend are served by the API, not imported from `core`.

## Consequences
- The frontend is replaceable and can run on another host; the API is testable with `TestClient`.
- Extra cost in Phase 1: HTTP client code, schema duplication risk, CORS/ports in dev (`uvicorn` + `streamlit`).
- The API is the place to later add auth and versioning; neither is decided yet.

## Alternatives
- Streamlit importing the experiment manager: faster start, but violates the "frontend talks to the API only" rule
  and entangles Streamlit reruns with job lifecycle.
- Flask / Django: no async or automatic OpenAPI/Pydantic integration; FastAPI spiked and passed
  (`spike-findings.md` section 2).
- Building the API in a later phase: forces a frontend rewrite.
