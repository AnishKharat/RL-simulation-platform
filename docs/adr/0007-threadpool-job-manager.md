# 0007. Background jobs via ThreadPoolExecutor job manager

- Status: Accepted
- Date: 2026-10-10

## Context
Training and evaluation take seconds to hours and must not block requests. The FastAPI spike
(`docs/architecture/spike-findings.md` section 2) compared `BackgroundTasks` and a module-level
`ThreadPoolExecutor`; both passed (202 + id, queued -> running -> done/failed, progress 0..1, 404 on unknown id).

## Decision
An app-owned `JobManager` wraps a `ThreadPoolExecutor(max_workers=N)`, created in the FastAPI lifespan and shut
down there. Job function signature is `(job_id, config) -> result`.
- Submit returns immediately; extra jobs beyond `max_workers` stay `queued`.
- The job store is thread-safe (lock). Worker exceptions are caught at the job boundary and recorded as `failed`
  plus an error message.
- Status and progress are persisted to the DB in Phase 1; on startup, orphaned `running` jobs are marked `failed`.
- Cancellation, when added, is cooperative (flag checked in the training callback); threads cannot be killed.

## Consequences
- Bounded concurrency, a `Future` handle, and tests that work under `TestClient` (BackgroundTasks blocks there).
- Threads share the GIL: heavy pure-Python simulation can starve the API. Mitigation: swap to
  `ProcessPoolExecutor` (needs picklable args, progress via DB rows or a `Manager`); the fixed signature keeps
  the swap local. Not yet spiked.
- In-memory state is lost on restart until the DB persistence lands.
- SB3/torch imports stay lazy so API start-up is fast.

## Alternatives
- `BackgroundTasks`: no concurrency cap, no handle, tied to request lifecycle, blocks in tests.
- Celery / RQ with a broker: overkill for a single-machine MVP; reconsider with Postgres/multi-host.
- `multiprocessing` from the start: higher complexity (pickling, spawn on Windows) before it is needed.
