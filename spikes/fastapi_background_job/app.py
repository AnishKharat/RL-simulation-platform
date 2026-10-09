"""Spike: long jobs behind FastAPI. Two launch styles share one job store.

POST /jobs        -> ThreadPoolExecutor (recommended)
POST /bg_jobs     -> FastAPI BackgroundTasks (for comparison)
GET  /jobs/{id}   -> {status: queued|running|done|failed, progress, result, error}
"""
from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
JOBS: dict[str, dict[str, Any]] = {}  # shortcut: in-memory, lost on restart; use DB in prod
LOCK = threading.Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=2)  # caps concurrent training jobs; extras stay queued


class JobRequest(BaseModel):
    steps: int = 10
    delay: float = 0.05
    fail_at: int | None = None


def _update(job_id: str, **kw: Any) -> None:
    with LOCK:
        JOBS[job_id].update(kw)


def work(job_id: str, req: JobRequest) -> None:
    _update(job_id, status="running")
    try:
        for i in range(1, req.steps + 1):
            time.sleep(req.delay)
            if req.fail_at == i:
                raise RuntimeError(f"boom at step {i}")
            _update(job_id, progress=i / req.steps)
        _update(job_id, status="done", result={"steps": req.steps})
    except Exception as e:  # noqa: BLE001 - job boundary: never let a worker die silently
        _update(job_id, status="failed", error=str(e))


def _new_job() -> str:
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"id": job_id, "status": "queued", "progress": 0.0}
    JOBS[job_id].update(result=None, error=None)
    return job_id


@app.post("/jobs", status_code=202)
def start_job(req: JobRequest) -> dict[str, str]:
    job_id = _new_job()
    EXECUTOR.submit(work, job_id, req)
    return {"id": job_id}


@app.post("/bg_jobs", status_code=202)
def start_bg_job(req: JobRequest, bg: BackgroundTasks) -> dict[str, str]:
    job_id = _new_job()
    bg.add_task(work, job_id, req)
    return {"id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with LOCK:
        if job_id not in JOBS:
            raise HTTPException(404, "unknown job")
        return dict(JOBS[job_id])
