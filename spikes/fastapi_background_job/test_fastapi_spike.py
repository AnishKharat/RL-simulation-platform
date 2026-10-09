"""Run only via: pytest spikes/fastapi_background_job (not collected by root `pytest`)."""
from __future__ import annotations

import threading
import time
from typing import Any

import app as spike
import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient


def wait_done(get: Any, job_id: str, timeout: float = 5.0) -> dict[str, Any]:
    end = time.time() + timeout
    while time.time() < end:
        j = get(f"/jobs/{job_id}").json()
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.02)
    raise AssertionError("timeout")


def test_executor_nonblocking_and_transitions() -> None:
    c = TestClient(spike.app)
    t0 = time.time()
    job_id = c.post("/jobs", json={"steps": 10, "delay": 0.05}).json()["id"]
    assert time.time() - t0 < 0.3  # job takes >= 0.5s; POST returned immediately
    seen = set()
    end = time.time() + 5
    while time.time() < end:
        j = c.get(f"/jobs/{job_id}").json()
        seen.add(j["status"])
        if j["status"] == "done":
            break
        time.sleep(0.01)
    assert j["progress"] == 1.0 and j["result"] == {"steps": 10}
    assert "running" in seen


def test_failure_path_and_404() -> None:
    c = TestClient(spike.app)
    job_id = c.post("/jobs", json={"steps": 5, "delay": 0.01, "fail_at": 3}).json()["id"]
    j = wait_done(c.get, job_id)
    assert j["status"] == "failed" and "boom" in j["error"] and 0 < j["progress"] < 1
    assert c.get("/jobs/nope").status_code == 404


def test_queueing_beyond_worker_cap() -> None:
    c = TestClient(spike.app)
    ids = [c.post("/jobs", json={"steps": 10, "delay": 0.05}).json()["id"] for _ in range(4)]
    assert "queued" in {c.get(f"/jobs/{i}").json()["status"] for i in ids}  # 2 workers, 4 jobs
    assert all(wait_done(c.get, i)["status"] == "done" for i in ids)


def test_background_tasks_testclient_blocks() -> None:
    # Gotcha: TestClient waits for BackgroundTasks, so it cannot show non-blocking behaviour.
    c = TestClient(spike.app)
    t0 = time.time()
    job_id = c.post("/bg_jobs", json={"steps": 10, "delay": 0.05}).json()["id"]
    assert time.time() - t0 >= 0.5
    assert c.get(f"/jobs/{job_id}").json()["status"] == "done"


@pytest.fixture
def live_server():  # noqa: ANN201
    server = uvicorn.Server(uvicorn.Config(spike.app, port=8765, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    while not server.started:
        time.sleep(0.02)
    yield "http://127.0.0.1:8765"
    server.should_exit = True
    th.join(5)


def test_background_tasks_nonblocking_on_real_server(live_server: str) -> None:
    with httpx.Client(base_url=live_server) as c:
        t0 = time.time()
        job_id = c.post("/bg_jobs", json={"steps": 10, "delay": 0.05}).json()["id"]
        assert time.time() - t0 < 0.3
        assert wait_done(c.get, job_id)["status"] == "done"
