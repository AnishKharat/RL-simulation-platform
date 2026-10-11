"""HTTP access to the API. The frontend imports nothing else from rcl (ADR 0006)."""

from __future__ import annotations

import os
from typing import Any

import httpx

# Tests inject an httpx-compatible client (FastAPI's TestClient) here.
override: httpx.Client | None = None
_default: httpx.Client | None = None


def _http() -> httpx.Client:
    global _default
    if override is not None:
        return override
    if _default is None:
        _default = httpx.Client(base_url=os.environ.get("RCL_API_URL", "http://localhost:8000"))
    return _default


def _get(path: str, **params: Any) -> Any:
    r = _http().get(path, params=params or None)
    r.raise_for_status()
    return r.json()


def environments() -> list[dict[str, Any]]:
    return list(_get("/environments"))


def compatible_controllers(env_id: str) -> list[dict[str, Any]]:
    return list(_get("/controllers/compatible", env_id=env_id))


def create_experiment(config: dict[str, Any]) -> tuple[int, Any]:
    """POST the config; returns ``(status_code, json)`` (422 bodies carry per-field errors)."""
    r = _http().post("/experiments", json=config)
    return r.status_code, r.json()


def run(run_id: str) -> dict[str, Any]:
    return dict(_get(f"/runs/{run_id}"))


def trajectory(run_id: str, episode: int) -> dict[str, Any]:
    return dict(_get(f"/runs/{run_id}/trajectory/{episode}"))


def controllers() -> list[dict[str, Any]]:
    return list(_get("/controllers"))


def metrics() -> list[dict[str, Any]]:
    return list(_get("/metrics"))


def create_benchmark(spec: dict[str, Any]) -> tuple[int, Any]:
    """POST a benchmark spec; returns ``(status_code, json)`` (422 bodies list the problems)."""
    r = _http().post("/benchmarks", json=spec)
    return r.status_code, r.json()


def benchmarks() -> list[dict[str, Any]]:
    return list(_get("/benchmarks"))


def benchmark(suite_id: str) -> dict[str, Any]:
    return dict(_get(f"/benchmarks/{suite_id}"))


def benchmark_summary(suite_id: str, env_id: str | None = None) -> list[dict[str, Any]]:
    return list(_get(f"/benchmarks/{suite_id}/summary", **({"env_id": env_id} if env_id else {})))


def perturbations() -> list[dict[str, Any]]:
    return list(_get("/perturbations"))


def robustness(suite_id: str, env_id: str, metric: str) -> tuple[int, Any]:
    """Degradation curves of a sweep benchmark; ``(status_code, json)`` (422: not a sweep)."""
    r = _http().get(
        f"/benchmarks/{suite_id}/robustness", params={"env_id": env_id, "metric": metric}
    )
    return r.status_code, r.json()
