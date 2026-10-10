"""FastAPI app (docs/architecture/api.md). Thin: validate, delegate, return."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Collection
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from rcl.api.schemas import EntryInfo
from rcl.experiments import METRIC_INFO, JobManager, MetricInfo
from rcl.persistence import Repository
from rcl.registry import (
    DISCOVERY_PACKAGES,
    Kind,
    Registry,
    UnknownIdError,
    discover,
    get_registry,
    use_registry,
)


def create_app(
    *,
    db_url: str | None = None,
    artifact_root: str | None = None,
    registry: Registry | None = None,
    discovery_packages: Collection[str] = DISCOVERY_PACKAGES,
    max_workers: int = 2,
) -> FastAPI:
    """Build the app. ``registry`` / ``discovery_packages`` let tests run against an isolated
    registry; by default the process-wide registry and the standard plugin packages are used."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        reg = registry or get_registry()
        with use_registry(reg):
            discover(discovery_packages)
        repo = Repository(
            db_url or os.environ.get("RCL_DB_URL", "sqlite:///rcl.db"),
            artifact_root or os.environ.get("RCL_ARTIFACTS", "artifacts"),
        )
        repo.fail_orphans()  # runs left over from a previous process can never finish
        jobs = JobManager(repo, reg, max_workers)
        app.state.registry, app.state.repo, app.state.jobs = reg, repo, jobs
        yield
        jobs.shutdown()

    app = FastAPI(title="Robotics Control Lab", lifespan=lifespan)

    def entries(request: Request, kind: Kind) -> list[dict[str, Any]]:
        return [e.info() for e in request.app.state.registry.entries(kind)]

    @app.get("/environments", response_model=list[EntryInfo])
    def list_environments(request: Request) -> list[dict[str, Any]]:
        return entries(request, "environment")

    @app.get("/controllers", response_model=list[EntryInfo])
    def list_controllers(request: Request) -> list[dict[str, Any]]:
        return entries(request, "controller")

    @app.get("/controllers/compatible", response_model=list[EntryInfo])
    def compatible_controllers(request: Request, env_id: str) -> list[dict[str, Any]]:
        try:
            found = request.app.state.registry.compatible_controllers(env_id)
        except UnknownIdError:
            raise HTTPException(404, f"unknown environment {env_id!r}") from None
        return [e.info() for e in found]

    @app.get("/metrics", response_model=list[MetricInfo])
    def list_metrics() -> list[MetricInfo]:
        return METRIC_INFO

    return app


app = create_app()
