"""FastAPI app (docs/architecture/api.md). Thin: validate, delegate, return."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Collection
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from rcl.api.schemas import (
    CreatedExperiment,
    EntryInfo,
    ExperimentDetail,
    RunDetail,
    TrajectoryOut,
)
from rcl.core.config import ExperimentConfig
from rcl.experiments import (
    METRIC_INFO,
    ConfigError,
    JobManager,
    MetricInfo,
    MlflowTracker,
    derive_seeds,
    resolve_config,
)
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

LABELS = ("state_labels", "state_units", "action_labels", "action_units")


def create_app(
    *,
    db_url: str | None = None,
    artifact_root: str | None = None,
    registry: Registry | None = None,
    discovery_packages: Collection[str] = DISCOVERY_PACKAGES,
    max_workers: int = 1,
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
        uri = os.environ.get("RCL_MLFLOW_URI")  # optional: log training curves/checkpoints
        jobs = JobManager(repo, reg, max_workers, MlflowTracker(uri) if uri else None)
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

    @app.post("/experiments", status_code=202, response_model=CreatedExperiment)
    def create_experiment(request: Request, config: ExperimentConfig) -> CreatedExperiment:
        st = request.app.state
        try:
            resolved, _, _ = resolve_config(config, st.registry)
        except ConfigError as exc:
            detail = [{**e, "loc": ["body", *e["loc"]]} for e in exc.errors]
            raise HTTPException(422, detail) from None
        seed = resolved.seed
        exp_id, run_id = st.repo.create_experiment(resolved, seed, derive_seeds(seed))
        st.jobs.submit(run_id, resolved, seed)
        return CreatedExperiment(experiment_id=exp_id, run_ids=[run_id])

    @app.get("/experiments/{experiment_id}", response_model=ExperimentDetail)
    def get_experiment(request: Request, experiment_id: str) -> dict[str, Any]:
        found = request.app.state.repo.get_experiment(experiment_id)
        if found is None:
            raise HTTPException(404, f"unknown experiment {experiment_id!r}")
        return found  # type: ignore[no-any-return]

    @app.get("/runs/{run_id}", response_model=RunDetail)
    def get_run(request: Request, run_id: str) -> dict[str, Any]:
        repo = request.app.state.repo
        run = repo.get_run(run_id)
        if run is None:
            raise HTTPException(404, f"unknown run {run_id!r}")
        return run | (repo.get_results(run_id) if run["status"] == "done" else {})  # type: ignore[no-any-return]

    @app.get("/runs/{run_id}/trajectory/{episode}", response_model=TrajectoryOut)
    def get_trajectory(request: Request, run_id: str, episode: int) -> dict[str, Any]:
        st = request.app.state
        run = st.repo.get_run(run_id)
        if run is None:
            raise HTTPException(404, f"unknown run {run_id!r}")
        if run["status"] != "done":
            raise HTTPException(409, f"run is {run['status']}, not done")
        arrays = st.repo.load_trajectory(run_id, episode)
        episodes = st.repo.get_results(run_id)["episodes"]
        ep = next((e for e in episodes if e["index"] == episode), None)
        if arrays is None or ep is None:
            raise HTTPException(404, f"unknown episode {episode} of run {run_id!r}")
        env_id = st.repo.get_experiment(run["experiment_id"])["env_id"]
        try:
            meta = st.registry.get("environment", env_id).metadata
            labels = {k: list(getattr(meta, k)) for k in LABELS}
        except UnknownIdError:  # environment no longer registered: plot without labels
            labels = {k: [] for k in LABELS}
        t = arrays["time"]
        return {
            "episode": episode,
            "seed": ep["seed"],
            "dt": float(t[1] - t[0]),
            "success": ep["success"],
            "terminated": ep["terminated"],
            "truncated": ep["truncated"],
            "time": t.tolist(),
            "states": arrays["states"].tolist(),
            "actions": arrays["actions"].reshape(len(arrays["actions"]), -1).tolist(),
            "rewards": arrays["rewards"].tolist(),
            **labels,
        }

    return app


app = create_app()
