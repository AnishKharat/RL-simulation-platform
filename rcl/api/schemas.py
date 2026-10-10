"""Request/response models (field names fixed by docs/architecture/api.md)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel


class EntryInfo(BaseModel):
    id: str
    kind: Literal["environment", "controller"]
    metadata: dict[str, Any]
    params_schema: dict[str, Any]


class CreatedExperiment(BaseModel):
    experiment_id: str
    run_ids: list[str]
    status: Literal["queued"] = "queued"


class RunSummary(BaseModel):
    run_id: str
    experiment_id: str
    status: Literal["queued", "running", "done", "failed"]
    stage: str | None
    progress: float
    seed: int
    queued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class ExperimentDetail(BaseModel):
    experiment_id: str
    name: str
    env_id: str
    controller_id: str
    config: dict[str, Any]
    config_hash: str
    created_at: datetime
    runs: list[RunSummary]


class RunError(BaseModel):
    type: str
    message: str
    stage: str | None


class EpisodeOut(BaseModel):
    index: int
    seed: int
    terminated: bool
    truncated: bool
    success: bool
    n_steps: int


class MetricOut(BaseModel):
    metric_name: str
    episode_index: int | None
    value: float | None
    unit: str
    status: Literal["ok", "censored", "undefined"]


class RunDetail(RunSummary):
    error: RunError | None
    versions: dict[str, str] | None
    derived_seeds: dict[str, int]
    code_version: str | None
    task_spec: dict[str, Any] | None
    episodes: list[EpisodeOut] = []
    metrics: list[MetricOut] = []
    train_curve: list[tuple[int, float]] = []


class TrajectoryOut(BaseModel):
    episode: int
    seed: int
    dt: float
    success: bool
    terminated: bool
    truncated: bool
    time: list[float]
    states: list[list[float]]
    actions: list[list[float]]
    rewards: list[float]
    state_labels: list[str]
    state_units: list[str]
    action_labels: list[str]
    action_units: list[str]


class SkippedPair(BaseModel):
    env_id: str
    controller_id: str
    reason: str


class CreatedBenchmark(BaseModel):
    suite_id: str
    benchmark_run_id: str
    status: Literal["queued"] = "queued"
    skipped: list[SkippedPair]


class BenchmarkRunOut(BaseModel):
    benchmark_run_id: str
    status: Literal["queued", "running", "done", "failed"]
    progress: float


class BenchmarkRunInfo(BaseModel):
    run_id: str
    seed: int
    status: Literal["queued", "running", "done", "failed"]


class BenchmarkCell(BaseModel):
    experiment_id: str
    env_id: str
    controller_id: str
    config_hash: str
    runs: list[BenchmarkRunInfo]


class BenchmarkSummaryItem(BaseModel):
    suite_id: str
    name: str
    created_at: datetime


class BenchmarkDetail(BenchmarkSummaryItem):
    spec: dict[str, Any]
    benchmark_runs: list[BenchmarkRunOut]
    experiments: list[BenchmarkCell]
    skipped: list[SkippedPair] = []


class SummaryOut(BaseModel):
    """Cross-seed summary of one metric for one cell; NaN values are sent as null."""

    env_id: str
    controller_id: str
    metric_name: str
    unit: str
    n_valid: int
    n_total: int
    n_censored: int
    mean: float | None
    std: float | None
    ci_low: float | None
    ci_high: float | None
    ci_method: str
    low_n: bool
