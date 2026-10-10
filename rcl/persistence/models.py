"""SQLAlchemy tables (Phase 1 subset of docs/architecture/data-model.md)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ExperimentRow(Base):
    __tablename__ = "experiment"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str]
    env_id: Mapped[str]
    controller_id: Mapped[str]
    config: Mapped[dict[str, Any]] = mapped_column(JSON)
    config_hash: Mapped[str]
    created_at: Mapped[datetime]


class RunRow(Base):
    __tablename__ = "run"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiment.id"), index=True)
    status: Mapped[str]  # queued running done failed
    progress: Mapped[float] = mapped_column(default=0.0)
    stage: Mapped[str | None]
    seed: Mapped[int]
    derived_seeds: Mapped[dict[str, int]] = mapped_column(JSON)
    versions: Mapped[dict[str, str] | None] = mapped_column(JSON)
    task_spec: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    code_version: Mapped[str | None]
    error: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error_stage: Mapped[str | None]
    queued_at: Mapped[datetime]
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]


class ArtifactRow(Base):
    __tablename__ = "artifact"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("run.id"), index=True)
    kind: Mapped[str]  # trajectory checkpoint curve
    uri: Mapped[str]
    sha256: Mapped[str]
    size_bytes: Mapped[int]
    mlflow_run_id: Mapped[str | None]


class EpisodeRow(Base):
    __tablename__ = "episode"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("run.id"), index=True)
    index: Mapped[int]
    seed: Mapped[int]
    terminated: Mapped[bool]
    truncated: Mapped[bool]
    success: Mapped[bool]
    n_steps: Mapped[int]
    trajectory_artifact_id: Mapped[str | None] = mapped_column(ForeignKey("artifact.id"))


class MetricResultRow(Base):
    __tablename__ = "metric_result"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("run.id"), index=True)
    episode_id: Mapped[str | None] = mapped_column(ForeignKey("episode.id"))
    metric_name: Mapped[str]
    value: Mapped[float | None]  # NULL when NaN (status = undefined)
    unit: Mapped[str]
    status: Mapped[str]  # ok censored undefined
