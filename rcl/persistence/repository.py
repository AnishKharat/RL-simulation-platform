"""Repository over the Phase 1 tables. Takes and returns plain dicts / numpy arrays only (no
ORM objects leak, no import of metrics), so callers stay independent of the schema.

Trajectories live in ``<artifact_root>/<run_id>/episodes/<index>.npz`` (arrays only, no pickle).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import create_engine, event, inspect, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from rcl.core.config import ExperimentConfig
from rcl.persistence.models import (
    ArtifactRow,
    Base,
    BenchmarkRunRow,
    BenchmarkSuiteRow,
    EpisodeRow,
    ExperimentRow,
    MetricResultRow,
    RunRow,
)

_RUN_FIELDS = (
    "status stage progress seed derived_seeds versions task_spec code_version error "
    "error_stage queued_at started_at finished_at"
).split()


def _id() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)  # naive UTC: SQLite drops tzinfo anyway


def _run_dict(r: RunRow) -> dict[str, Any]:
    return {"run_id": r.id, "experiment_id": r.experiment_id} | {
        f: getattr(r, f) for f in _RUN_FIELDS
    }


class Repository:
    def __init__(self, url: str = "sqlite:///rcl.db", artifact_root: Path | str = "artifacts"):
        kw: dict[str, Any] = {"connect_args": {"check_same_thread": False}}
        if url in ("sqlite://", "sqlite:///:memory:"):
            kw["poolclass"] = StaticPool  # one shared connection for in-memory DBs
        self._engine = create_engine(url, **kw)
        if url.startswith("sqlite") and ":memory:" not in url and url != "sqlite://":
            with self._engine.connect() as c:  # WAL + wait on concurrent progress writes
                c.exec_driver_sql("PRAGMA journal_mode=WAL")
            event.listen(
                self._engine, "connect", lambda con, _: con.execute("PRAGMA busy_timeout=30000")
            )
        self._session = sessionmaker(self._engine, expire_on_commit=False)
        self.artifact_root = Path(artifact_root)
        Base.metadata.create_all(self._engine)  # Alembic arrives in Phase 5 (ADR 0008)
        self._add_missing_columns()

    def _add_missing_columns(self) -> None:
        """shortcut: create_all never alters tables, so a DB created by Phase 1 lacks
        experiment.suite_id; add it in place. Replaced by Alembic migrations in Phase 5."""
        with self._engine.begin() as c:
            cols = {i["name"] for i in inspect(c).get_columns("experiment")}
            if "suite_id" not in cols:
                c.exec_driver_sql("ALTER TABLE experiment ADD COLUMN suite_id VARCHAR(32)")

    # --- writes -----------------------------------------------------------------------------
    def create_experiment(
        self,
        config: ExperimentConfig,
        seed: int,
        derived_seeds: dict[str, int],
        suite_id: str | None = None,
    ) -> tuple[str, str]:
        """Insert an Experiment (config verbatim, expected resolved) and one queued Run."""
        exp_id = _id()
        with self._session.begin() as s:
            s.add(
                ExperimentRow(
                    id=exp_id,
                    name=config.name,
                    env_id=config.environment.id,
                    controller_id=config.controller.id,
                    config=config.model_dump(mode="json"),
                    config_hash=config.config_hash(),
                    created_at=_now(),
                    suite_id=suite_id,
                )
            )
        return exp_id, self.add_run(exp_id, seed, derived_seeds)

    def add_run(self, experiment_id: str, seed: int, derived_seeds: dict[str, int]) -> str:
        """Add one queued Run (another seed) to an existing Experiment (ADR 0009)."""
        run_id = _id()
        with self._session.begin() as s:
            s.add(
                RunRow(
                    id=run_id,
                    experiment_id=experiment_id,
                    status="queued",
                    progress=0.0,
                    seed=seed,
                    derived_seeds=derived_seeds,
                    queued_at=_now(),
                )
            )
        return run_id

    def update_run(self, run_id: str, **fields: Any) -> None:
        """Set ``status`` / ``stage`` / ``progress``; stamps started_at / finished_at."""
        with self._session.begin() as s:
            run = s.get(RunRow, run_id)
            if run is None:
                raise KeyError(run_id)
            for k, v in fields.items():
                setattr(run, k, v)
            if fields.get("status") == "running" and run.started_at is None:
                run.started_at = _now()
            if fields.get("status") in ("done", "failed"):
                run.finished_at = _now()

    def fail_run(self, run_id: str, error_type: str, message: str, stage: str | None) -> None:
        self.update_run(
            run_id,
            status="failed",
            error={"type": error_type, "message": message, "stage": stage},
            error_stage=stage,
        )

    def fail_orphans(self) -> int:
        """Mark runs left queued/running by a previous process as failed (server restart)."""
        with self._session.begin() as s:
            runs = s.scalars(select(RunRow).where(RunRow.status.in_(("queued", "running"))))
            n = 0
            for r in runs:
                r.status, r.finished_at = "failed", _now()
                r.error = {"type": "interrupted", "message": "server restarted", "stage": r.stage}
                r.error_stage = r.stage
                n += 1
        return n

    def save_results(
        self,
        run_id: str,
        *,
        versions: dict[str, str],
        code_version: str,
        task_spec: dict[str, Any],
        episodes: list[dict[str, Any]],
        metrics: list[dict[str, Any]],
        train_curve: list[tuple[int, float]],
    ) -> None:
        """Store a finished run and mark it ``done``.

        ``episodes[i]``: ``index, seed, terminated, truncated, success, n_steps, arrays`` (dict
        name -> ndarray, written as the trajectory artifact). ``metrics``: ``metric_name,
        episode_index (None = run level), value (None = NaN), unit, status``.
        """
        with self._session.begin() as s:
            run = s.get(RunRow, run_id)
            if run is None:
                raise KeyError(run_id)
            run.versions, run.code_version, run.task_spec = versions, code_version, task_spec
            episode_ids: dict[int, str] = {}
            for ep in episodes:
                art = self._write_npz(s, run_id, ep["index"], ep["arrays"])
                eid = episode_ids[ep["index"]] = _id()
                s.add(
                    EpisodeRow(
                        id=eid,
                        run_id=run_id,
                        trajectory_artifact_id=art,
                        **{
                            k: ep[k]
                            for k in ("index", "seed", "terminated", "truncated", "success")
                        },
                        n_steps=ep["n_steps"],
                    )
                )
            s.flush()  # episodes/artifacts before the rows that reference them
            for m in metrics:
                idx = m["episode_index"]
                s.add(
                    MetricResultRow(
                        id=_id(),
                        run_id=run_id,
                        episode_id=None if idx is None else episode_ids[idx],
                        metric_name=m["metric_name"],
                        value=m["value"],
                        unit=m["unit"],
                        status=m["status"],
                    )
                )
            self._write_json_artifact(s, run_id, "curve", [list(p) for p in train_curve])
            run.status, run.stage, run.progress, run.finished_at = "done", None, 1.0, _now()

    def _write_npz(self, s: Session, run_id: str, index: int, arrays: dict[str, Any]) -> str:
        path = self.artifact_root / run_id / "episodes" / f"{index}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return self._add_artifact(s, run_id, "trajectory", path)

    def _write_json_artifact(self, s: Session, run_id: str, kind: str, data: Any) -> str:
        path = self.artifact_root / run_id / f"{kind}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return self._add_artifact(s, run_id, kind, path)

    def _add_artifact(self, s: Session, run_id: str, kind: str, path: Path) -> str:
        blob = path.read_bytes()
        art = ArtifactRow(
            id=_id(),
            run_id=run_id,
            kind=kind,
            uri=str(path.resolve()),
            sha256=hashlib.sha256(blob).hexdigest(),
            size_bytes=len(blob),
        )
        s.add(art)
        return art.id

    def checkpoint_dir(self, run_id: str) -> Path:
        return self.artifact_root / run_id / "checkpoint"

    # --- reads ------------------------------------------------------------------------------
    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._session() as s:
            run = s.get(RunRow, run_id)
            return None if run is None else _run_dict(run)

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        with self._session() as s:
            e = s.get(ExperimentRow, experiment_id)
            if e is None:
                return None
            runs = s.scalars(
                select(RunRow).where(RunRow.experiment_id == e.id).order_by(RunRow.queued_at)
            )
            return {
                "experiment_id": e.id,
                "name": e.name,
                "env_id": e.env_id,
                "controller_id": e.controller_id,
                "config": e.config,
                "config_hash": e.config_hash,
                "created_at": e.created_at,
                "runs": [_run_dict(r) for r in runs],
            }

    def get_results(self, run_id: str) -> dict[str, Any]:
        """``episodes`` (by index), ``metrics`` (as in ``save_results``) and ``train_curve``."""
        with self._session() as s:
            eps = list(
                s.scalars(
                    select(EpisodeRow).where(EpisodeRow.run_id == run_id).order_by(EpisodeRow.index)
                )
            )
            by_id = {e.id: e.index for e in eps}
            rows = s.scalars(select(MetricResultRow).where(MetricResultRow.run_id == run_id))
            metrics = [
                {
                    "metric_name": m.metric_name,
                    "episode_index": None if m.episode_id is None else by_id[m.episode_id],
                    "value": m.value,
                    "unit": m.unit,
                    "status": m.status,
                }
                for m in rows
            ]
            curve = s.scalars(
                select(ArtifactRow).where(ArtifactRow.run_id == run_id, ArtifactRow.kind == "curve")
            ).first()
            return {
                "episodes": [
                    {
                        k: getattr(e, k)
                        for k in ("index", "seed", "terminated", "truncated", "success", "n_steps")
                    }
                    for e in eps
                ],
                "metrics": metrics,
                "train_curve": [] if curve is None else json.loads(Path(curve.uri).read_text()),
            }

    def load_trajectory(self, run_id: str, index: int) -> dict[str, np.ndarray] | None:
        """Arrays of one episode, or None if there is no such episode."""
        with self._session() as s:
            ep = s.scalars(
                select(EpisodeRow).where(EpisodeRow.run_id == run_id, EpisodeRow.index == index)
            ).first()
            art = ep and ep.trajectory_artifact_id and s.get(ArtifactRow, ep.trajectory_artifact_id)
            if not art:
                return None
            with np.load(art.uri, allow_pickle=False) as z:
                return {k: z[k] for k in z.files}

    # --- benchmarks -------------------------------------------------------------------------
    def create_suite(self, name: str, spec: dict[str, Any]) -> tuple[str, str]:
        """Insert a BenchmarkSuite (spec verbatim) and one queued BenchmarkRun."""
        suite_id, bench_id = _id(), _id()
        with self._session.begin() as s:
            s.add(BenchmarkSuiteRow(id=suite_id, name=name, spec=spec, created_at=_now()))
            s.flush()
            s.add(BenchmarkRunRow(id=bench_id, suite_id=suite_id, status="queued", progress=0.0))
        return suite_id, bench_id

    def update_benchmark_run(self, bench_id: str, **fields: Any) -> None:
        """Set ``status`` / ``progress`` / ``error``; stamps started_at / finished_at."""
        with self._session.begin() as s:
            row = s.get(BenchmarkRunRow, bench_id)
            if row is None:
                raise KeyError(bench_id)
            for k, v in fields.items():
                setattr(row, k, v)
            if fields.get("status") == "running" and row.started_at is None:
                row.started_at = _now()
            if fields.get("status") in ("done", "failed"):
                row.finished_at = _now()

    def get_benchmark_run(self, bench_id: str) -> dict[str, Any] | None:
        with self._session() as s:
            b = s.get(BenchmarkRunRow, bench_id)
            if b is None:
                return None
            return {
                "benchmark_run_id": b.id,
                "suite_id": b.suite_id,
                "status": b.status,
                "progress": b.progress,
                "error": b.error,
                "started_at": b.started_at,
                "finished_at": b.finished_at,
            }

    def get_suite(self, suite_id: str) -> dict[str, Any] | None:
        """Suite definition, its benchmark runs and, per experiment (cell), the runs by seed."""
        with self._session() as s:
            suite = s.get(BenchmarkSuiteRow, suite_id)
            if suite is None:
                return None
            exps = list(
                s.scalars(
                    select(ExperimentRow)
                    .where(ExperimentRow.suite_id == suite_id)
                    .order_by(ExperimentRow.created_at, ExperimentRow.id)
                )
            )
            bench = s.scalars(select(BenchmarkRunRow).where(BenchmarkRunRow.suite_id == suite_id))
            return {
                "suite_id": suite.id,
                "name": suite.name,
                "spec": suite.spec,
                "created_at": suite.created_at,
                "benchmark_runs": [
                    {"benchmark_run_id": b.id, "status": b.status, "progress": b.progress}
                    for b in bench
                ],
                "experiments": [
                    {
                        "experiment_id": e.id,
                        "env_id": e.env_id,
                        "controller_id": e.controller_id,
                        "config_hash": e.config_hash,
                        "runs": [
                            {"run_id": r.id, "seed": r.seed, "status": r.status}
                            for r in s.scalars(
                                select(RunRow)
                                .where(RunRow.experiment_id == e.id)
                                .order_by(RunRow.seed)
                            )
                        ],
                    }
                    for e in exps
                ],
            }

    def list_suites(self) -> list[dict[str, Any]]:
        with self._session() as s:
            rows = s.scalars(
                select(BenchmarkSuiteRow).order_by(BenchmarkSuiteRow.created_at.desc())
            )
            return [{"suite_id": r.id, "name": r.name, "created_at": r.created_at} for r in rows]

    def suite_metric_rows(self, suite_id: str) -> list[dict[str, Any]]:
        """Run-level metric values of every finished run in the suite, one row per
        (env, controller, seed, metric): the input of cross-seed analytics."""
        q = (
            select(
                ExperimentRow.env_id,
                ExperimentRow.controller_id,
                RunRow.seed,
                RunRow.id,
                MetricResultRow.metric_name,
                MetricResultRow.value,
                MetricResultRow.unit,
                MetricResultRow.status,
            )
            .join(RunRow, RunRow.experiment_id == ExperimentRow.id)
            .join(MetricResultRow, MetricResultRow.run_id == RunRow.id)
            .where(
                ExperimentRow.suite_id == suite_id,
                RunRow.status == "done",
                MetricResultRow.episode_id.is_(None),
            )
            .order_by(ExperimentRow.env_id, ExperimentRow.controller_id, RunRow.seed)
        )
        keys = (
            "env_id",
            "controller_id",
            "seed",
            "run_id",
            "metric_name",
            "value",
            "unit",
            "status",
        )
        with self._session() as s:
            return [dict(zip(keys, row, strict=True)) for row in s.execute(q)]

    # --- tracker links ----------------------------------------------------------------------
    def link_checkpoint(self, run_id: str, checkpoint_dir: Path, mlflow_run_id: str | None) -> None:
        """Record a run's checkpoint directory (pointer, never the weights) and, if the run was
        also logged to MLflow, the MLflow run id (ARTIFACT.mlflow_run_id)."""
        digest, size = hashlib.sha256(), 0
        for f in sorted(p for p in checkpoint_dir.rglob("*") if p.is_file()):
            blob = f.read_bytes()
            digest.update(f.relative_to(checkpoint_dir).as_posix().encode() + blob)
            size += len(blob)
        with self._session.begin() as s:
            s.add(
                ArtifactRow(
                    id=_id(),
                    run_id=run_id,
                    kind="checkpoint",
                    uri=str(checkpoint_dir.resolve()),
                    sha256=digest.hexdigest(),
                    size_bytes=size,
                    mlflow_run_id=mlflow_run_id,
                )
            )

    def get_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        with self._session() as s:
            rows = s.scalars(select(ArtifactRow).where(ArtifactRow.run_id == run_id))
            return [
                {
                    "kind": a.kind,
                    "uri": a.uri,
                    "sha256": a.sha256,
                    "size_bytes": a.size_bytes,
                    "mlflow_run_id": a.mlflow_run_id,
                }
                for a in rows
            ]
