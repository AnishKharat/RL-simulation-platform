"""Benchmark runner: controllers x environments x seeds under one fairness protocol.

Fairness (project_context 6.8): every cell uses the SAME seed list, evaluation episode count,
training budget and environment definition (params, hence time limit). For a run seed the
evaluation seeds are derived from the seed alone (ADR 0010), so all controllers see identical
initial states. Incompatible (controller, environment) pairs are skipped and reported, never
silently dropped.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rcl.core.config import (
    ComponentConfig,
    EvaluationSection,
    ExperimentConfig,
    TrainingSection,
)
from rcl.experiments.manager import ConfigError, RunResult, execute, resolve_config
from rcl.registry import Registry, get_registry

log = logging.getLogger(__name__)


class BenchmarkSpec(BaseModel):
    """The grid. ``training`` applies to every controller that requires training."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    environments: list[ComponentConfig] = Field(min_length=1)
    controllers: list[ComponentConfig] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    episodes: int = Field(gt=0)
    training: TrainingSection | None = None

    @field_validator("seeds")
    @classmethod
    def _seeds_ok(cls, v: list[int]) -> list[int]:
        if len(set(v)) != len(v) or min(v) < 0:
            raise ValueError("seeds must be unique and >= 0")
        return v

    @field_validator("environments", "controllers")
    @classmethod
    def _ids_unique(cls, v: list[ComponentConfig]) -> list[ComponentConfig]:
        ids = [c.id for c in v]
        if len(set(ids)) != len(ids):
            raise ValueError("component ids must be unique within a benchmark")
        return v

    def experiment(self, env: ComponentConfig, ctl: ComponentConfig, seed: int) -> ExperimentConfig:
        """The ExperimentConfig of one (env, controller, seed) run."""
        return ExperimentConfig(
            name=f"{self.name}:{env.id}:{ctl.id}",
            seed=seed,
            environment=env,
            controller=ctl,
            training=self.training,
            evaluation=EvaluationSection(episodes=self.episodes),
        )


@dataclass
class Cell:
    """All seeds of one (environment, controller) pair, in ``spec.seeds`` order."""

    env_id: str
    controller_id: str
    config_hash: str
    runs: list[RunResult]


@dataclass
class Skipped:
    env_id: str
    controller_id: str
    reason: str


@dataclass
class BenchmarkResult:
    spec: BenchmarkSpec
    cells: list[Cell]
    skipped: list[Skipped]


def plan(
    spec: BenchmarkSpec, registry: Registry
) -> tuple[list[tuple[ComponentConfig, ComponentConfig]], list[Skipped]]:
    """Validate the whole grid up front. Returns runnable (env, controller) pairs and the
    skipped incompatible pairs; raises ``ConfigError`` for anything else that is wrong."""
    pairs: list[tuple[ComponentConfig, ComponentConfig]] = []
    skipped: list[Skipped] = []
    errors: list[dict[str, object]] = []
    for i, env in enumerate(spec.environments):
        for j, ctl in enumerate(spec.controllers):
            try:
                resolve_config(spec.experiment(env, ctl, spec.seeds[0]), registry)
            except ConfigError as exc:
                if all(e["type"] == "incompatible" for e in exc.errors):
                    skipped.append(Skipped(env.id, ctl.id, exc.errors[0]["msg"]))
                    continue
                for e in exc.errors:  # re-address the error to the benchmark body
                    loc = list(e["loc"])
                    if loc[0] == "environment":
                        loc = ["environments", i, *loc[1:]]
                    elif loc[0] == "controller":
                        loc = ["controllers", j, *loc[1:]]
                    item = {**e, "loc": loc, "cell": [env.id, ctl.id]}
                    if not any(x["loc"] == loc and x["type"] == e["type"] for x in errors):
                        errors.append(item)
                continue
            pairs.append((env, ctl))
    if errors:
        raise ConfigError(errors)
    if not pairs:
        raise ConfigError(
            [
                {
                    "loc": ["controllers"],
                    "msg": "no compatible (environment, controller) pair",
                    "type": "empty",
                }
            ]
        )
    return pairs, skipped


def run_benchmark(
    spec: BenchmarkSpec,
    *,
    registry: Registry | None = None,
    progress: Callable[[int, int], bool | None] | None = None,
) -> BenchmarkResult:
    """Run every compatible cell for every seed, sequentially.

    ``progress(done_runs, total_runs)`` is called after each run; returning True stops early
    (the result then holds only the completed runs of the last cell).
    """
    reg = registry or get_registry()
    pairs, skipped = plan(spec, reg)
    total, done, cells = len(pairs) * len(spec.seeds), 0, []
    for env, ctl in pairs:
        runs: list[RunResult] = []
        for seed in spec.seeds:
            runs.append(execute(spec.experiment(env, ctl, seed), seed=seed, registry=reg))
            done += 1
            log.info(
                "benchmark %s: run %d/%d (%s on %s, seed %d)",
                spec.name,
                done,
                total,
                ctl.id,
                env.id,
                seed,
            )
            if progress is not None and progress(done, total):
                cells.append(Cell(env.id, ctl.id, runs[0].record.config_hash, runs))
                return BenchmarkResult(spec, cells, skipped)
        cells.append(Cell(env.id, ctl.id, runs[0].record.config_hash, runs))
    return BenchmarkResult(spec, cells, skipped)
