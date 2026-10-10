"""Experiment manager: validated config -> (train) -> rollout -> metrics, no I/O of its own.

Persistence and background execution wrap :func:`execute` (api / job runner); this module only
needs the registry, so it never names a concrete robot or controller.
"""

from __future__ import annotations

import importlib.metadata
import logging
import platform
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rcl.core.config import ExperimentConfig, RunRecord
from rcl.core.environment import BaseEnvironment
from rcl.core.types import EpisodeTrajectory, TrainConfig, TrainResult
from rcl.experiments.rollout import rollout
from rcl.metrics import MetricValue, aggregate, compute_metrics
from rcl.registry import Entry, Registry, get_registry

log = logging.getLogger(__name__)
_LIBS = ("numpy", "gymnasium", "pydantic", "stable-baselines3", "torch", "rcl")


@dataclass
class RunResult:
    """Everything one run produced. ``metrics`` are the per-run means over the evaluation
    episodes; ``episode_metrics[i]`` belongs to ``trajectories[i]``."""

    record: RunRecord
    train: TrainResult | None
    trajectories: list[EpisodeTrajectory]
    episode_metrics: list[list[MetricValue]]
    metrics: list[MetricValue]


def derive_seeds(seed: int) -> dict[str, int]:
    """``train`` and ``eval`` seeds spawned from ``seed`` (fixed order, ADR 0010)."""
    train, eval_ = (int(c.generate_state(1)[0]) for c in np.random.SeedSequence(seed).spawn(2))
    return {"train": train, "eval": eval_}


def _code_version() -> str:
    try:
        git = ["git", "-C", str(Path(__file__).parent)]
        sha = subprocess.run(
            [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            [*git, "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def collect_versions(env: Entry, controller: Entry) -> dict[str, str]:
    """Library, platform and registry-entry versions (ADR 0010)."""
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for lib in _LIBS:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            pass
    versions[f"env:{env.id}"] = env.metadata.version
    versions[f"controller:{controller.id}"] = controller.metadata.version
    if controller.metadata.requires_training:
        import torch

        versions.update(
            cuda=torch.version.cuda or "none",
            device="cpu",
            num_threads=str(torch.get_num_threads()),
        )
    return versions


def execute(
    config: ExperimentConfig,
    *,
    seed: int | None = None,
    registry: Registry | None = None,
    checkpoint_dir: Path | None = None,
    on_stage: Callable[[str], None] | None = None,
    progress: Callable[[int, int], bool | None] | None = None,
) -> RunResult:
    """Run one experiment end to end.

    ``seed`` overrides ``config.seed`` for this run. Raises ``UnknownIdError`` /
    ``ParamValidationError`` (registry) or ``ValueError`` (incompatible pair, missing
    training section, unsupported perturbations) before any work starts. ``on_stage`` is
    called with ``"training"`` / ``"evaluating"``; ``progress`` is forwarded to training.
    If ``checkpoint_dir`` is given a trained controller is saved there.
    """
    reg = registry or get_registry()
    env_entry = reg.get("environment", config.environment.id)
    ctl_entry = reg.get("controller", config.controller.id)
    if not reg.is_compatible(ctl_entry.id, env_entry.id):
        raise ValueError(f"controller {ctl_entry.id!r} is not compatible with {env_entry.id!r}")
    if config.perturbations:
        raise ValueError("perturbations are not supported before Phase 4")
    needs_training = ctl_entry.metadata.requires_training
    if needs_training and config.training is None:
        raise ValueError(f"controller {ctl_entry.id!r} requires a 'training' section")
    if needs_training and not env_entry.metadata.simulated:
        raise ValueError("training on a non-simulated environment is not allowed")

    env_params = env_entry.validate_params(config.environment.params)
    ctl_params = ctl_entry.validate_params(config.controller.params)
    resolved = config.model_copy(
        update={
            "environment": config.environment.model_copy(update={"params": env_params}),
            "controller": config.controller.model_copy(update={"params": ctl_params}),
        }
    )
    run_seed = config.seed if seed is None else seed
    seeds = derive_seeds(run_seed)
    record = RunRecord(
        seed=run_seed,
        derived_seeds=seeds,
        config=resolved,
        config_hash=resolved.config_hash(),
        versions=collect_versions(env_entry, ctl_entry),
        code_version=_code_version(),
    )

    def env_factory() -> BaseEnvironment:
        env: BaseEnvironment = env_entry.cls(**env_params)
        return env

    env = env_factory()
    try:
        controller = ctl_entry.cls()
        controller.configure(env.get_env_spec(), ctl_params, env.get_linearization())
        train_result = None
        if needs_training and config.training is not None:
            if on_stage:
                on_stage("training")
            train_result = controller.train(
                env_factory,
                TrainConfig(total_timesteps=config.training.total_timesteps, seed=seeds["train"]),
                progress,
            )
            if checkpoint_dir is not None:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                controller.save(checkpoint_dir)
        if on_stage:
            on_stage("evaluating")
        trajectories = rollout(env, controller, config.evaluation.episodes, seeds["eval"])
        task = env.get_task_spec()
    finally:
        env.close()
    episode_metrics = [compute_metrics(t, task) for t in trajectories]
    log.info("run done: %s on %s seed=%d", ctl_entry.id, env_entry.id, run_seed)
    return RunResult(
        record, train_result, trajectories, episode_metrics, aggregate(episode_metrics)
    )
