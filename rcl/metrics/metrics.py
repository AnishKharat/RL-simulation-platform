"""Phase 1 metrics: ``return`` and ``angle_error``. Pure functions, no I/O, no RNG.

See rcl/metrics/DEFINITIONS.md for definitions, indexing and the NaN/edge-case policy.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from rcl.core.types import EpisodeTrajectory, TaskSpec


class MetricValue(BaseModel):
    """One metric result. ``value`` is NaN iff ``status == "undefined"``. ``companions`` maps a
    secondary value's name to ``(value, unit)`` and shares the primary's status."""

    model_config = ConfigDict(frozen=True)

    name: str
    value: float
    unit: str
    status: Literal["ok", "censored", "undefined"] = "ok"
    companions: dict[str, tuple[float, str]] = Field(default_factory=dict)


class MetricInfo(BaseModel):
    """Catalogue entry served by ``GET /metrics``. ``unit`` None = the task's output unit."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    unit: str | None
    higher_is_better: bool
    companions: list[dict[str, str | None]] = Field(default_factory=list)


def _check(traj: EpisodeTrajectory) -> bool:
    """Raise on T = 0 (programming error); return False if any array is non-finite."""
    if traj.n_steps == 0:
        raise ValueError("trajectory has no steps (T = 0)")
    return all(np.isfinite(a).all() for a in (traj.states, traj.actions, traj.rewards))


def _undefined(name: str, unit: str, companions: tuple[str, ...] = ()) -> MetricValue:
    nan = math.nan
    return MetricValue(
        name=name,
        value=nan,
        unit=unit,
        status="undefined",
        companions={c: (nan, unit) for c in companions},
    )


def episode_return(traj: EpisodeTrajectory, task: TaskSpec) -> MetricValue:
    """Undiscounted reward sum over the T transitions."""
    if not _check(traj):
        return _undefined("return", "reward")
    return MetricValue(name="return", value=float(traj.rewards.sum()), unit="reward")


def wrap_angle(d: np.ndarray) -> np.ndarray:
    """Map angles to (-pi, pi]."""
    return np.pi - ((np.pi - d) % (2 * np.pi))


def angle_error(traj: EpisodeTrajectory, task: TaskSpec) -> MetricValue:
    """RMS of e_k = ||wrap(y(x_k) - y_ref)||_2 over k = 0..T; companion ``max_abs_angle_error``."""
    unit = task.output_unit or ""
    if not _check(traj):
        return _undefined("angle_error", unit, ("max_abs_angle_error",))
    idx = list(task.error_state_indices)
    d = traj.states[:, idx] - np.asarray(task.y_ref)
    wrapped = [i for i, s in enumerate(idx) if s in task.wrap_indices]
    d[:, wrapped] = wrap_angle(d[:, wrapped])
    e = np.linalg.norm(d, axis=1)
    return MetricValue(
        name="angle_error",
        value=float(np.sqrt(np.mean(e**2))),
        unit=unit,
        companions={"max_abs_angle_error": (float(e.max()), unit)},
    )


# Phase 1 metric list, chosen by the experiment manager (not by users or the frontend).
METRICS: dict[str, Callable[[EpisodeTrajectory, TaskSpec], MetricValue]] = {
    "return": episode_return,
    "angle_error": angle_error,
}


def compute_metrics(traj: EpisodeTrajectory, task: TaskSpec) -> list[MetricValue]:
    """All Phase 1 metrics for one episode."""
    return [fn(traj, task) for fn in METRICS.values()]


def aggregate(per_episode: list[list[MetricValue]]) -> list[MetricValue]:
    """Per-run value: mean of each metric (and companion) over the evaluation episodes.

    ``undefined`` episodes are excluded; the result is ``undefined`` if none is valid and
    ``censored`` if any included episode is (DEFINITIONS.md, "Aggregation across seeds").
    """
    out: list[MetricValue] = []
    for k in range(len(per_episode[0])):
        ms = [ep[k] for ep in per_episode if ep[k].status != "undefined"]
        first = per_episode[0][k]
        if not ms:
            out.append(_undefined(first.name, first.unit, tuple(first.companions)))
            continue
        out.append(
            MetricValue(
                name=first.name,
                value=float(np.mean([m.value for m in ms])),
                unit=first.unit,
                status="censored" if any(m.status == "censored" for m in ms) else "ok",
                companions={
                    c: (float(np.mean([m.companions[c][0] for m in ms])), u)
                    for c, (_, u) in first.companions.items()
                },
            )
        )
    return out


METRIC_INFO: list[MetricInfo] = [
    MetricInfo(
        name="return",
        description="Undiscounted sum of rewards over the episode.",
        unit="reward",
        higher_is_better=True,
    ),
    MetricInfo(
        name="angle_error",
        description="RMS distance of the controlled output from its reference.",
        unit=None,
        higher_is_better=False,
        companions=[{"name": "max_abs_angle_error", "unit": None}],
    ),
]
