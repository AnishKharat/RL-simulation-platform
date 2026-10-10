"""Actuation metrics: energy proxy and control effort (DEFINITIONS.md, Phase 2)."""

from __future__ import annotations

import numpy as np

from rcl.core.types import EpisodeTrajectory, TaskSpec
from rcl.metrics.metrics import MetricValue, _check, _undefined


def _actions(traj: EpisodeTrajectory) -> np.ndarray:
    """Applied actions as (T, d), one row per step."""
    return traj.actions.reshape(traj.n_steps, -1).astype(np.float64)


def energy_usage(traj: EpisodeTrajectory, task: TaskSpec) -> MetricValue:
    """Actuation-energy proxy: sum_k ||u_k||_2^2 * dt over k < T (a proxy, not mechanical work)."""
    unit = "action^2*s"
    if not _check(traj):
        return _undefined("energy_usage", unit)
    value = float((_actions(traj) ** 2).sum() * traj.dt)
    return MetricValue(name="energy_usage", value=value, unit=unit)


def control_effort(traj: EpisodeTrajectory, task: TaskSpec) -> MetricValue:
    """sum_k ||u_k||_2 * dt; companion ``control_effort_rate`` = sum_{k>=1} ||u_k - u_{k-1}||_2
    (smoothness; 0 when T < 2)."""
    unit = "action*s"
    if not _check(traj):
        return _undefined("control_effort", unit, ("control_effort_rate",))
    u = _actions(traj)
    rate = float(np.linalg.norm(np.diff(u, axis=0), axis=1).sum())
    return MetricValue(
        name="control_effort",
        value=float(np.linalg.norm(u, axis=1).sum() * traj.dt),
        unit=unit,
        companions={"control_effort_rate": (rate, "action")},
    )
