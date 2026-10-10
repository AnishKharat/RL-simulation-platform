"""Task-outcome metrics: success rate and time to success (DEFINITIONS.md, Phase 3 additions).

They read the success verdict of the rollout (``EpisodeTrajectory.success``: the TaskSpec
tolerance held for ``success_hold_steps`` consecutive steps), so they are the headline metrics for
"did the controller achieve the task", unlike the reward sum.
"""

from __future__ import annotations

from rcl.core.types import EpisodeTrajectory, TaskSpec
from rcl.metrics.metrics import MetricValue, _check, _undefined


def success_rate(traj: EpisodeTrajectory, task: TaskSpec) -> MetricValue:
    """1.0 if the episode succeeded, else 0.0 per episode; the per-run mean is the success rate."""
    if not _check(traj):
        return _undefined("success_rate", "fraction")
    return MetricValue(name="success_rate", value=1.0 if traj.success else 0.0, unit="fraction")


def time_to_success(traj: EpisodeTrajectory, task: TaskSpec) -> MetricValue:
    """Time at which the final successful in-tolerance streak began: ``(T - h + 1) * dt`` with
    ``h = success_hold_steps``. An episode without success is censored at ``T_max``
    (``horizon_steps * dt``), the same convention as settling time."""
    unit = "s"
    if not _check(traj):
        return _undefined("time_to_success", unit)
    if not traj.success:
        return MetricValue(
            name="time_to_success",
            value=task.horizon_steps * traj.dt,
            unit=unit,
            status="censored",
        )
    start = max(traj.n_steps - task.success_hold_steps + 1, 0)
    return MetricValue(name="time_to_success", value=start * traj.dt, unit=unit)
