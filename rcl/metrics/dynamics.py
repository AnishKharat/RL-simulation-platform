"""Step-response metrics (settling/rise time, overshoot, steady-state error).

Definitions: rcl/metrics/DEFINITIONS.md "Phase 2 metrics". Pure functions over a trajectory.
"""

from __future__ import annotations

import numpy as np

from rcl.core.types import EpisodeTrajectory, TaskSpec
from rcl.metrics.metrics import MetricValue, _check, _undefined, wrap_angle


def _geometry(
    traj: EpisodeTrajectory, task: TaskSpec
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Return ``(e, S, w, w_ref)``: error norms e_k (T+1,), step size S, motion w_k = wrap(y_k -
    y_0) (T+1, d) and commanded step w_ref = wrap(y_ref - y_0) (d,)."""
    idx = list(task.error_state_indices)
    wrapped = [i for i, s in enumerate(idx) if s in task.wrap_indices]
    y, ref = traj.states[:, idx], np.asarray(task.y_ref)

    def wrap(d: np.ndarray) -> np.ndarray:
        d = np.array(d, dtype=np.float64)
        d[..., wrapped] = wrap_angle(d[..., wrapped])
        return d

    w_ref = wrap(ref - y[0])
    return (
        np.linalg.norm(wrap(y - ref), axis=1),
        float(np.linalg.norm(w_ref)),
        wrap(y - y[0]),
        w_ref,
    )


def _t_max(traj: EpisodeTrajectory, task: TaskSpec) -> float:
    return task.horizon_steps * traj.dt


def settling_time(traj: EpisodeTrajectory, task: TaskSpec) -> MetricValue:
    """Time after which the error stays within 2% of max(S, step_floor); companion ``settled``.

    Never settling (including episodes ended by failure) gives T_max with status censored.
    """
    unit = "s"
    if not _check(traj):
        return _undefined("settling_time", unit, ("settled",))
    e, s, _, _ = _geometry(traj, task)
    band = 0.02 * max(s, task.step_floor)
    outside = np.flatnonzero(e > band)
    settled = not traj.terminated and (len(outside) == 0 or outside[-1] < len(e) - 1)
    if not settled:
        return MetricValue(
            name="settling_time",
            value=_t_max(traj, task),
            unit=unit,
            status="censored",
            companions={"settled": (0.0, "bool")},
        )
    k_s = 0 if len(outside) == 0 else int(outside[-1]) + 1
    return MetricValue(
        name="settling_time",
        value=k_s * traj.dt,
        unit=unit,
        companions={"settled": (1.0, "bool")},
    )
