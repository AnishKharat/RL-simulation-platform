"""Step-response metrics (settling/rise time, overshoot, steady-state error).

Definitions: rcl/metrics/DEFINITIONS.md "Phase 2 metrics". Pure functions over a trajectory.
"""

from __future__ import annotations

import math

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
        return _undefined("settling_time", unit, {"settled": "bool"})
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


def _progress(traj: EpisodeTrajectory, task: TaskSpec) -> np.ndarray | None:
    """Normalised step progress p_k (0 at the start, 1 at the setpoint), or None if S < S_min."""
    _, s, w, w_ref = _geometry(traj, task)
    if s < task.step_floor:
        return None
    return np.asarray(w @ w_ref / s**2)


def _first_crossing(p: np.ndarray, level: float, dt: float) -> float | None:
    """Time of the first sample where p >= level, linearly interpolated; None if never."""
    hits = np.flatnonzero(p >= level)
    if len(hits) == 0:
        return None
    k = int(hits[0])
    if k == 0:
        return 0.0
    return float((k - 1 + (level - p[k - 1]) / (p[k] - p[k - 1])) * dt)


def rise_time(traj: EpisodeTrajectory, task: TaskSpec) -> MetricValue:
    """Time from 10% to 90% of the commanded step; undefined if the step is below the floor.

    If 90% is never reached the value is T_max - t_10 (T_max if 10% is not reached either),
    status censored.
    """
    unit = "s"
    if not _check(traj):
        return _undefined("rise_time", unit)
    p = _progress(traj, task)
    if p is None:
        return _undefined("rise_time", unit)
    t10, t90 = _first_crossing(p, 0.1, traj.dt), _first_crossing(p, 0.9, traj.dt)
    if t90 is not None and t10 is not None:
        return MetricValue(name="rise_time", value=t90 - t10, unit=unit)
    t_max = _t_max(traj, task)
    return MetricValue(name="rise_time", value=t_max - (t10 or 0.0), unit=unit, status="censored")


def overshoot(traj: EpisodeTrajectory, task: TaskSpec) -> MetricValue:
    """Percent of the step size by which the output exceeds the setpoint (0 if it never does);
    undefined if the step is below the floor."""
    if not _check(traj):
        return _undefined("overshoot", "%")
    p = _progress(traj, task)
    if p is None:
        return _undefined("overshoot", "%")
    return MetricValue(name="overshoot", value=100.0 * max(0.0, float(p.max()) - 1.0), unit="%")


def steady_state_error(traj: EpisodeTrajectory, task: TaskSpec) -> MetricValue:
    """Mean error over the last max(1, ceil(0.1 * (T + 1))) samples; censored if the episode
    ended in failure (the system never reached steady state)."""
    unit = task.output_unit or ""
    if not _check(traj):
        return _undefined("steady_state_error", unit)
    e, _, _, _ = _geometry(traj, task)
    window = max(1, math.ceil(0.1 * len(e)))
    return MetricValue(
        name="steady_state_error",
        value=float(e[-window:].mean()),
        unit=unit,
        status="censored" if traj.terminated else "ok",
    )
