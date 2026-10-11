"""Robustness metrics: failure count, recovery time, degradation curves.

Definitions: rcl/metrics/DEFINITIONS.md "Phase 4 metrics". Pure functions over a trajectory.
Cross-seed intervals for the degradation curve are the analytics layer's job (Stage 3).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from rcl.core.types import EpisodeTrajectory, TaskSpec
from rcl.metrics.dynamics import _geometry, _t_max
from rcl.metrics.metrics import MetricValue, _check, _undefined


def failure_count(traj: EpisodeTrajectory, task: TaskSpec) -> MetricValue:
    """1.0 if the episode ended in failure (``terminated``), else 0.0 (unit ``fraction``).

    The per-run mean is the failure rate; times the episode count it is the failure count.
    """
    if not _check(traj):
        return _undefined("failure_count", "fraction")
    return MetricValue(name="failure_count", value=1.0 if traj.terminated else 0.0, unit="fraction")


def recovery_time(traj: EpisodeTrajectory, task: TaskSpec) -> MetricValue:
    """Time from the first discrete perturbation event until the error stays within the 2% band.

    Undefined without an event; censored (``T_max - k_p * dt``) if the episode failed after the
    event or never re-enters the band.
    """
    unit = "s"
    if not _check(traj):
        return _undefined("recovery_time", unit)
    if not traj.perturbation_steps:
        return _undefined("recovery_time", unit)
    k_p = min(traj.perturbation_steps)
    e, s, _, _ = _geometry(traj, task)
    band = 0.02 * max(s, task.step_floor)
    outside = np.flatnonzero(e[k_p:] > band)
    if traj.terminated or (len(outside) and outside[-1] == len(e) - k_p - 1):
        return MetricValue(
            name="recovery_time",
            value=_t_max(traj, task) - k_p * traj.dt,
            unit=unit,
            status="censored",
        )
    k_r = 0 if len(outside) == 0 else int(outside[-1]) + 1
    return MetricValue(name="recovery_time", value=k_r * traj.dt, unit=unit)


def breakdown_magnitude(curve: Mapping[float, float], nominal: float) -> float:
    """Smallest magnitude above ``nominal`` whose mean value is below half the nominal value.

    ``curve`` maps magnitude -> mean success_rate (the nominal magnitude must be a key). NaN if
    the curve never falls below half, or the nominal value is 0 (DEFINITIONS.md).
    """
    base = curve[nominal]
    if base <= 0:
        return math.nan
    return next((m for m in sorted(curve) if m > nominal and curve[m] < base / 2), math.nan)


def relative_curve(curve: Mapping[float, float], nominal: float) -> dict[float, float]:
    """``value / value_nominal`` per magnitude; all NaN if the nominal value is 0."""
    base = curve[nominal]
    return {m: (v / base if base else math.nan) for m, v in curve.items()}


def mean_by_magnitude(per_seed: Mapping[float, Sequence[float]]) -> dict[float, float]:
    """Cross-seed mean per magnitude, in ascending magnitude order."""
    return {m: float(np.mean(per_seed[m])) for m in sorted(per_seed)}
