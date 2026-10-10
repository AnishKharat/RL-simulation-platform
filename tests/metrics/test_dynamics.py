import math

import numpy as np
import pytest

from rcl.core.types import EpisodeTrajectory, TaskSpec
from rcl.metrics.dynamics import settling_time

DT = 0.1


def task(setpoint=0.0, floor=0.05) -> TaskSpec:
    return TaskSpec(
        setpoint=(setpoint, 0.0),
        tolerance=(0.1, 1e9),
        error_state_indices=(0,),
        wrap_indices=(0,),
        output_unit="rad",
        step_floor=floor,
        horizon_steps=10,
    )


def traj(theta, terminated=False, actions=None) -> EpisodeTrajectory:
    theta = np.asarray(theta, dtype=float)
    t = len(theta) - 1
    return EpisodeTrajectory(
        observations=np.zeros((t + 1, 2)),
        actions=np.zeros((t, 1)) if actions is None else np.asarray(actions, dtype=float),
        rewards=np.ones(t),
        states=np.stack([theta, np.zeros_like(theta)], axis=1),
        dt=DT,
        terminated=terminated,
        truncated=not terminated,
        seed=0,
        success=False,
    )


def test_settling_time_finds_last_exit_from_band():
    m = settling_time(traj(0.5 ** np.arange(21)), task())  # S = 1, band 0.02, 0.5**6 < 0.02
    assert m.value == pytest.approx(0.6) and m.status == "ok" and m.companions["settled"][0] == 1.0


def test_settling_time_re_exit_resets_the_clock():
    theta = [1.0, 0.0, 0.0, 0.5, 0.0, 0.0]  # leaves the band again at k=3
    assert settling_time(traj(theta), task()).value == pytest.approx(0.4)


def test_never_settles_or_fails_is_censored_at_t_max():
    for t in (traj([1.0] * 6), traj([1.0, 0.0, 0.0], terminated=True)):
        m = settling_time(t, task())
        assert (m.value, m.status, m.companions["settled"][0]) == (
            pytest.approx(1.0),
            "censored",
            0.0,
        )


def test_inside_band_from_the_start_is_zero_and_floor_applies():
    m = settling_time(traj([0.0005, 0.0004, 0.0]), task())  # S tiny -> band = 0.02 * floor
    assert m.value == 0.0 and m.status == "ok"


def test_non_zero_setpoint_and_wrapping():
    m = settling_time(traj([0.0, 1.0, 1.99, 2.0, 2.0]), task(setpoint=2.0))
    assert m.value == pytest.approx(0.2)
    wrapped = settling_time(traj([2 * math.pi - 0.0001, 0.0]), task())  # 2*pi-0.0001 is ~0 rad
    assert wrapped.value == 0.0


def test_non_finite_undefined():
    m = settling_time(traj([1.0, np.nan, 0.0]), task())
    assert m.status == "undefined" and math.isnan(m.value) and "settled" in m.companions
