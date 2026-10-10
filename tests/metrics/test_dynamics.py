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


def test_rise_time_interpolates_between_samples():
    from rcl.metrics.dynamics import rise_time

    # p = [0, .2, .6, .95, 1]: t10 = 0.05, t90 = 0.2 + (0.9-0.6)/0.35 * 0.1
    m = rise_time(traj([1.0, 0.8, 0.4, 0.05, 0.0]), task())
    assert m.value == pytest.approx(0.2 + 0.3 / 0.35 * 0.1 - 0.05) and m.status == "ok"


def test_rise_time_censored_when_90_percent_not_reached():
    from rcl.metrics.dynamics import rise_time

    m = rise_time(traj([1.0, 0.8, 0.6, 0.4]), task())  # p tops out at 0.6; t10 = 0.05
    assert m.status == "censored" and m.value == pytest.approx(1.0 - 0.05)
    none = rise_time(traj([1.0, 0.99, 0.98]), task())  # never even 10%
    assert none.status == "censored" and none.value == pytest.approx(1.0)


def test_rise_time_undefined_below_step_floor_and_for_non_finite():
    from rcl.metrics.dynamics import rise_time

    small = rise_time(traj([0.01, 0.0, 0.0]), task())
    assert small.status == "undefined" and math.isnan(small.value)
    assert rise_time(traj([1.0, np.nan]), task()).status == "undefined"


def test_rise_time_non_zero_setpoint_and_negative_step():
    from rcl.metrics.dynamics import rise_time

    up = rise_time(traj([0.0, 0.2, 1.0, 1.8, 2.0]), task(setpoint=2.0))  # p = theta / 2
    assert up.value == pytest.approx(0.2)
    down = rise_time(traj([2.0, 1.8, 1.0, 0.2, 0.0]), task(setpoint=0.0))
    assert down.value == pytest.approx(up.value)  # symmetric step, same progress curve


def test_overshoot_percent_of_step():
    from rcl.metrics.dynamics import overshoot

    assert overshoot(traj([1.0, 0.5, -0.2, 0.0]), task()).value == pytest.approx(20.0)
    # non-zero initial error and setpoint: p = theta / 2, peak 2.5 / 2
    assert overshoot(traj([0.0, 1.0, 2.5, 2.0]), task(setpoint=2.0)).value == pytest.approx(25.0)


def test_overshoot_zero_when_monotone_and_undefined_for_tiny_step():
    from rcl.metrics.dynamics import overshoot

    m = overshoot(traj([1.0, 0.5, 0.1, 0.0]), task())
    assert m.value == 0.0 and m.status == "ok" and m.unit == "%"
    assert overshoot(traj([0.01, 0.0]), task()).status == "undefined"
    assert overshoot(traj([1.0, np.inf]), task()).status == "undefined"


def test_steady_state_error_uses_last_ten_percent_window():
    from rcl.metrics.dynamics import steady_state_error

    theta = [1.0] * 9 + [0.2, 0.4]  # 11 samples -> window ceil(1.1) = 2
    m = steady_state_error(traj(theta), task())
    assert m.value == pytest.approx(0.3) and m.unit == "rad" and m.status == "ok"
    assert steady_state_error(traj([0.5, 0.25]), task()).value == pytest.approx(0.25)  # window 1


def test_steady_state_error_censored_on_failure_and_undefined_on_nan():
    from rcl.metrics.dynamics import steady_state_error

    assert steady_state_error(traj([0.0, 0.5], terminated=True), task()).status == "censored"
    assert steady_state_error(traj([0.0, np.nan]), task()).status == "undefined"
