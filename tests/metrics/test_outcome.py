import math

import numpy as np
import pytest

from rcl.core.types import EpisodeTrajectory, TaskSpec
from rcl.metrics import aggregate
from rcl.metrics.outcome import success_rate, time_to_success

TASK = TaskSpec(
    setpoint=(0.0,),
    tolerance=(0.1,),
    error_state_indices=(0,),
    step_floor=0.05,
    horizon_steps=100,
    success_hold_steps=5,
)


def traj(steps: int, success: bool, dt: float = 0.1, bad: bool = False) -> EpisodeTrajectory:
    states = np.zeros((steps + 1, 1))
    if bad:
        states[1, 0] = np.nan
    return EpisodeTrajectory(
        observations=np.zeros((steps + 1, 1)),
        actions=np.zeros((steps, 1)),
        rewards=np.ones(steps),
        states=states,
        dt=dt,
        terminated=False,
        truncated=not success,
        seed=0,
        success=success,
    )


def test_success_rate_is_one_or_zero_per_episode_and_the_mean_is_the_rate():
    assert success_rate(traj(20, True), TASK).value == 1.0
    assert success_rate(traj(100, False), TASK).value == 0.0
    runs = [[success_rate(traj(20, s), TASK)] for s in (True, False, False, False)]
    assert aggregate(runs)[0].value == pytest.approx(0.25)
    assert aggregate(runs)[0].unit == "fraction"


def test_time_to_success_is_the_start_of_the_final_in_tolerance_streak():
    m = time_to_success(traj(30, True, dt=0.1), TASK)  # streak = steps 26..30
    assert m.value == pytest.approx(2.6) and m.status == "ok" and m.unit == "s"
    assert time_to_success(traj(5, True), TASK).value == pytest.approx(0.1)  # streak = steps 1..5
    assert time_to_success(traj(3, True), TASK).value == 0.0  # shorter than the hold: clamped


def test_failure_is_censored_at_the_time_limit():
    m = time_to_success(traj(100, False, dt=0.1), TASK)
    assert (m.value, m.status) == (pytest.approx(10.0), "censored")
    early = time_to_success(traj(40, False), TASK)  # e.g. terminated early: still T_max
    assert early.value == pytest.approx(10.0) and early.status == "censored"


def test_per_run_time_to_success_counts_failures_as_censored():
    eps = [[time_to_success(traj(n, ok), TASK)] for n, ok in ((20, True), (100, False))]
    agg = aggregate(eps)[0]
    assert agg.status == "censored" and agg.value == pytest.approx((1.6 + 10.0) / 2)


def test_non_finite_is_undefined_and_zero_steps_raise():
    for fn in (success_rate, time_to_success):
        m = fn(traj(5, True, bad=True), TASK)
        assert m.status == "undefined" and math.isnan(m.value)
    empty = EpisodeTrajectory(
        observations=np.zeros((1, 1)),
        actions=np.zeros((0, 1)),
        rewards=np.zeros(0),
        states=np.zeros((1, 1)),
        dt=0.1,
        terminated=False,
        truncated=False,
        seed=0,
        success=False,
    )
    with pytest.raises(ValueError):
        success_rate(empty, TASK)
