import math

import numpy as np
import pytest

from rcl.core.types import EpisodeTrajectory, TaskSpec
from rcl.metrics import aggregate
from rcl.metrics.robustness import (
    breakdown_magnitude,
    failure_count,
    mean_by_magnitude,
    recovery_time,
    relative_curve,
)

TASK = TaskSpec(
    setpoint=(0.0,),
    tolerance=(0.1,),
    error_state_indices=(0,),
    step_floor=1.0,
    horizon_steps=100,
    success_hold_steps=5,
)


def traj(errors, events=(), terminated=False, dt=0.1) -> EpisodeTrajectory:
    t = len(errors) - 1
    return EpisodeTrajectory(
        observations=np.zeros((t + 1, 1)),
        actions=np.zeros((t, 1)),
        rewards=np.ones(t),
        states=np.asarray(errors, dtype=float).reshape(-1, 1),
        dt=dt,
        terminated=terminated,
        truncated=not terminated,
        seed=0,
        success=False,
        perturbation_steps=tuple(events),
    )


def test_failure_count_counts_every_terminated_episode_and_mean_is_the_rate():
    assert failure_count(traj([0, 0, 0], terminated=True), TASK).value == 1.0
    assert failure_count(traj([0, 0, 0]), TASK).value == 0.0
    runs = [[failure_count(traj([0, 0], terminated=f), TASK)] for f in (True, False, False, False)]
    assert aggregate(runs)[0].value == pytest.approx(0.25)


def test_recovery_time_measures_from_the_event_to_the_final_return_to_the_band():
    # event at k=2, outside the tolerance (0.1, inclusive) at k=2,3, inside from k=4
    m = recovery_time(traj([0, 0, 0.5, 0.4, 0.1, 0.0, 0.0], events=[2]), TASK)
    assert (m.value, m.status) == (pytest.approx(0.2), "ok")


def test_recovery_time_is_zero_when_the_band_is_never_left():
    assert recovery_time(traj([0, 0.01, 0.0, 0.0], events=[1]), TASK).value == 0.0
    assert recovery_time(traj([0, 0.09, 0.1, 0.0], events=[1]), TASK).value == 0.0  # edge: <=


def test_recovery_time_is_censored_when_the_episode_fails_right_after_the_event():
    m = recovery_time(traj([0.0, 0.5], events=[0], terminated=True), TASK)
    assert (m.value, m.status) == (pytest.approx(10.0), "censored")  # T_max, event at 0


def test_recovery_time_uses_task_tolerance_per_component_and_wraps_angles():
    task = TaskSpec(
        setpoint=(0.0, 0.0), tolerance=(0.1, 1e9), error_state_indices=(0,), wrap_indices=(0,),
        step_floor=1.0, horizon_steps=100, success_hold_steps=5,
    )  # fmt: skip
    states = np.array([[0.0, 5.0], [0.5, 5.0], [2 * np.pi + 0.05, -7.0], [0.0, 9.0]])
    t = traj([0, 0, 0, 0], events=[1])
    t = EpisodeTrajectory(**{**t.__dict__, "states": states})
    assert recovery_time(t, task).value == pytest.approx(0.1)  # wrapped 0.05 is inside at k=2


def test_recovery_time_is_censored_when_it_never_returns_or_fails_afterwards():
    expected = 100 * 0.1 - 2 * 0.1
    for t in (traj([0, 0, 0.5, 0.5, 0.5], events=[2]), traj([0, 0, 0, 0.0], [2], terminated=True)):
        m = recovery_time(t, TASK)
        assert (m.value, m.status) == (pytest.approx(expected), "censored")


def test_recovery_time_uses_the_first_event_and_is_undefined_without_one():
    first = recovery_time(traj([0, 0.5, 0.0, 0.5, 0.0, 0.0], events=[3, 1]), TASK)
    assert first.value == pytest.approx(0.3)  # from k=1: last outside at k=3, so k_r = 4 (not 1)
    none = recovery_time(traj([0, 0.5, 0.0]), TASK)
    assert none.status == "undefined" and math.isnan(none.value)
    assert math.isnan(aggregate([[none]])[0].value)


def test_non_finite_is_undefined_and_zero_steps_raise():
    for fn in (failure_count, recovery_time):
        m = fn(traj([0, np.nan, 0], events=[1]), TASK)
        assert m.status == "undefined" and math.isnan(m.value)
        with pytest.raises(ValueError):
            fn(traj([0.0]), TASK)


def test_degradation_curve_helpers():
    curve = {0.0: 1.0, 0.1: 0.9, 0.2: 0.4, 0.3: 0.1}
    assert breakdown_magnitude(curve, 0.0) == 0.2  # first magnitude with value < 0.5
    assert relative_curve(curve, 0.0)[0.2] == pytest.approx(0.4)
    assert math.isnan(breakdown_magnitude({0.0: 1.0, 0.1: 0.8}, 0.0))  # never below half
    assert math.isnan(breakdown_magnitude({0.0: 0.0, 0.1: 0.0}, 0.0))  # nominal 0
    assert all(math.isnan(v) for v in relative_curve({0.0: 0.0, 0.1: 0.0}, 0.0).values())
    assert mean_by_magnitude({0.2: [0.0, 1.0], 0.0: [1.0, 1.0]}) == {0.0: 1.0, 0.2: 0.5}
