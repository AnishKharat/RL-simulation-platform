import math

import numpy as np
import pytest

from rcl.core.types import EpisodeTrajectory, TaskSpec
from rcl.metrics.effort import control_effort, energy_usage

TASK = TaskSpec(
    setpoint=(0.0,),
    tolerance=(0.1,),
    error_state_indices=(0,),
    step_floor=0.05,
    horizon_steps=10,
)


def traj(actions) -> EpisodeTrajectory:
    a = np.asarray(actions, dtype=float)
    t = len(a)
    return EpisodeTrajectory(
        observations=np.zeros((t + 1, 1)),
        actions=a,
        rewards=np.zeros(t),
        states=np.zeros((t + 1, 1)),
        dt=0.5,
        terminated=False,
        truncated=True,
        seed=0,
        success=False,
    )


def test_energy_is_squared_norm_sum_times_dt():
    m = energy_usage(traj([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]]), TASK)  # 25 + 0 + 1
    assert m.value == pytest.approx(26 * 0.5) and m.unit == "action^2*s"
    assert energy_usage(traj([[1.0], [-1.0]]), TASK).value == pytest.approx(1.0)  # sign-free


def test_control_effort_and_smoothness_companion():
    m = control_effort(traj([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]]), TASK)
    assert m.value == pytest.approx((5.0 + 0.0 + 1.0) * 0.5)
    assert m.companions["control_effort_rate"] == (pytest.approx(5.0 + 1.0), "action")
    assert control_effort(traj([[2.0]]), TASK).companions["control_effort_rate"][0] == 0.0  # T < 2


def test_scalar_action_shape_and_non_finite():
    assert control_effort(
        traj(np.array([1.0, -1.0, 1.0]).reshape(3, 1)), TASK
    ).value == pytest.approx(1.5)
    bad = control_effort(traj([[1.0], [np.nan]]), TASK)
    assert (
        bad.status == "undefined"
        and math.isnan(bad.value)
        and "control_effort_rate" in bad.companions
    )
    assert energy_usage(traj([[np.inf]]), TASK).status == "undefined"
