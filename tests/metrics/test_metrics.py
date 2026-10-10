import math

import numpy as np
import pytest

from rcl.core.types import EpisodeTrajectory, TaskSpec
from rcl.metrics import angle_error, compute_metrics, episode_return

TASK = TaskSpec(
    setpoint=(0.0, 0.0),
    tolerance=(0.1, 1e9),
    error_state_indices=(0,),
    wrap_indices=(0,),
    output_unit="rad",
    step_floor=0.05,
    horizon_steps=10,
)


def traj(theta, rewards=None, actions=None) -> EpisodeTrajectory:
    theta = np.asarray(theta, dtype=float)
    t = len(theta) - 1
    return EpisodeTrajectory(
        observations=np.zeros((t + 1, 2)),
        actions=np.zeros((t, 1)) if actions is None else actions,
        rewards=np.ones(t) if rewards is None else np.asarray(rewards, dtype=float),
        states=np.stack([theta, np.zeros_like(theta)], axis=1),
        dt=0.1,
        terminated=False,
        truncated=True,
        seed=0,
        success=False,
    )


def test_return_is_reward_sum():
    m = episode_return(traj([0, 0, 0, 0], rewards=[1, 2, 3]), TASK)
    assert (m.name, m.value, m.unit, m.status) == ("return", 6.0, "reward", "ok")


def test_angle_error_rms_and_max():
    m = angle_error(traj([0.3, -0.4, 0.0]), TASK)  # k = 0..T, T+1 samples
    assert m.value == pytest.approx(math.sqrt((0.09 + 0.16) / 3))
    assert m.companions["max_abs_angle_error"] == (pytest.approx(0.4), "rad")
    assert m.unit == "rad"


def test_angle_error_wraps_to_pi():
    m = angle_error(traj([2 * math.pi + 0.1, -2 * math.pi - 0.1]), TASK)
    assert m.value == pytest.approx(0.1)


def test_error_uses_euclidean_norm_over_components():
    task = TASK.model_copy(
        update={"error_state_indices": (0, 1), "wrap_indices": (0,), "setpoint": (0.0, 0.0)}
    )
    t = traj([0.3, 0.3])
    t.states[:, 1] = 0.4
    assert angle_error(t, task).value == pytest.approx(0.5)


def test_non_finite_is_undefined_not_raised():
    bad = traj([0.1, np.nan, 0.0])
    for m in (episode_return(bad, TASK), angle_error(bad, TASK)):
        assert m.status == "undefined" and math.isnan(m.value)
    assert math.isnan(angle_error(bad, TASK).companions["max_abs_angle_error"][0])


def test_zero_steps_raises():
    empty = EpisodeTrajectory(
        observations=np.zeros((1, 2)),
        actions=np.zeros((0, 1)),
        rewards=np.zeros(0),
        states=np.zeros((1, 2)),
        dt=0.1,
        terminated=False,
        truncated=False,
        seed=0,
        success=False,
    )
    with pytest.raises(ValueError):
        angle_error(empty, TASK)


def test_compute_metrics_lists_phase1_names():
    assert [m.name for m in compute_metrics(traj([0.1, 0.0]), TASK)] == ["return", "angle_error"]


def test_aggregate_means_and_excludes_undefined():
    from rcl.metrics import aggregate

    eps = [compute_metrics(traj(th), TASK) for th in ([0.3, 0.0], [0.1, 0.0], [0.1, np.nan])]
    agg = {m.name: m for m in aggregate(eps)}
    assert agg["return"].value == pytest.approx(1.0) and agg["return"].status == "ok"
    e = [angle_error(traj(th), TASK) for th in ([0.3, 0.0], [0.1, 0.0])]
    assert agg["angle_error"].value == pytest.approx(np.mean([x.value for x in e]))
    assert agg["angle_error"].companions["max_abs_angle_error"][0] == pytest.approx(0.2)
    allbad = aggregate([compute_metrics(traj([0.1, np.nan]), TASK)])
    assert all(m.status == "undefined" for m in allbad)
