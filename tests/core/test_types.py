from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from rcl.core.types import (
    ActionSpaceKind,
    ControllerFamily,
    ControllerMetadata,
    EnvSpec,
    EpisodeTrajectory,
    LinearModel,
    TaskSpec,
    TrainConfig,
)


def _traj(t: int = 3, **kw: Any) -> EpisodeTrajectory:
    args: dict[str, Any] = {
        "observations": np.zeros((t + 1, 4)),
        "actions": np.zeros((t, 1)),
        "rewards": np.zeros(t),
        "states": np.zeros((t + 1, 4)),
        "dt": 0.02,
        "terminated": False,
        "truncated": True,
        "seed": 1,
        "success": False,
    }
    args.update(kw)
    return EpisodeTrajectory(**args)


def test_trajectory_valid() -> None:
    tr = _traj()
    assert tr.n_steps == 3
    assert tr.time[-1] == pytest.approx(0.06)


@pytest.mark.parametrize(
    "kw",
    [
        {"actions": np.zeros((2, 1))},
        {"states": np.zeros((3, 4))},
        {"observations": np.zeros((5, 4))},
        {"dt": 0.0},
        {"terminated": True, "truncated": True},
        {"rewards": np.zeros((3, 1))},
    ],
)
def test_trajectory_invariants(kw: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _traj(**kw)


def test_linear_model_shapes() -> None:
    LinearModel(np.eye(2), np.ones((2, 1)), np.zeros(2), np.zeros(1), 0.1)
    with pytest.raises(ValueError):
        LinearModel(np.eye(3), np.ones((2, 1)), np.zeros(2), np.zeros(1), 0.1)


def test_requires_training_is_mandatory_controller_metadata() -> None:
    with pytest.raises(ValidationError):
        ControllerMetadata.model_validate(
            {"name": "x", "family": "rl", "supported_action_spaces": ["discrete"]}
        )
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"total_timesteps": 1, "params": {}})  # no hyperparameters here


def test_specs_forbid_extra_and_are_frozen() -> None:
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"total_timesteps": 10, "bogus": 1})
    with pytest.raises(ValidationError):
        TrainConfig(total_timesteps=0)
    cfg = TrainConfig(total_timesteps=1)
    with pytest.raises(ValidationError):
        cfg.seed = 3  # type: ignore[misc]


def test_metadata_and_specs_round_trip() -> None:
    md = ControllerMetadata(
        name="pid",
        version="0.1.0",
        family=ControllerFamily.CLASSICAL,
        supported_action_spaces=(ActionSpaceKind.CONTINUOUS,),
        requires_training=False,
    )
    assert ControllerMetadata.model_validate_json(md.model_dump_json()) == md
    with pytest.raises(ValidationError):
        ControllerMetadata(
            name="x",
            version="0.1.0",
            family=ControllerFamily.RL,
            supported_action_spaces=(),
            requires_training=True,
        )
    es = EnvSpec(
        env_id="cartpole",
        action_space_kind=ActionSpaceKind.DISCRETE,
        obs_shape=(4,),
        state_dim=4,
        dt=0.02,
        n_actions=2,
    )
    assert EnvSpec.model_validate_json(es.model_dump_json()) == es
    ts = TaskSpec(
        setpoint=(0.0,),
        tolerance=(0.1,),
        error_state_indices=(0,),
        step_floor=0.05,
        horizon_steps=10,
    )
    assert TaskSpec.model_validate_json(ts.model_dump_json()) == ts


_ES: dict[str, Any] = {"env_id": "e", "obs_shape": (2,), "state_dim": 2, "dt": 0.1}


@pytest.mark.parametrize(
    "kw",
    [
        {"action_space_kind": "discrete"},
        {"action_space_kind": "discrete", "n_actions": 2, "action_low": (0.0,)},
        {"action_space_kind": "continuous"},
        {
            "action_space_kind": "continuous",
            "n_actions": 2,
            "action_low": (0,),
            "action_high": (1,),
        },
        {"action_space_kind": "continuous", "action_low": (0.0,), "action_high": (1.0, 2.0)},
        {"action_space_kind": "continuous", "action_low": (1.0,), "action_high": (0.0,)},
        {
            "action_space_kind": "continuous",
            "action_shape": (2,),
            "action_low": (0.0,),
            "action_high": (1.0,),
        },
    ],
)
def test_env_spec_action_invariants(kw: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        EnvSpec(**_ES, **kw)


def test_env_spec_valid_continuous() -> None:
    EnvSpec(
        **_ES, action_space_kind="continuous", action_shape=(1,), action_low=(-1,), action_high=(1,)
    )


def test_task_spec_derived_y_ref_and_validation() -> None:
    kw: dict[str, Any] = {"tolerance": (0.1, 0.1), "step_floor": 0.05, "horizon_steps": 5}
    ts = TaskSpec(setpoint=(0.0, 2.5), error_state_indices=(1,), **kw)
    assert ts.y_ref == (2.5,) and ts.wrap_indices == ()
    multi = TaskSpec(
        setpoint=(0.0, 2.5, 1.0),
        tolerance=(0.1,) * 3,
        error_state_indices=(2, 0),
        wrap_indices=(0,),
        step_floor=0.05,
        horizon_steps=5,
    )
    assert multi.y_ref == (1.0, 0.0)
    bad: list[dict[str, Any]] = [
        {"error_state_indices": (2,)},  # out of range
        {"error_state_indices": ()},  # min 1
        {"error_state_indices": (0, 0)},  # duplicate
        {"error_state_indices": (0,), "wrap_indices": (1,)},  # not a subset
        {"error_state_indices": (0,), "y_ref": (0.0,)},  # derived, not a field
    ]
    for b in bad:
        with pytest.raises(ValidationError):
            TaskSpec(setpoint=(0.0, 2.5), **b, **kw)
    with pytest.raises(ValidationError):
        TaskSpec(setpoint=(0.0,), error_state_indices=(0,), **kw)  # tolerance length


def test_success_episode_has_neither_flag() -> None:
    tr = _traj(terminated=False, truncated=False, success=True)
    assert tr.success and not (tr.terminated or tr.truncated)
