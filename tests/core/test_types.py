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
        family=ControllerFamily.CLASSICAL,
        supported_action_spaces=(ActionSpaceKind.CONTINUOUS,),
    )
    assert ControllerMetadata.model_validate_json(md.model_dump_json()) == md
    with pytest.raises(ValidationError):
        ControllerMetadata(name="x", family=ControllerFamily.RL, supported_action_spaces=())
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
        error_state_index=0,
        y_ref=0.0,
        step_floor=0.05,
        time_limit_s=0.2,
        horizon_steps=10,
    )
    assert TaskSpec.model_validate_json(ts.model_dump_json()) == ts
