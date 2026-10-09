from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rcl.core.controller import Action, BaseController
from rcl.core.types import (
    ActionSpaceKind,
    ControllerFamily,
    ControllerMetadata,
    EnvSpec,
    LinearModel,
    TrainConfig,
)


def test_cannot_instantiate_abstract() -> None:
    with pytest.raises(TypeError):
        BaseController()  # type: ignore[abstract]


class _Const(BaseController):
    metadata = ControllerMetadata(
        name="const",
        version="0.1.0",
        family=ControllerFamily.CLASSICAL,
        supported_action_spaces=(ActionSpaceKind.CONTINUOUS,),
        requires_training=False,
    )

    def configure(
        self,
        env_spec: EnvSpec,
        params: dict[str, Any],
        linear_model: LinearModel | None = None,
    ) -> None:
        self.model = linear_model

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Action:
        return np.zeros(1)

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass


def test_default_train_is_noop() -> None:
    res = _Const().train(None, TrainConfig(total_timesteps=1))  # type: ignore[arg-type]
    assert res.timesteps == 0


def test_evaluate_is_not_part_of_the_contract() -> None:
    assert not hasattr(BaseController, "evaluate")
    assert "evaluate" not in BaseController.__abstractmethods__


def test_default_reset_is_noop() -> None:
    assert _Const().reset() is None


def test_configure_takes_optional_linear_model() -> None:
    import inspect

    p = inspect.signature(BaseController.configure).parameters
    assert p["linear_model"].default is None
    c = _Const()
    es = EnvSpec(
        env_id="e",
        action_space_kind=ActionSpaceKind.CONTINUOUS,
        obs_shape=(2,),
        state_dim=2,
        dt=0.1,
        action_low=(-1.0,),
        action_high=(1.0,),
    )
    lm = LinearModel(np.eye(2), np.ones((2, 1)), np.zeros(2), np.zeros(1), 0.1)
    c.configure(es, {}, lm)
    assert c.model is lm
