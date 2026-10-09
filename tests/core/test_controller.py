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
    TrainConfig,
)


def test_cannot_instantiate_abstract() -> None:
    with pytest.raises(TypeError):
        BaseController()  # type: ignore[abstract]


class _Const(BaseController):
    metadata = ControllerMetadata(
        name="const",
        family=ControllerFamily.CLASSICAL,
        supported_action_spaces=(ActionSpaceKind.CONTINUOUS,),
        requires_training=False,
    )

    def configure(self, env_spec: EnvSpec, params: dict[str, Any]) -> None:
        pass

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
