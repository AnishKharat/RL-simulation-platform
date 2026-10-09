from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from rcl.core.controller import Action, BaseController
from rcl.core.types import (
    ActionSpaceKind,
    ControllerFamily,
    ControllerMetadata,
    EnvSpec,
    LinearModel,
    TrainConfig,
    TrainResult,
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
    res = _Const().train(lambda: None, TrainConfig(total_timesteps=1))  # type: ignore[arg-type,return-value]
    assert res.timesteps == 0


def test_evaluate_is_not_part_of_the_contract() -> None:
    assert not hasattr(BaseController, "evaluate")
    assert "evaluate" not in BaseController.__abstractmethods__


def test_train_takes_env_factory_and_save_gets_directory(tmp_path: Path) -> None:
    import inspect

    assert list(inspect.signature(BaseController.train).parameters) == [
        "self",
        "env_factory",
        "config",
        "progress",
    ]
    assert inspect.signature(BaseController.train).parameters["progress"].default is None
    calls: list[int] = []
    res = _Const().train(lambda: calls.append(1), TrainConfig(total_timesteps=1, n_envs=3))  # type: ignore[arg-type,return-value]
    assert res.curve == () and calls == []  # classical default never builds an env
    _Const().save(tmp_path)  # caller-created directory
    assert TrainConfig(total_timesteps=1).n_envs == 1
    with pytest.raises(ValidationError):
        TrainConfig(total_timesteps=1, n_envs=0)
    r = TrainResult(timesteps=2, curve=((1, 0.5), (2, 1.0)))
    assert TrainResult.model_validate_json(r.model_dump_json()) == r


def test_default_train_never_calls_progress() -> None:
    seen: list[tuple[int, int]] = []
    res = _Const().train(
        lambda: None,  # type: ignore[arg-type,return-value]
        TrainConfig(total_timesteps=1),
        lambda d, t: seen.append((d, t)),  # type: ignore[func-returns-value]
    )
    assert res.timesteps == 0 and seen == []


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
