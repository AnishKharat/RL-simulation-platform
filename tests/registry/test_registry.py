import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel

from rcl.core.controller import Action, BaseController
from rcl.core.environment import BaseEnvironment
from rcl.core.types import (
    ActionSpaceKind,
    ControllerFamily,
    ControllerMetadata,
    EnvMetadata,
    EnvSpec,
    TaskSpec,
)
from rcl.registry import (
    DuplicateIdError,
    Registry,
    RegistryError,
    UnknownIdError,
    discover,
    get_registry,
    register_controller,
    register_environment,
    use_registry,
)


class Params(BaseModel):
    kp: float = 1.0


def _env_meta(kind: ActionSpaceKind = ActionSpaceKind.CONTINUOUS) -> EnvMetadata:
    return EnvMetadata(name="e", dof=1, action_space_kind=kind, physics_backend="none")


def _ctl_meta(*kinds: ActionSpaceKind) -> ControllerMetadata:
    return ControllerMetadata(
        name="c",
        family=ControllerFamily.CLASSICAL,
        supported_action_spaces=kinds or (ActionSpaceKind.CONTINUOUS,),
        requires_training=False,
    )


def _env_cls(kind: ActionSpaceKind = ActionSpaceKind.CONTINUOUS) -> type:
    return type("E", (), {"env_metadata": _env_meta(kind)})


def _named_env(qualname: str) -> type:
    c = _env_cls()
    c.__module__, c.__qualname__ = "fx_mod", qualname
    return c


def _ctl_cls(*kinds: ActionSpaceKind) -> type:
    return type("C", (), {"metadata": _ctl_meta(*kinds)})


@pytest.fixture
def reg():
    with use_registry(Registry()) as r:
        yield r


def test_register_lookup_and_info(reg: Registry) -> None:
    @register_controller("pid", params=Params)
    class PID:
        metadata = _ctl_meta()

    entry = reg.get("controller", "pid")
    assert entry.cls is PID
    assert entry.info()["metadata"]["requires_training"] is False
    info = entry.info()
    assert info["params_schema"]["properties"]["kp"]["default"] == 1.0
    assert entry.validate_params({"kp": 2}) == {"kp": 2.0}
    assert get_registry() is reg


def test_duplicate_id(reg: Registry) -> None:
    register_environment("e")(_named_env("A"))
    with pytest.raises(DuplicateIdError):
        register_environment("e")(_named_env("B"))
    register_controller("e")(_ctl_cls())  # separate namespace


def test_same_class_twice_is_noop(reg: Registry) -> None:
    c = _env_cls()
    register_environment("e")(c)
    register_environment("e")(c)
    assert len(reg.entries("environment")) == 1


def test_reloaded_class_is_noop_but_different_class_conflicts(reg: Registry) -> None:
    register_environment("e")(_named_env("E"))
    register_environment("e")(_named_env("E"))  # new class object, same module+qualname
    assert len(reg.entries("environment")) == 1
    with pytest.raises(DuplicateIdError):
        register_environment("e")(_named_env("Other"))


@pytest.mark.parametrize("bad_id", ["Bad", "1x", "a-b", ""])
def test_bad_id(reg: Registry, bad_id: str) -> None:
    with pytest.raises(RegistryError):
        register_environment(bad_id)(_env_cls())


def test_metadata_attribute_name_and_type_checked(reg: Registry) -> None:
    with pytest.raises(RegistryError):
        register_environment("e")(type("NoMeta", (), {}))
    # environments use `env_metadata`, controllers use `metadata`; a dict is not metadata
    with pytest.raises(RegistryError):
        register_environment("e")(type("E", (), {"metadata": _env_meta()}))
    with pytest.raises(RegistryError):
        register_environment("e")(type("E", (), {"env_metadata": {"name": "e"}}))
    with pytest.raises(RegistryError):
        register_controller("c")(type("C", (), {"metadata": _env_meta()}))  # wrong type
    with pytest.raises(RegistryError):
        register_controller("c")(type("C", (), {"env_metadata": _ctl_meta()}))
    bad: Any = int
    with pytest.raises(RegistryError):
        register_environment("e", params=bad)(_env_cls())


def test_real_base_environment_subclass_registers_via_env_metadata(reg: Registry) -> None:
    """Integration: gym's own `metadata` dict must not shadow `env_metadata` (ADR 0012)."""

    @register_environment("real")
    class Real(BaseEnvironment):
        env_metadata = _env_meta()
        metadata = {"render_modes": []}  # noqa: RUF012 - gymnasium convention

        def get_env_spec(self) -> EnvSpec:
            raise NotImplementedError

        def get_state(self) -> np.ndarray:
            raise NotImplementedError

        def get_task_spec(self) -> TaskSpec:
            raise NotImplementedError

    entry = reg.get("environment", "real")
    assert entry.cls is Real
    assert entry.metadata is Real.env_metadata
    assert entry.info()["metadata"]["action_space_kind"] == "continuous"


def test_real_base_controller_subclass_registers_via_metadata(reg: Registry) -> None:
    @register_controller("realc")
    class RealC(BaseController):
        metadata = _ctl_meta()

        def configure(self, env_spec: EnvSpec, params: dict[str, Any]) -> None: ...
        def predict(self, obs: np.ndarray, deterministic: bool = True) -> Action:
            return np.zeros(1)

        def save(self, path: Path) -> None: ...
        def load(self, path: Path) -> None: ...

    assert reg.get("controller", "realc").metadata is RealC.metadata


def test_unknown_id(reg: Registry) -> None:
    with pytest.raises(UnknownIdError):
        reg.get("environment", "nope")


def test_compatibility_filter(reg: Registry) -> None:
    cont, disc = ActionSpaceKind.CONTINUOUS, ActionSpaceKind.DISCRETE
    register_environment("cont")(_env_cls(cont))
    register_environment("disc")(_env_cls(disc))
    register_controller("pid")(_ctl_cls(cont))
    register_controller("dqn")(_ctl_cls(disc))
    register_controller("ppo")(_ctl_cls(disc, cont))
    assert [e.id for e in reg.compatible_controllers("cont")] == ["pid", "ppo"]
    assert [e.id for e in reg.compatible_controllers("disc")] == ["dqn", "ppo"]
    assert [e.id for e in reg.compatible_environments("dqn")] == ["disc"]


def _write_pkg(root: Path, name: str, modules: dict[str, str]) -> None:
    pkg = root / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for mod, body in modules.items():
        (pkg / f"{mod}.py").write_text(textwrap.dedent(body))


def test_discovery_of_fixture_package(
    reg: Registry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = """
        from rcl.core.types import ActionSpaceKind, EnvMetadata
        from rcl.registry import register_environment

        @register_environment("{id}")
        class E:
            env_metadata = EnvMetadata(
                name="{id}", dof=1, action_space_kind=ActionSpaceKind.CONTINUOUS,
                physics_backend="none",
            )
    """
    _write_pkg(tmp_path, "fx_envs", {"a": body.format(id="a"), "b": body.format(id="b")})
    monkeypatch.syspath_prepend(str(tmp_path))
    discover(["fx_envs"])
    assert [e.id for e in reg.entries("environment")] == ["a", "b"]


def test_discovery_reports_broken_module(
    reg: Registry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_pkg(tmp_path, "fx_broken", {"boom": "import does_not_exist_xyz\n"})
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(RegistryError, match="fx_broken.boom"):
        discover(["fx_broken"])
    sys.modules.pop("fx_broken", None)
