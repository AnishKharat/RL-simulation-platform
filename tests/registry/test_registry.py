import json
import sys
import textwrap
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field

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
    ParamValidationError,
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
    model_config = ConfigDict(extra="forbid")
    kp: float = 1.0
    gains: tuple[float, float] = (0.0, 1.0)


def _env_meta(kind: ActionSpaceKind = ActionSpaceKind.CONTINUOUS) -> EnvMetadata:
    return EnvMetadata(
        name="e", version="0.1.0", dof=1, action_space_kind=kind, physics_backend="none"
    )


def _ctl_meta(*kinds: ActionSpaceKind) -> ControllerMetadata:
    return ControllerMetadata(
        name="c",
        version="0.1.0",
        family=ControllerFamily.CLASSICAL,
        supported_action_spaces=kinds or (ActionSpaceKind.CONTINUOUS,),
        requires_training=False,
    )


def _env_cls(kind: ActionSpaceKind = ActionSpaceKind.CONTINUOUS) -> type:
    return type("E", (BaseEnvironment,), {"env_metadata": _env_meta(kind)})


def _named_env(qualname: str) -> type:
    c = _env_cls()
    c.__module__, c.__qualname__ = "fx_mod", qualname
    return c


def _ctl_cls(*kinds: ActionSpaceKind) -> type:
    return type("C", (BaseController,), {"metadata": _ctl_meta(*kinds)})


@pytest.fixture
def reg():
    with use_registry(Registry()) as r:
        yield r


def test_register_lookup_and_info(reg: Registry) -> None:
    @register_controller("pid", params=Params)
    class PID(BaseController):
        metadata = _ctl_meta()

    entry = reg.get("controller", "pid")
    assert entry.cls is PID
    assert entry.info()["metadata"]["requires_training"] is False
    info = entry.info()
    assert info["params_schema"]["properties"]["kp"]["default"] == 1.0
    assert "params_schema" not in info["metadata"]  # schema comes from the params model only
    # resolved params: defaults filled, JSON-safe (tuple -> list)
    assert entry.validate_params({"kp": 2}) == {"kp": 2.0, "gains": [0.0, 1.0]}
    with pytest.raises(ParamValidationError):
        entry.validate_params({"kd": 1})
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


def test_params_model_must_forbid_extra(reg: Registry) -> None:
    class Loose(BaseModel):
        kp: float = 1.0

    with pytest.raises(RegistryError, match="forbid"):
        register_controller("loose", params=Loose)(_ctl_cls())


def test_no_params_model_accepts_only_empty(reg: Registry) -> None:
    register_environment("e")(_env_cls())
    entry = reg.get("environment", "e")
    assert entry.validate_params(None) == {} and entry.info()["params_schema"] == {}
    with pytest.raises(ValueError):
        entry.validate_params({"x": 1})


def test_class_must_subclass_base_of_its_kind(reg: Registry) -> None:
    plain_env = type("E", (), {"env_metadata": _env_meta()})
    plain_ctl = type("C", (), {"metadata": _ctl_meta()})
    with pytest.raises(RegistryError, match="BaseEnvironment"):
        register_environment("e")(plain_env)
    with pytest.raises(RegistryError, match="BaseController"):
        register_controller("c")(plain_ctl)
    with pytest.raises(RegistryError, match="BaseController"):
        register_controller("c")(_env_cls())  # wrong kind of base
    assert reg.entries("environment") == [] and reg.entries("controller") == []


def test_is_compatible(reg: Registry) -> None:
    cont, disc = ActionSpaceKind.CONTINUOUS, ActionSpaceKind.DISCRETE
    register_environment("cont")(_env_cls(cont))
    register_environment("disc")(_env_cls(disc))
    register_controller("pid")(_ctl_cls(cont))
    assert reg.is_compatible("pid", "cont") and not reg.is_compatible("pid", "disc")
    with pytest.raises(UnknownIdError):
        reg.is_compatible("nope", "cont")
    with pytest.raises(UnknownIdError):
        reg.is_compatible("pid", "nope")


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
        from rcl.core.environment import BaseEnvironment
        from rcl.core.types import ActionSpaceKind, EnvMetadata
        from rcl.registry import register_environment

        @register_environment("{id}")
        class E(BaseEnvironment):
            env_metadata = EnvMetadata(
                name="{id}", version="0.1.0", dof=1, action_space_kind=ActionSpaceKind.CONTINUOUS,
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


def test_compatibility_checks_linearization_and_action_dim(reg: Registry) -> None:
    def env(**kw: Any) -> type:
        meta = EnvMetadata(
            name="e",
            version="0.1.0",
            dof=2,
            action_space_kind=ActionSpaceKind.CONTINUOUS,
            physics_backend="none",
            **kw,
        )
        return type("E", (BaseEnvironment,), {"env_metadata": meta})

    def ctl(**kw: Any) -> type:
        meta = ControllerMetadata(
            name="c",
            version="0.1.0",
            family=ControllerFamily.CLASSICAL,
            supported_action_spaces=(ActionSpaceKind.CONTINUOUS,),
            requires_training=False,
            **kw,
        )
        return type("C", (BaseController,), {"metadata": meta})

    reg.register("environment", "plain", env())
    reg.register("environment", "lin", env(linearizable=True))
    reg.register("environment", "lin2", env(linearizable=True, action_dim=2))
    reg.register("controller", "any", ctl())
    reg.register("controller", "lqr", ctl(requires_linearization=True))
    reg.register("controller", "scalar", ctl(max_action_dim=1))
    ids = lambda es: [e.id for e in es]  # noqa: E731
    assert ids(reg.compatible_controllers("plain")) == ["any", "scalar"]
    assert ids(reg.compatible_controllers("lin")) == ["any", "lqr", "scalar"]
    assert ids(reg.compatible_controllers("lin2")) == ["any", "lqr"]
    assert ids(reg.compatible_environments("lqr")) == ["lin", "lin2"]
    assert ids(reg.compatible_environments("scalar")) == ["lin", "plain"]


def test_discovery_packages_constant_matches_import_linter_contracts() -> None:
    import tomllib
    from pathlib import Path

    from rcl.registry import DISCOVERY_PACKAGES

    cfg = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8"))
    contracts = {c["name"]: c for c in cfg["tool"]["importlinter"]["contracts"]}
    layers = next(c for c in contracts.values() if c["type"] == "layers")["layers"]
    tier = next(layer for layer in layers if "rcl.environments" in layer)
    independent = next(c for c in contracts.values() if c["type"] == "independence")["modules"]
    assert set(DISCOVERY_PACKAGES) == set(independent) == {m.strip() for m in tier.split("|")}
    discover()  # default scan imports every plugin package, including the empty ones


def test_param_validation_error_shape(reg: Registry) -> None:
    register_controller("pid", params=Params)(_ctl_cls())
    entry = reg.get("controller", "pid")
    with pytest.raises(ParamValidationError) as exc:
        entry.validate_params({"kp": "abc", "kd": 1, "gains": [1.0]})
    errs = exc.value.errors
    assert {tuple(e["loc"]) for e in errs} == {("kp",), ("kd",), ("gains", 1)}
    for e in errs:
        assert set(e) == {"loc", "msg", "type"} and isinstance(e["loc"], list)
        assert isinstance(e["msg"], str) and isinstance(e["type"], str)
    assert isinstance(exc.value, ValueError)
    json.dumps(errs)  # API-ready
    reg.register("environment", "plain", _env_cls())
    with pytest.raises(ParamValidationError) as exc2:
        reg.get("environment", "plain").validate_params({"x": 1})
    assert exc2.value.errors == [
        {"loc": ["x"], "msg": "no params accepted", "type": "extra_forbidden"}
    ]


class _Mode(Enum):
    A = "a"
    B = "b"


class _Inner(BaseModel):
    x: int = 0


class _GoodRich(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: _Mode = _Mode.A
    kind: Literal["p", "q"] = "p"
    n: int | None = Field(default=None, ge=1, le=10)
    gains: tuple[float, float, float] = (1.0, 0.0, 0.0)
    names: list[str] = []
    flag: bool = False


class _NestedParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inner: _Inner = _Inner()


class _ListOfObjects(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[_Inner] = []


class _DictParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    extra_map: dict[str, float] = {}


class _NestedArray(BaseModel):
    model_config = ConfigDict(extra="forbid")
    m: list[list[float]] = []


def test_param_subset_accepts_scalars_enums_tuples(reg: Registry) -> None:
    register_controller("rich", params=_GoodRich)(_ctl_cls())
    assert reg.get("controller", "rich").validate_params({"mode": "b"})["mode"] == "b"


@pytest.mark.parametrize("model", [_NestedParams, _ListOfObjects, _DictParams, _NestedArray])
def test_param_subset_rejects_nested_objects_and_refs(reg: Registry, model: type) -> None:
    with pytest.raises(RegistryError, match="supported subset"):
        register_controller("bad", params=model)(_ctl_cls())
    assert reg.entries("controller") == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_params_are_rejected_with_loc(reg: Registry, bad: float) -> None:
    register_controller("pid", params=Params)(_ctl_cls())
    entry = reg.get("controller", "pid")
    with pytest.raises(ParamValidationError) as exc:
        entry.validate_params({"kp": bad, "gains": [1.0, bad]})
    assert {(tuple(e["loc"]), e["type"]) for e in exc.value.errors} == {
        (("kp",), "finite_number"),
        (("gains", 1), "finite_number"),
    }
    assert entry.validate_params({"kp": 1e300})["kp"] == 1e300  # large but finite is fine
