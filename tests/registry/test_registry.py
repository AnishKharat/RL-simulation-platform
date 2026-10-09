import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

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


class Meta(BaseModel):
    action_space_kind: str = "continuous"
    supported_action_spaces: tuple[str, ...] = ("continuous",)


class Params(BaseModel):
    kp: float = 1.0


def _cls(meta: Meta | None = None) -> type:
    return type("X", (), {"metadata": meta or Meta()})


@pytest.fixture
def reg():
    with use_registry(Registry()) as r:
        yield r


def test_register_lookup_and_info(reg: Registry) -> None:
    @register_controller("pid", params=Params)
    class PID:
        metadata = Meta()

    entry = reg.get("controller", "pid")
    assert entry.cls is PID
    info = entry.info()
    assert info["params_schema"]["properties"]["kp"]["default"] == 1.0
    assert entry.validate_params({"kp": 2}) == {"kp": 2.0}
    assert get_registry() is reg


def test_duplicate_id(reg: Registry) -> None:
    register_environment("e")(_cls())
    with pytest.raises(DuplicateIdError):
        register_environment("e")(_cls())
    register_controller("e")(_cls())  # separate namespace


def test_same_class_twice_is_noop(reg: Registry) -> None:
    c = _cls()
    register_environment("e")(c)
    register_environment("e")(c)
    assert len(reg.entries("environment")) == 1


@pytest.mark.parametrize("bad_id", ["Bad", "1x", "a-b", ""])
def test_bad_id(reg: Registry, bad_id: str) -> None:
    with pytest.raises(RegistryError):
        register_environment(bad_id)(_cls())


def test_missing_metadata_and_bad_params(reg: Registry) -> None:
    with pytest.raises(RegistryError):
        register_environment("e")(type("NoMeta", (), {}))
    bad: Any = int
    with pytest.raises(RegistryError):
        register_environment("e", params=bad)(_cls())


def test_unknown_id(reg: Registry) -> None:
    with pytest.raises(UnknownIdError):
        reg.get("environment", "nope")


def test_compatibility_filter(reg: Registry) -> None:
    register_environment("cont")(_cls(Meta(action_space_kind="continuous")))
    register_environment("disc")(_cls(Meta(action_space_kind="discrete")))
    register_controller("pid")(_cls(Meta(supported_action_spaces=("continuous",))))
    register_controller("dqn")(_cls(Meta(supported_action_spaces=("discrete",))))
    register_controller("ppo")(_cls(Meta(supported_action_spaces=("discrete", "continuous"))))
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
        from rcl.registry import register_environment

        @register_environment("{id}")
        class E:
            metadata = {{"action_space_kind": "continuous"}}
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
    with pytest.raises(RegistryError, match="fx_broken"):
        discover(["fx_broken"])
    sys.modules.pop("fx_broken", None)
