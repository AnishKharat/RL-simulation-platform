"""Environment and controller registry. See docs/architecture/registry.md.

Contract: imports only ``rcl.core``. Environments expose their static description as
``cls.env_metadata`` (ADR 0012), controllers as ``cls.metadata``; both are type-checked on register.
The default registry is the single piece of global state in the project.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

from rcl.core.types import ControllerMetadata, EnvMetadata

Kind = Literal["environment", "controller"]
# Class attribute holding the static metadata, and its required type, per kind.
_META: dict[str, tuple[str, type[BaseModel]]] = {
    "environment": ("env_metadata", EnvMetadata),
    "controller": ("metadata", ControllerMetadata),
}
_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class RegistryError(Exception):
    """Bad entry or failed discovery."""


class DuplicateIdError(RegistryError):
    """An id is already registered for this kind by a different class."""


class UnknownIdError(RegistryError, KeyError):
    """Lookup of an id that is not registered."""


@dataclass(frozen=True)
class Entry:
    """One registered class. ``metadata`` is ``cls.env_metadata`` (environments) or
    ``cls.metadata`` (controllers): core's ``EnvMetadata`` / ``ControllerMetadata``."""

    kind: Kind
    id: str
    cls: type
    metadata: Any
    params_model: type[BaseModel] | None = None

    def info(self) -> dict[str, Any]:
        """API/frontend-facing description: metadata plus the JSON schema for the param form."""
        meta = self.metadata.model_dump(mode="json")
        schema = self.params_model.model_json_schema() if self.params_model else {}
        return {"id": self.id, "kind": self.kind, "metadata": meta, "params_schema": schema}

    def validate_params(self, raw: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate ``raw`` against the param model (defaults filled in); {} if none declared."""
        if self.params_model is None:
            return dict(raw or {})
        return self.params_model(**(raw or {})).model_dump()


class Registry:
    """Two id namespaces (environments, controllers) with compatibility queries."""

    def __init__(self) -> None:
        self._entries: dict[Kind, dict[str, Entry]] = {"environment": {}, "controller": {}}

    def register(
        self, kind: Kind, id: str, cls: type, params: type[BaseModel] | None = None
    ) -> None:
        """Add an entry; raises RegistryError (bad entry) or DuplicateIdError."""
        if not _ID_RE.match(id):
            raise RegistryError(f"invalid {kind} id {id!r}: must match {_ID_RE.pattern}")
        attr, meta_type = _META[kind]
        meta = getattr(cls, attr, None)
        if not isinstance(meta, meta_type):
            raise RegistryError(
                f"{cls.__qualname__}.{attr} must be a {meta_type.__name__} instance, "
                f"got {type(meta).__name__}"
            )
        if params is not None and not (isinstance(params, type) and issubclass(params, BaseModel)):
            raise RegistryError(f"params for {id!r} must be a pydantic BaseModel subclass")
        existing = self._entries[kind].get(id)
        if existing is not None:
            # Same class re-imported (importlib.reload, module imported under two names)
            # yields a new class object, so identity is the wrong test.
            if _class_key(existing.cls) == _class_key(cls):
                return
            raise DuplicateIdError(
                f"{kind} {id!r} already registered by {existing.cls.__qualname__}"
            )
        self._entries[kind][id] = Entry(kind, id, cls, meta, params)

    def get(self, kind: Kind, id: str) -> Entry:
        try:
            return self._entries[kind][id]
        except KeyError:
            known = sorted(self._entries[kind])
            raise UnknownIdError(f"unknown {kind} {id!r}; known: {known}") from None

    def entries(self, kind: Kind) -> list[Entry]:
        return [self._entries[kind][i] for i in sorted(self._entries[kind])]

    def compatible_controllers(self, env_id: str) -> list[Entry]:
        """Controllers whose supported_action_spaces contain the env's action_space_kind."""
        kind = self.get("environment", env_id).metadata.action_space_kind
        return [c for c in self.entries("controller") if _supports(c.metadata, kind)]

    def compatible_environments(self, controller_id: str) -> list[Entry]:
        """Inverse of :meth:`compatible_controllers`."""
        meta = self.get("controller", controller_id).metadata
        return [
            e for e in self.entries("environment") if _supports(meta, e.metadata.action_space_kind)
        ]


def _class_key(cls: type) -> tuple[str, str]:
    return cls.__module__, cls.__qualname__


def _supports(controller_meta: Any, kind: str) -> bool:
    spaces: Collection[str] = controller_meta.supported_action_spaces
    return kind in spaces


_default = Registry()


def get_registry() -> Registry:
    """The process-wide registry that the decorators write to."""
    return _default


@contextmanager
def use_registry(registry: Registry) -> Iterator[Registry]:
    """Temporarily swap the default registry (test isolation)."""
    global _default
    previous, _default = _default, registry
    try:
        yield registry
    finally:
        _default = previous


def _decorator(kind: Kind, id: str, params: type[BaseModel] | None) -> Callable[[type], type]:
    def wrap(cls: type) -> type:
        get_registry().register(kind, id, cls, params)
        return cls

    return wrap


def register_environment(
    id: str, *, params: type[BaseModel] | None = None
) -> Callable[[type], type]:
    """Class decorator: register an environment under ``id``. Returns the class unchanged."""
    return _decorator("environment", id, params)


def register_controller(
    id: str, *, params: type[BaseModel] | None = None
) -> Callable[[type], type]:
    """Class decorator: register a controller under ``id``. Returns the class unchanged."""
    return _decorator("controller", id, params)


def discover(packages: Collection[str] = ("rcl.environments", "rcl.controllers")) -> None:
    """Import every module under ``packages`` (recursively) so their decorators run.

    A module that fails to import raises ``RegistryError`` naming that module (not just the
    package), chaining the original error.
    """
    for name in packages:
        for mod_name in _module_names(name):
            try:
                importlib.import_module(mod_name)
            except RegistryError:
                raise
            except Exception as exc:
                raise RegistryError(f"discovery failed importing {mod_name!r}: {exc!r}") from exc


def _module_names(package: str) -> list[str]:
    try:
        pkg = importlib.import_module(package)
        paths = getattr(pkg, "__path__", None)
        if paths is None:
            return [package]
        found = pkgutil.walk_packages(paths, prefix=f"{package}.", onerror=_raise)
        return [package, *(m.name for m in found)]
    except Exception as exc:
        raise RegistryError(f"discovery failed in {package!r}: {exc!r}") from exc


def _raise(name: str) -> None:
    raise ImportError(f"cannot import {name}")
