"""Environment and controller registries."""

from rcl.registry.registry import (
    DISCOVERY_PACKAGES,
    DuplicateIdError,
    Entry,
    Kind,
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

__all__ = [
    "DISCOVERY_PACKAGES",
    "DuplicateIdError",
    "Entry",
    "Kind",
    "ParamValidationError",
    "Registry",
    "RegistryError",
    "UnknownIdError",
    "discover",
    "get_registry",
    "register_controller",
    "register_environment",
    "use_registry",
]
