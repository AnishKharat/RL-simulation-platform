import pytest

from rcl.core.perturbation import ParamPerturbation, StepPerturbation
from rcl.core.types import PerturbationMetadata
from rcl.registry import Registry, RegistryError, register_perturbation, use_registry

META = PerturbationMetadata(
    name="p", version="0.1.0", stage="params", magnitude_param="scale", nominal_magnitude=1.0
)


class _Param(ParamPerturbation):
    perturbation_metadata = META

    @classmethod
    def apply(cls, env_params, params):  # type: ignore[no-untyped-def]
        return dict(env_params)


def test_register_and_lookup_perturbation() -> None:
    reg = Registry()
    with use_registry(reg):
        register_perturbation("dummy")(_Param)
    entry = reg.get("perturbation", "dummy")
    assert entry.cls is _Param
    assert entry.info()["metadata"]["magnitude_param"] == "scale"
    assert [e.id for e in reg.entries("perturbation")] == ["dummy"]


def test_perturbation_must_subclass_a_perturbation_base() -> None:
    class Plain:
        perturbation_metadata = META

    with pytest.raises(RegistryError, match="StepPerturbation or ParamPerturbation"):
        Registry().register("perturbation", "plain", Plain)


def test_perturbation_requires_metadata() -> None:
    class NoMeta(StepPerturbation):
        pass

    with pytest.raises(RegistryError, match="perturbation_metadata"):
        Registry().register("perturbation", "nometa", NoMeta)
