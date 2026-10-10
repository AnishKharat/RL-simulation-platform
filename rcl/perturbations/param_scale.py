"""Mass/parameter variation: scale named numeric environment params before construction."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rcl.core.perturbation import ParamPerturbation
from rcl.core.types import PerturbationMetadata
from rcl.registry import register_perturbation


class ParamScaleParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    targets: list[str] = Field(min_length=1, description="Environment params to scale")
    scale: float = Field(1.0, gt=0, description="Multiplier applied to every target")


@register_perturbation("param_scale", params=ParamScaleParams)
class ParamScale(ParamPerturbation):
    perturbation_metadata = PerturbationMetadata(
        name="Parameter scale",
        version="0.1.0",
        description="Multiplies the named environment params (e.g. a mass) by `scale`.",
        stage="params",
        magnitude_param="scale",
        magnitude_unit="x nominal",
        nominal_magnitude=1.0,
    )

    @classmethod
    def apply(cls, env_params: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        out = dict(env_params)
        for name in params["targets"]:
            value = out.get(name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"cannot scale {name!r}: not a numeric environment param")
            out[name] = value * params["scale"]
        return out
