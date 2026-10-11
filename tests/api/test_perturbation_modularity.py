"""Modularity proof for perturbations: a new perturbation is ONE file with ONE decorator,
discovered outside rcl/perturbations, and the whole stack (listing, sweep validation, stored
runs, robustness curves) picks it up unchanged."""

import sys
import textwrap
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rcl.api.main import create_app
from rcl.controllers.classical.pid import PID, PIDParams
from rcl.environments.cartpole import CartPole, CartPoleParams
from rcl.registry import Registry, get_registry

DUMMY = """
    from typing import Any

    from pydantic import BaseModel, ConfigDict

    from rcl.core.environment import BaseEnvironment
    from rcl.core.perturbation import StepPerturbation
    from rcl.core.types import ActionSpaceKind, PerturbationMetadata
    from rcl.perturbations._action import as_command
    from rcl.registry import register_perturbation


    class BiasParams(BaseModel):
        model_config = ConfigDict(extra="forbid")
        bias: float = 0.0


    @register_perturbation("actuator_bias", params=BiasParams)
    class ActuatorBias(StepPerturbation):
        perturbation_metadata = PerturbationMetadata(
            name="Actuator bias", version="0.1.0", stage="step", magnitude_param="bias",
            magnitude_unit="N", nominal_magnitude=0.0,
            action_spaces=(ActionSpaceKind.CONTINUOUS,),
        )

        def __init__(self, env: BaseEnvironment, position: int = 0, bias: float = 0.0) -> None:
            super().__init__(env, position)
            self.bias = bias

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            self._seed_rng(seed)
            return self.env.reset(seed=seed, options=options)

        def step(self, action: Any):
            return self.env.step(as_command(action, self.action_space) + self.bias)
"""


def test_one_file_one_decorator_is_enough(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = tmp_path / "fx_perts"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "actuator_bias.py").write_text(textwrap.dedent(DUMMY))  # the ONLY new file
    monkeypatch.syspath_prepend(str(tmp_path))

    reg = Registry()  # isolated; only the env and controller are added by hand
    reg.register("environment", "cartpole", CartPole, CartPoleParams)
    reg.register("controller", "pid", PID, PIDParams)
    app = create_app(
        db_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
        artifact_root=str(tmp_path / "artifacts"),
        registry=reg,
        discovery_packages=["fx_perts"],
    )
    try:
        with TestClient(app) as client:
            listed = client.get("/perturbations").json()
            assert [p["id"] for p in listed] == ["actuator_bias"]
            assert listed[0]["metadata"]["magnitude_param"] == "bias"
            assert "bias" in listed[0]["params_schema"]["properties"]

            spec = {
                "name": "proof",
                "environments": [{"id": "cartpole"}],
                "controllers": [{"id": "pid"}],
                "seeds": [0, 1],
                "episodes": 1,
                "perturbation": {"id": "actuator_bias", "levels": [3.0]},
            }
            resp = client.post("/benchmarks", json=spec)
            assert resp.status_code == 202
            suite = resp.json()["suite_id"]
            deadline = time.time() + 120
            while time.time() < deadline:
                bench = client.get(f"/benchmarks/{suite}").json()["benchmark_runs"][0]
                if bench["status"] in ("done", "failed"):
                    break
                time.sleep(0.2)
            assert bench["status"] == "done"
            (curve,) = client.get(
                f"/benchmarks/{suite}/robustness", params={"metric": "angle_error"}
            ).json()
            assert [p["level"] for p in curve["points"]] == [0.0, 3.0]
            assert curve["points"][1]["mean"] != curve["points"][0]["mean"]  # the bias acted
    finally:
        sys.modules.pop("fx_perts.actuator_bias", None)
        sys.modules.pop("fx_perts", None)

    assert "actuator_bias" not in [e.id for e in get_registry().entries("perturbation")]
    assert not (Path(__file__).parents[2] / "rcl" / "perturbations" / "actuator_bias.py").exists()
