"""Modularity proof: a new robot is ONE file with ONE decorator, discovered outside
rcl/environments, and the whole stack (API listing, compatibility, run) picks it up unchanged."""

import sys
import textwrap
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rcl.api.main import create_app
from rcl.controllers.classical.pid import PID, PIDParams
from rcl.registry import Registry, get_registry

DUMMY_ENV = """
    from typing import Any, ClassVar

    import gymnasium as gym
    import numpy as np

    from rcl.core.environment import BaseEnvironment
    from rcl.core.types import ActionSpaceKind, EnvMetadata, EnvSpec, TaskSpec
    from rcl.registry import register_environment


    @register_environment("point_mass")
    class PointMass(BaseEnvironment):
        env_metadata: ClassVar[EnvMetadata] = EnvMetadata(
            name="Point Mass", version="0.1.0", dof=1, physics_backend="none",
            action_space_kind=ActionSpaceKind.CONTINUOUS,
            state_labels=("x",), state_units=("m",), action_labels=("u",), action_units=("m/s",),
        )
        metadata = {"render_modes": []}  # noqa: RUF012

        def __init__(self) -> None:
            super().__init__()
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), np.float64)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float64)
            self._x, self._t = np.zeros(1), 0

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            super().reset(seed=seed)
            self._x, self._t = self.np_random.uniform(-1, 1, size=1), 0
            return self._x.copy(), {}

        def step(self, action: Any):
            u = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
            self._x = self._x + 0.1 * u
            self._t += 1
            info = {"applied_action": np.array([u])}
            return self._x.copy(), -abs(float(self._x[0])), False, self._t >= 50, info

        def get_state(self) -> np.ndarray:
            return self._x.copy()

        def get_env_spec(self) -> EnvSpec:
            return EnvSpec(
                env_id="point_mass", action_space_kind=ActionSpaceKind.CONTINUOUS,
                obs_shape=(1,), state_dim=1, dt=0.1, action_shape=(1,),
                action_low=(-1.0,), action_high=(1.0,),
            )

        def get_task_spec(self) -> TaskSpec:
            return TaskSpec(
                setpoint=(0.0,), tolerance=(0.05,), error_state_indices=(0,), step_floor=0.05,
                horizon_steps=50, success_hold_steps=5,
            )
"""


def test_one_file_one_decorator_is_enough(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = tmp_path / "fx_robots"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "point_mass.py").write_text(textwrap.dedent(DUMMY_ENV))  # the ONLY new file
    monkeypatch.syspath_prepend(str(tmp_path))

    reg = Registry()  # isolated; the existing controller is added by hand, nothing else is
    reg.register("controller", "pid", PID, PIDParams)
    app = create_app(
        db_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
        artifact_root=str(tmp_path / "artifacts"),
        registry=reg,
        discovery_packages=["fx_robots"],
    )
    try:
        with TestClient(app) as client:
            listed = [e["id"] for e in client.get("/environments").json()]
            assert listed == ["point_mass"]
            compat = client.get("/controllers/compatible", params={"env_id": "point_mass"})
            assert [e["id"] for e in compat.json()] == ["pid"]

            config = {
                "name": "proof",
                "seed": 1,
                "environment": {"id": "point_mass"},
                "controller": {"id": "pid", "params": {"obs_index": 0, "kp": -2.0, "kd": 0.0}},
                "evaluation": {"episodes": 2},
            }
            resp = client.post("/experiments", json=config)
            assert resp.status_code == 202
            run_id = resp.json()["run_ids"][0]
            deadline = time.time() + 60
            while time.time() < deadline:
                run = client.get(f"/runs/{run_id}").json()
                if run["status"] in ("done", "failed"):
                    break
                time.sleep(0.1)
            assert run["status"] == "done", run["error"]
            traj = client.get(f"/runs/{run_id}/trajectory/0").json()
            assert traj["state_labels"] == ["x"] and traj["action_units"] == ["m/s"]
    finally:
        sys.modules.pop("fx_robots.point_mass", None)
        sys.modules.pop("fx_robots", None)

    assert "point_mass" not in [e.id for e in get_registry().entries("environment")]
    assert not (Path(__file__).parents[2] / "rcl" / "environments" / "point_mass.py").exists()
