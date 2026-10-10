"""Phase 3 modularity check: every registry-compatible controller runs on the PyBullet robots
with zero controller code changes (project_context 7, Phase 3)."""

import numpy as np
import pytest

pytest.importorskip("pybullet")

from rcl.core.config import ExperimentConfig  # noqa: E402
from rcl.experiments import execute  # noqa: E402
from rcl.registry import discover, get_registry  # noqa: E402

NEW_ROBOTS = ("balance_bot", "planar_arm")
TINY = {"n_steps": 32, "batch_size": 32, "n_epochs": 1, "learning_starts": 16, "buffer_size": 500}


def _cases():
    discover()
    reg = get_registry()
    return [(env, ctl.id) for env in NEW_ROBOTS for ctl in reg.compatible_controllers(env)]


@pytest.mark.parametrize(("env_id", "ctl_id"), _cases())
def test_compatible_controller_runs_unchanged(env_id: str, ctl_id: str):
    entry = get_registry().get("controller", ctl_id)
    schema = entry.params_model.model_fields if entry.params_model else {}
    params = {k: v for k, v in TINY.items() if k in schema}  # shrink RL budgets only
    config = ExperimentConfig.model_validate(
        {
            "name": "unchanged",
            "seed": 0,
            "environment": {"id": env_id},
            "controller": {"id": ctl_id, "params": params},
            "evaluation": {"episodes": 1},
            "training": {"total_timesteps": 64} if entry.metadata.requires_training else None,
        }
    )
    result = execute(config)
    (traj,) = result.trajectories
    spec_dim = get_registry().get("environment", env_id).metadata.action_dim
    assert traj.actions.reshape(traj.n_steps, -1).shape[1] == spec_dim
    assert all(np.isfinite(m.value) or m.status == "undefined" for m in result.metrics)
    assert result.record.versions["env:" + env_id] and "pybullet" in result.record.versions


def test_the_expected_controllers_are_compatible():
    pairs = set(_cases())
    assert {("planar_arm", "ppo"), ("planar_arm", "sac")} <= pairs  # Phase 3 exit criterion
    assert {("balance_bot", c) for c in ("lqr", "pid", "ppo", "sac")} <= pairs
