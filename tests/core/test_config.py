import copy
import json
from typing import Any

import pytest
from pydantic import ValidationError

from rcl.core.config import ExperimentConfig, RunRecord

VALID: dict[str, Any] = {
    "name": "pid_cartpole",
    "seed": 42,
    "environment": {"id": "cartpole", "params": {"length": 0.5}},
    "controller": {"id": "pid", "params": {"kp": 1.0}},
    "evaluation": {"episodes": 20},
}


def test_valid_and_defaults() -> None:
    cfg = ExperimentConfig.model_validate(VALID)
    assert cfg.training is None
    assert cfg.controller.params == {"kp": 1.0}


def test_json_round_trip() -> None:
    d = copy.deepcopy(VALID)
    d["training"] = {"total_timesteps": 1000}
    cfg = ExperimentConfig.model_validate(d)
    again = ExperimentConfig.model_validate(json.loads(json.dumps(cfg.model_dump(mode="json"))))
    assert again == cfg
    assert ExperimentConfig.model_validate_json(cfg.model_dump_json()) == cfg


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(seed=-1),
        lambda d: d.update(extra=1),
        lambda d: d["environment"].update(typo=1),
        lambda d: d["evaluation"].update(episodes=0),
        lambda d: d.pop("controller"),
        lambda d: d["controller"].update(id=""),
        lambda d: d.update(training={"total_timesteps": 0}),
    ],
)
def test_invalid(mutate: Any) -> None:
    d = copy.deepcopy(VALID)
    mutate(d)
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(d)


def test_run_record() -> None:
    cfg = ExperimentConfig.model_validate(VALID)
    rec = RunRecord(
        seed=7,
        derived_seeds={"env": 1, "train": 2, "eval": 3},
        config=cfg,
        config_hash=cfg.config_hash(),
        versions={"numpy": "2.0", "controller:pid": "0.1.0"},
        code_version="abc123",
    )
    assert RunRecord.model_validate_json(rec.model_dump_json()) == rec
    with pytest.raises(ValidationError):
        RunRecord.model_validate({**rec.model_dump(), "extra": 1})


def test_config_hash_ignores_seed_and_name_only() -> None:
    base = ExperimentConfig.model_validate(VALID)
    other = ExperimentConfig.model_validate({**VALID, "seed": 99, "name": "renamed"})
    assert base.config_hash() == other.config_hash()
    assert len(base.config_hash()) == 64
    changed = copy.deepcopy(VALID)
    changed["controller"]["params"]["kp"] = 2.0
    assert ExperimentConfig.model_validate(changed).config_hash() != base.config_hash()
    reordered = {k: VALID[k] for k in reversed(list(VALID))}
    assert ExperimentConfig.model_validate(reordered).config_hash() == base.config_hash()
