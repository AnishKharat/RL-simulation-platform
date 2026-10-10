import numpy as np
import pytest

from rcl.core.config import ExperimentConfig
from rcl.experiments import ConfigError, execute
from rcl.registry import discover


@pytest.fixture(autouse=True, scope="module")
def _discovered():
    discover()


def cfg(perturbations=(), seed=3, episodes=3) -> ExperimentConfig:
    return ExperimentConfig(
        name="t",
        seed=seed,
        environment={"id": "cartpole"},
        controller={"id": "pid", "params": {"kp": 55.0}},
        perturbations=list(perturbations),
        evaluation={"episodes": episodes},
    )


def with_(config: ExperimentConfig, **update) -> ExperimentConfig:
    return ExperimentConfig.model_validate({**config.model_dump(), **update})


DISTURB = {"id": "disturbance", "params": {"magnitude": 0.3, "start_min": 5, "start_max": 10}}


def test_config_hash_comes_from_spec_not_from_recorded_steps() -> None:
    assert cfg([DISTURB]).config_hash() == cfg([DISTURB]).config_hash()
    assert cfg([DISTURB]).config_hash() != cfg().config_hash()
    assert cfg().config_hash() == cfg(perturbations=[]).config_hash()
    result = execute(cfg([DISTURB]))
    assert all(t.perturbation_steps for t in result.trajectories)  # steps were recorded...
    # ...yet the stored hash equals the hash of the input spec (resolved params only)
    assert result.record.config_hash == result.record.config.config_hash()
    again = execute(cfg([DISTURB]))
    assert result.record.config_hash == again.record.config_hash


def test_perturbed_run_is_reproducible_and_records_steps() -> None:
    a, b = execute(cfg([DISTURB])), execute(cfg([DISTURB]))
    for x, y in zip(a.trajectories, b.trajectories, strict=True):
        assert np.array_equal(x.states, y.states) and x.perturbation_steps == y.perturbation_steps
    steps = a.trajectories[0].perturbation_steps
    assert len(steps) == 5 and 5 <= steps[0] <= 10
    assert a.record.config.perturbations[0].params["duration_steps"] == 5  # resolved


def test_eval_perturbation_changes_the_run_but_nominal_controller_is_configured_clean() -> None:
    clean = execute(cfg())
    noisy = execute(cfg([{"id": "obs_noise", "params": {"std": 0.05}}]))
    assert not np.array_equal(
        clean.trajectories[0].observations, noisy.trajectories[0].observations
    )
    assert all(not t.perturbation_steps for t in noisy.trajectories)


def test_param_scale_changes_the_plant() -> None:
    heavy = {"id": "param_scale", "params": {"targets": ["masspole"], "scale": 5.0}}
    a, b = execute(cfg()), execute(cfg([heavy]))
    assert not np.array_equal(a.trajectories[0].states, b.trajectories[0].states)


def test_same_perturbation_sequence_for_every_controller() -> None:
    def steps(ctl):
        c = with_(cfg([DISTURB]), controller={"id": ctl, "params": {}})
        return [t.perturbation_steps[:1] for t in execute(c).trajectories]

    assert steps("pid") == steps("lqr")


@pytest.mark.parametrize(
    ("pert", "loc", "kind"),
    [
        ({"id": "nope"}, ("perturbations", 0, "id"), "unknown_id"),
        ({"id": "obs_noise", "params": {"std": -1}}, ("perturbations", 0, "params", "std"), None),
        (
            {"id": "param_scale", "params": {"targets": ["nope"], "scale": 2}},
            ("perturbations", 0, "params"),
            "invalid_configuration",
        ),
    ],
)
def test_bad_perturbations_are_rejected_with_a_location(pert, loc, kind) -> None:
    with pytest.raises(ConfigError) as exc:
        execute(cfg([pert]))
    assert any(tuple(e["loc"]) == loc and kind in (None, e["type"]) for e in exc.value.errors)


def test_action_perturbation_is_rejected_on_a_discrete_environment() -> None:
    c = with_(cfg([DISTURB]), environment={"id": "cartpole_discrete"})
    with pytest.raises(ConfigError) as exc:
        execute(c)
    assert ("perturbations", 0, "id") in {(*e["loc"],) for e in exc.value.errors}
