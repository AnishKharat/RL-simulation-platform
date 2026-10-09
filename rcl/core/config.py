"""Experiment configuration schema: the single source of truth for a run (project_context 5.4).

YAML-friendly: load YAML to a plain dict and call `ExperimentConfig.model_validate(d)`;
`model_dump(mode="json")` gives a dict that round-trips through YAML/JSON/DB storage verbatim.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComponentConfig(_Strict):
    """Reference to a registry entry: `id` is a registry key, `params` are validated later
    against that entry's params model (core cannot know it). The experiment manager replaces
    `params` with the RESOLVED params (defaults filled, `Entry.validate_params`) before the
    config is hashed and stored, so a stored config never depends on today's defaults."""

    id: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class PerturbationConfig(ComponentConfig):
    """A perturbation (noise, disturbance, delay) applied as an `EnvWrapper` around the
    environment. `id` and `params` are as in `ComponentConfig`. `apply_to` selects the stack:
    "eval" (default; robustness testing of a controller trained on the clean env), "train"
    or "both". The wrapper stack is applied in list order."""

    apply_to: Literal["train", "eval", "both"] = "eval"


class TrainingSection(_Strict):
    """Training budget (RL only; ignored by classical controllers). Hyperparameters live in
    `controller.params`, not here."""

    total_timesteps: int = Field(gt=0)


class EvaluationSection(_Strict):
    """Evaluation protocol: identical `episodes` count for every compared controller."""

    episodes: int = Field(gt=0)


class ExperimentConfig(_Strict):
    """Complete, validated description of one experiment; sufficient to reproduce it.

    Seeding contract: `seed` is the ONLY seed in the config, the default root seed of the
    experiment. Each Run may override it (`RunRecord.seed` is authoritative for that run);
    the environment, training and evaluation seeds are derived from the run's seed by the
    experiment manager with `numpy.random.SeedSequence(seed)` spawning; evaluation episode i
    resets the environment with `eval_seed + i`.

    Fairness contract: for a given run seed every controller is evaluated on identical
    evaluation seeds (`eval_seed + i`), and the training environment is built with the same
    wrapper stack as the evaluation environment, differing only by perturbations explicitly marked
    `apply_to="eval"`.

    `config_hash()` identifies the experiment DEFINITION and therefore excludes `seed` (varied
    per Run) and `name` (a label), so all seeds of one benchmark cell share one hash.
    """

    schema_version: int = 1
    name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    environment: ComponentConfig
    controller: ComponentConfig
    perturbations: list[PerturbationConfig] = Field(default_factory=list)
    training: TrainingSection | None = None
    evaluation: EvaluationSection

    def config_hash(self) -> str:
        """sha256 hex of the canonical JSON (sorted keys, no spaces) of the config without
        `seed` and `name`; an empty `perturbations` list is left out of the payload, so adding
        the field did not change the hash of perturbation-free configs. Hash the RESOLVED
        config (defaults filled by the registry)."""
        data = self.model_dump(mode="json", exclude={"seed", "name"})
        if not data["perturbations"]:
            del data["perturbations"]
        blob = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class RunRecord(_Strict):
    """Reproducibility block stored with every run: what exactly was executed.

    * `seed`: the AUTHORITATIVE root seed of this run (may differ from `config.seed`, the
      experiment default); `derived_seeds` are spawned from it, keys `env`, `train`, `eval`.
    * `config` is the validated config as run (resolved params) and `config_hash` its
      `ExperimentConfig.config_hash()` (excludes seed and name).
    * `versions` maps name to version: python, numpy, gymnasium, pydantic, backend libs,
      platform, plus registry entries as `env:<id>` and `controller:<id>`. RL runs also record
      the numeric environment: `torch`, `cuda` ("none" if unavailable), `device` (e.g. "cpu")
      and `num_threads` (`torch.get_num_threads()`), all as strings.
    * `code_version` is the commit sha (suffix `-dirty` if the tree had uncommitted changes).
    """

    seed: int = Field(ge=0)
    derived_seeds: dict[str, int]
    config: ExperimentConfig
    config_hash: str
    versions: dict[str, str]
    code_version: str
