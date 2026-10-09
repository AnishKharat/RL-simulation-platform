"""Experiment configuration schema: the single source of truth for a run (project_context 5.4).

YAML-friendly: load YAML to a plain dict and call `ExperimentConfig.model_validate(d)`;
`model_dump(mode="json")` gives a dict that round-trips through YAML/JSON/DB storage verbatim.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComponentConfig(_Strict):
    """Reference to a registry entry: `id` is a registry key, `params` are validated later
    against that entry's parameter schema (core cannot know it)."""

    id: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class TrainingSection(_Strict):
    """Training budget (RL only; ignored by classical controllers)."""

    total_timesteps: int = Field(gt=0)
    params: dict[str, Any] = Field(default_factory=dict)


class EvaluationSection(_Strict):
    """Evaluation protocol: identical `episodes` count for every compared controller."""

    episodes: int = Field(gt=0)


class ExperimentConfig(_Strict):
    """Complete, validated description of one experiment; sufficient to reproduce it.

    Seeding contract: `seed` is the ONLY seed in the config. The environment, training and
    evaluation seeds are derived from it by the experiment manager with
    `numpy.random.SeedSequence(seed)` spawning; evaluation episode i resets the environment
    with `eval_seed + i`. Stored verbatim in the DB.
    """

    name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    environment: ComponentConfig
    controller: ComponentConfig
    training: TrainingSection | None = None
    evaluation: EvaluationSection


class RunRecord(_Strict):
    """Reproducibility block stored with every run: what exactly was executed.

    `config` is the validated config as run, `library_versions` maps package name to
    version (numpy, gymnasium, pydantic, backend libs, ...), `code_version` is the git commit
    (suffix `-dirty` if the tree had uncommitted changes).
    """

    seed: int = Field(ge=0)
    config: ExperimentConfig
    library_versions: dict[str, str]
    code_version: str
