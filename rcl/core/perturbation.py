"""Perturbation base classes (ADR 0021). Imports nothing from the project.

Two stages share one registry kind ``perturbation``:

* ``step``: a :class:`StepPerturbation` (an ``EnvWrapper``) acting during an episode.
* ``params``: a :class:`ParamPerturbation` rewriting the environment's constructor params
  before the environment is built (mass/parameter variation); it never wraps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from rcl.core.environment import BaseEnvironment, EnvWrapper
from rcl.core.types import PerturbationMetadata


class StepPerturbation(EnvWrapper):
    """A seeded in-episode perturbation.

    Contract
    --------
    * ``position`` is the index in the perturbation stack; ``reset(seed=s)`` builds
      ``self.rng = default_rng(SeedSequence([s, position]))``, so the stream depends only on the
      episode seed and the stack position, never on the controller or the global RNG.
      ``reset`` without a seed raises ``ValueError`` (no perturbation is ever unseeded).
    * Draw a fixed number of random values per step, independent of action and state, and
      schedule discrete events by step index, so every controller sees the same sequence.
    * Set ``info["perturbed"] = True`` on each step at which a discrete event is injected.
    * Subclasses declare ``perturbation_metadata`` (stage ``"step"``).
    """

    perturbation_metadata: ClassVar[PerturbationMetadata]

    def __init__(self, env: BaseEnvironment, position: int = 0) -> None:
        super().__init__(env)
        self.position = position
        self.rng = np.random.default_rng(0)

    def _seed_rng(self, seed: int | None) -> None:
        if seed is None:
            raise ValueError(f"{type(self).__name__}.reset requires an explicit seed")
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, self.position]))


class ParamPerturbation(ABC):
    """Rewrites an environment's constructor params (stage ``"params"``)."""

    perturbation_metadata: ClassVar[PerturbationMetadata]

    @classmethod
    @abstractmethod
    def apply(cls, env_params: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """Return a NEW params dict for the environment; ``env_params`` is not modified.
        Raise ``ValueError`` for a target the environment does not have."""
