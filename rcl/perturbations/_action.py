"""Shared helpers for actuator-side perturbations (continuous Box action spaces)."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


def action_limit(space: gym.Space[Any]) -> np.ndarray:
    """Per-component actuator limit ``high`` of a Box action space."""
    assert isinstance(space, gym.spaces.Box)
    return np.asarray(space.high, dtype=np.float64)


def as_command(action: object, space: gym.Space[Any]) -> np.ndarray:
    """The controller's command clipped to the actuator range, with the space's shape."""
    assert isinstance(space, gym.spaces.Box)
    a = np.asarray(action, dtype=np.float64).reshape(space.shape)
    return np.clip(a, space.low, space.high)
