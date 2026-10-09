"""Spike: minimal continuous-action inverted pendulum with own dynamics (Gymnasium API)."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TinyPendulumEnv(gym.Env):
    """theta=0 is upright. obs=[sin, cos, omega]; action=torque in [-1, 1]."""

    DT, G, MAX_TORQUE, MAX_OMEGA = 0.02, 9.81, 4.0, 8.0

    def __init__(self, max_steps: int = 200) -> None:
        self.max_steps = max_steps
        self.observation_space = spaces.Box(-np.inf, np.inf, (3,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        self._state = np.zeros(2)
        self._t = 0

    def _obs(self) -> np.ndarray:
        th, om = self._state
        return np.array([np.sin(th), np.cos(th), om], dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)  # seeds self.np_random; the only RNG used
        self._state = self.np_random.uniform(-0.3, 0.3, size=2)
        self._t = 0
        return self._obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        th, om = self._state
        u = float(np.clip(action, -1, 1)[0]) * self.MAX_TORQUE
        om += (self.G * np.sin(th) + u) * self.DT
        om = float(np.clip(om, -self.MAX_OMEGA, self.MAX_OMEGA))
        th = float((th + om * self.DT + np.pi) % (2 * np.pi) - np.pi)
        self._state = np.array([th, om])
        self._t += 1
        reward = -(th**2 + 0.1 * om**2 + 0.001 * u**2)
        return self._obs(), reward, False, self._t >= self.max_steps, {}
