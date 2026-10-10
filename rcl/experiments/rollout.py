"""The single shared evaluation loop (ADR 0013)."""

from __future__ import annotations

import numpy as np

from rcl.core.controller import BaseController
from rcl.core.environment import BaseEnvironment
from rcl.core.types import EpisodeTrajectory, TaskSpec
from rcl.metrics import wrap_angle


def _in_tolerance(state: np.ndarray, task: TaskSpec) -> bool:
    """|wrap(state - setpoint)| <= tolerance for every component (TaskSpec success rule)."""
    d = state - np.asarray(task.setpoint)
    w = list(task.wrap_indices)
    d[w] = wrap_angle(d[w])
    return bool(np.all(np.abs(d) <= np.asarray(task.tolerance)))


def rollout(
    env: BaseEnvironment, controller: BaseController, n_episodes: int, seed: int
) -> list[EpisodeTrajectory]:
    """Evaluate ``controller`` for ``n_episodes``; episode ``i`` resets with ``seed + i``.

    Each episode calls ``env.reset``, ``controller.reset`` and then ``predict(obs,
    deterministic=True)`` / ``env.step`` until failure (``terminated``), the task horizon
    (``truncated``) or ``success_hold_steps`` consecutive in-tolerance steps (both flags False).
    Records true states, observations, raw commands and the applied actions
    (``info["applied_action"]``). The caller owns (and closes) the environment.
    """
    task, dt = env.get_task_spec(), env.get_env_spec().dt
    trajectories = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        controller.reset()
        observations, states = [np.asarray(obs, dtype=np.float64)], [env.get_state()]
        commanded, applied, rewards = [], [], []
        terminated = truncated = success = False
        hold = 0
        for _ in range(task.horizon_steps):
            command = controller.predict(obs, deterministic=True)
            obs, reward, terminated, env_truncated, info = env.step(command)
            commanded.append(np.asarray(command, dtype=np.float64))
            applied.append(np.asarray(info["applied_action"], dtype=np.float64))
            rewards.append(float(reward))
            observations.append(np.asarray(obs, dtype=np.float64))
            states.append(env.get_state())
            if terminated:
                break
            hold = hold + 1 if _in_tolerance(states[-1], task) else 0
            if hold >= task.success_hold_steps:
                success = True
                break
            if env_truncated:
                truncated = True
                break
        else:
            truncated = True
        trajectories.append(
            EpisodeTrajectory(
                observations=np.stack(observations),
                actions=np.stack(applied),
                rewards=np.asarray(rewards),
                states=np.stack(states),
                dt=dt,
                terminated=terminated,
                truncated=truncated,
                seed=seed + i,
                success=success,
                commanded_actions=np.stack(commanded),
            )
        )
    return trajectories
