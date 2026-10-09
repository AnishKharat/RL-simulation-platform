"""Train PPO briefly, compare to a PID controller on the same env. Run: python run_spike.py"""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import stable_baselines3 as sb3
import torch
from pendulum_env import TinyPendulumEnv

Act = Callable[[np.ndarray], np.ndarray]


class PID:
    """Classical controller; only sees obs (same interface as a policy)."""

    def __init__(self, kp: float = 20.0, kd: float = 4.0, ki: float = 0.0) -> None:
        self.kp, self.kd, self.ki, self.i = kp, kd, ki, 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        th = float(np.arctan2(obs[0], obs[1]))
        self.i += th * TinyPendulumEnv.DT
        u = -(self.kp * th + self.kd * float(obs[2]) + self.ki * self.i)
        return np.array([np.clip(u / TinyPendulumEnv.MAX_TORQUE, -1, 1)], dtype=np.float32)


def rollout(env: TinyPendulumEnv, act: Act, seed: int) -> tuple[float, np.ndarray]:
    obs, _ = env.reset(seed=seed)
    total, traj, done = 0.0, [], False
    while not done:
        obs, r, term, trunc, _ = env.step(act(obs))
        total, done = total + r, term or trunc
        traj.append(obs)
    return total, np.array(traj)


def train(steps: int = 4096, seed: int = 0) -> sb3.PPO:
    return sb3.PPO(
        "MlpPolicy", TinyPendulumEnv(), seed=seed, device="cpu", n_steps=512
    ).learn(total_timesteps=steps)


def main() -> None:
    versions = {"sb3": sb3.__version__, "torch": torch.__version__, "numpy": np.__version__}
    print("versions:", versions)
    env = TinyPendulumEnv()
    t0 = time.time()
    model = train()
    print(f"train 4096 steps: {time.time() - t0:.1f}s")

    def ppo(o: np.ndarray) -> np.ndarray:
        return model.predict(o, deterministic=True)[0]

    pid = PID()
    print("ppo return seed=1:", round(rollout(env, ppo, 1)[0], 2))
    print("pid return seed=1:", round(rollout(env, pid.act, 1)[0], 2))


if __name__ == "__main__":
    main()
