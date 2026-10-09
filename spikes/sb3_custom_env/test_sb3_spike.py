"""Run only via: pytest spikes/sb3_custom_env (not collected by root `pytest`)."""

from __future__ import annotations

import numpy as np
from pendulum_env import TinyPendulumEnv
from run_spike import PID, rollout, train
from stable_baselines3 import PPO


def test_env_seed_reproducible() -> None:
    env = TinyPendulumEnv()
    a = rollout(env, PID().act, seed=3)[1]
    b = rollout(env, PID().act, seed=3)[1]
    c = rollout(env, PID().act, seed=4)[1]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_pid_stabilises() -> None:
    assert rollout(TinyPendulumEnv(), PID().act, seed=1)[0] > -5


def test_ppo_train_seed_reproducible_and_save_load(tmp_path) -> None:  # noqa: ANN001
    m1, m2 = train(1024, seed=0), train(1024, seed=0)
    env = TinyPendulumEnv()

    def det(m: PPO):  # noqa: ANN202
        return lambda o: m.predict(o, deterministic=True)[0]

    t1 = rollout(env, det(m1), seed=7)[1]
    assert np.array_equal(t1, rollout(env, det(m2), seed=7)[1])  # same train seed -> same policy
    m1.save(tmp_path / "ppo")
    loaded = PPO.load(tmp_path / "ppo", device="cpu")
    assert np.array_equal(t1, rollout(env, det(loaded), seed=7)[1])  # round trip exact
