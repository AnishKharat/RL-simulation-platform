# Phase 3 benchmarks: PyBullet robots

Generated with the Phase 2 benchmark CLI (same fairness protocol: identical seeds, episodes, training budget and robot
definition for every controller). Each robot folder has the full `report.md`, `summary.csv` and SVG charts (means
with 95% intervals across 5 seeds).

```bash
pip install -e ".[dev,physics]"   # PyBullet: wheel on Linux/CPython 3.11, else a C++ build (ADR 0020)
python -m rcl.benchmark --spec docs/benchmarks/phase3/balance_bot.json --out out/balance_bot
python -m rcl.benchmark --spec docs/benchmarks/phase3/planar_arm.json --out out/planar_arm
```

Run on Windows 11, CPython 3.12, PyBullet 3.2.7 (built from source), CPU. Reproducible on the same machine and build
(ADR 0010, spike `spikes/pybullet_determinism`).

## Self-balancing robot (`balance_bot`): PID vs LQR vs PPO vs SAC

| controller | angle error (RMS pitch, rad) | return | control effort (N*m*s) | notes |
|---|---|---|---|---|
| LQR (from `get_linearization`) | **0.012** [0.011, 0.014] | 117 [107, 127] | 0.005 | reaches the success band; episodes end early on success |
| PID (pitch only: kp 3, kd 0.3) | 0.038 [0.038, 0.038] | 497 | 7.0 | balances the whole horizon, but outside the 0.02 rad success band |
| PPO (10k steps) | 0.114 [0.105, 0.124] | 51 | 0.56 | falls; budget too small |
| SAC (10k steps) | 0.121 [0.070, 0.173] | 144 [76, 213] | 1.1 | falls in most episodes; budget too small |

## Planar arm (`planar_arm`): PPO vs SAC

| controller | end-effector error (RMS, m) | return | notes |
|---|---|---|---|
| PPO (10k steps) | 0.68 [0.52, 0.84] | 70 [11, 129] | no reaching learned yet |
| SAC (10k steps) | 0.57 [0.43, 0.72] | 108 [68, 147] | no reaching learned yet; settling censored at T_max (4 s) |

## Reading these results

- **The Phase 3 modularity criterion holds:** every compatible existing controller ran on both new robots with zero
  code changes (`tests/experiments/test_controllers_on_new_robots.py`). LQR used the robot's finite-difference
  linearisation; the registry hid PID and LQR on the arm.
- **Return is not the headline metric when success ends an episode.** LQR settles inside the success band, so its
  episodes stop early and collect fewer +1 rewards than PID's full-length episodes. Compare `angle_error`, success and
  settling time instead.
- **RL budgets here are deliberately small** (10k steps, so the whole benchmark runs in minutes). They show the
  pipeline, not tuned RL performance. Expect PPO/SAC to need about 1e5-1e6 steps on these tasks.
- Intervals come from 5 seeds (Student t; bootstrap for settling and rise time). Pairs with overlapping intervals
  are not claimed to differ.
