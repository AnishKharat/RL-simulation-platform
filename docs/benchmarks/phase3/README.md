# Phase 3 benchmarks: PyBullet robots

Generated with the Phase 2 benchmark CLI (same fairness protocol: identical seeds, episodes, training budget and robot
definition for every controller). Each folder has the full `report.md`, `summary.csv` and SVG charts (means with 95%
intervals across 5 seeds).

```bash
pip install -e ".[dev,physics]"   # PyBullet: wheel on Linux/CPython 3.11, else a C++ build (ADR 0020)
python -m rcl.benchmark --spec docs/benchmarks/phase3/balance_bot.json --out out/balance_bot
python -m rcl.benchmark --spec docs/benchmarks/phase3/planar_arm.json --out out/planar_arm   # ~65 min on CPU
```

Run on Windows 11, CPython 3.12, PyBullet 3.2.7 (built from source), CPU. Reproducible on the same machine and build
(ADR 0010, spike `spikes/pybullet_determinism`).

## Phase 3 exit criterion: "two PyBullet robots, each benchmarked by at least 3 controllers"

**Met by count only.** `balance_bot` has four controllers (PID, LQR, PPO, SAC) and one of them, LQR, genuinely solves
the task. `planar_arm` has three (PPO, SAC, TD3; PID and LQR are correctly hidden by the registry), but **no
controller ever reached the task: 0 successes in 75 evaluation episodes at 30k training steps** (and 1 in 75 across the
earlier 10k-step runs). The arm benchmark therefore shows that the pipeline runs three controllers on a 2-action robot
with no code changes; it does not show a competent arm controller.

## Headline metrics (read these first)

`success_rate` (fraction of episodes in which the tolerance was held) and `time_to_success` are the headline metrics.
**`return` is not the headline metric**: success ends an episode early, so a controller that succeeds can collect a
lower return than one that never does (LQR vs PID below). Reports and the dashboard now lead with the headline metrics.

## Self-balancing robot (`balance_bot`): PID vs LQR vs PPO vs SAC (10k RL steps)

Success rates are counted from the stored episodes (the run predates the `success_rate` metric).

| controller | success (25 episodes) | angle error (RMS pitch, rad) | return | notes |
|---|---|---|---|---|
| LQR (from `get_linearization`) | **25/25** | **0.012** [0.011, 0.014] | 117 [107, 127] | succeeds; episodes end early, hence the lower return |
| PID (pitch only: kp 3, kd 0.3) | 0/25 | 0.038 [0.038, 0.038] | 497 | balances the whole horizon, but outside the 0.02 rad success band |
| PPO (10k steps) | 0/25 | 0.114 [0.105, 0.124] | 51 | falls; budget too small |
| SAC (10k steps) | 1/25 | 0.121 [0.070, 0.173] | 144 [76, 213] | falls in most episodes; budget too small |

## Planar arm (`planar_arm`): PPO vs SAC vs TD3 (30k RL steps, 5 seeds x 5 episodes)

Reaching a random 2-D target; RMS end-effector error in metres. Reference: a **zero-torque baseline** (the arm does
not move) has mean RMS error **0.89 m** on the same evaluation episodes (measured with an ad-hoc script, not kept).

| controller | success (25 episodes) | time to success | RMS ee error (m) | return |
|---|---|---|---|---|
| SAC | 0/25 | never (censored at 4 s) | **0.47** [0.39, 0.55] | 141 [120, 161] |
| PPO | 0/25 | never | 0.66 [0.52, 0.80] | 83 [43, 124] |
| TD3 | 0/25 | never | 0.74 [0.56, 0.92] | 42 [-40, 123] |

What this says, and does not say:
- **Nothing succeeds.** All 75 episodes fail the 0.02 m tolerance, so `success_rate` is 0 with a degenerate [0, 0]
  interval and `time_to_success` is censored at the 4 s limit for every run.
- **Only SAC learned something measurable**: its error is about half the do-nothing baseline (0.47 vs 0.89 m) and its
  interval lies clearly below it. PPO is modestly better than the baseline; TD3's interval reaches the baseline, so no
  improvement is claimed. SAC's and PPO's intervals overlap slightly (0.52 to 0.55 m), so the only ranking claimed
  is "SAC has the lowest mean error"; with 5 seeds that is thin evidence.
- **Earlier 10k-step run** (`planar_arm_10k/`, PPO and SAC only): success 0/25 and 1/25; error 0.68 and 0.57 m. Tripling
  the budget helped SAC (0.57 -> 0.47 m) and did not help PPO.
- **Budgets are small for this task.** 30k steps took about 65 minutes for the full arm benchmark on CPU (PPO ~2 min,
  SAC ~5 min, TD3 ~4.5 min per seed). Reaching a 2 cm tolerance with a learned torque policy will likely need an order
  of magnitude more steps and probably reward shaping or tuning; none of that was attempted.

## Reading these results

- **The Phase 3 modularity criterion holds:** every compatible existing controller ran on both new robots with zero
  code changes (`tests/experiments/test_controllers_on_new_robots.py`). LQR used the robot's finite-difference
  linearisation; the registry hid PID and LQR on the arm.
- Intervals come from 5 seeds (Student t; bootstrap for settling time, rise time, success rate and time to success).
  Pairs with overlapping intervals are not claimed to differ.
