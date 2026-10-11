# Phase 4 benchmarks: robustness

Six degradation sweeps made with the benchmark CLI (`--spec` files in this folder). Each folder has `report.md` and
`summary.csv` (nominal level only), `robustness.csv` (every level, every metric: mean, 95% interval, relative to
nominal, breakdown level) and SVG charts.

```bash
python -m rcl.benchmark --spec docs/benchmarks/phase4/cartpole_action_noise.json --out out/cartpole_action_noise
```

Protocol (docs/architecture/phase4-robustness.md): 5 seeds x 5 evaluation episodes per level. Training is on the NOMINAL
environment, then the same trained policy is evaluated at every level with identical evaluation seeds, so levels differ
only by the perturbation. Controllers are configured on the nominal model (LQR sees the nominal linearisation, so
`param_scale` is a model-mismatch test). Windows 11, CPython 3.12, PyBullet 3.2.7, CPU. Intervals are across 5 seeds
(success rate: bootstrap); with 5 seeds, differences between neighbouring levels are not significance claims.

Sweeps: `action_noise` (Gaussian noise, std as a fraction of the actuator limit), `disturbance` (a 5-step pulse at a
random step, magnitude as a fraction of the actuator limit), `param_scale` (multiplier on `masspole` for CartPole,
`chassis_mass` for the balance bot). Controllers: PID, LQR, PPO (CartPole only, 20k steps); balance bot PID uses the
Phase 3 gains. Planar arm is skipped: no controller ever reached its task (Phase 3).

## Success rate by level (mean over 5 seeds of the per-seed fraction of 5 episodes)

| env | controller | action_noise std: 0 / 0.1 / 0.25 / 0.5 / 1.0 | disturbance: 0 / 0.25 / 0.5 / 1.0 | mass scale: 0.5 / 1 / 2 / 4 |
|---|---|---|---|---|
| CartPole | PID | 1.0 / 1.0 / 0.04 / 0 / 0 | 1 / 1 / 1 / 1 | 1 / 1 / 1 / 1 |
| CartPole | LQR | 1.0 / 0.96 / 0.08 / 0 / 0 | 1 / 1 / 1 / 1 | 1 / 1 / 1 / 1 |
| CartPole | PPO | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| Balance bot | PID | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / **1 / 1** |
| Balance bot | LQR | 1.0 / **0** / 0 / 0 / 0 | 1 / 1 / 0.96 / 0 | 1 / 1 / 1 / **0** |

Failure rate (episodes that terminated) tells the same story: CartPole PID fails 8% / 32% / 92% of episodes at noise 0.25 / 0.5 / 1.0 and LQR 0% / 4% / 100%
(success collapses earlier than failure: noisy episodes survive but leave the tolerance band); the balance-bot LQR fails in 100% of episodes at disturbance 1.0 and mass x4; balance-bot PID fails
12% at noise 0.5 and 100% at 1.0. See `robustness.csv` for the numbers and intervals.

## What this shows, and what it does not

- **Action noise is the harshest perturbation for the working controllers**: CartPole PID and LQR go from ~100% to
  ~0-8% success between noise 0.1 and 0.25 of the actuator limit (breakdown level 0.25 for both).
- **Pulse disturbances are absorbed up to the actuator limit** on CartPole (no failures at 1.0 for PID or LQR) because
  the disturbance saturates at the limit (the force is added to the clipped command and the environment clips the sum;
  there is no external-force hook). The balance-bot LQR holds up to 0.5 and fails at 1.0. This understates a true
  external push.
- **Mass mismatch**: CartPole is insensitive to pole mass x0.5 to x4 for both controllers; the balance-bot LQR breaks
  at chassis mass x4 (breakdown level 4.0) while its PID does not fail.
- **Unexpected, reported as measured:** (1) the balance-bot LQR fails in every episode already at noise 0.1 of the
  1 N*m torque limit, much harsher than on CartPole; I did not investigate whether this is the robot, the LQR gains or
  the noise model. (2) The balance-bot PID never reaches the 0.02 rad success band at nominal (Phase 3: it balances
  but outside the band) yet succeeds at chassis mass x2 and x4, so its "success" curve rises with the perturbation;
  that is a nominal-tuning effect, not robustness. Its breakdown level is undefined.
- **PPO is not informative.** The 20k-step CartPole PPO never succeeds and fails every episode at every level, including
  nominal, so there is no degradation to measure. This benchmark gives no evidence about RL robustness.
- **`recovery_time` is not usable on these tasks.** It is defined with a 2% band of `max(S, step_floor)` = 1 mrad
  here, while success only needs 0.02 rad and success ends the episode, so nearly every recovery is censored at
  `T_max - t_p` (about 8.8 s for LQR, a bit under 2 s for CartPole PID). The column in `robustness.csv` reflects that
  definition, not recovery quality. Raised in `docs/OPEN_QUESTIONS.md`.
- **Limits:** one parameter value per run (no per-episode randomisation); no PyBullet run mixes perturbations; one
  perturbation per sweep; 5 seeds; one trained policy per (controller, seed) so the trained RL result is a single
  short run. The disturbance saturation above also applies to the balance bot.
