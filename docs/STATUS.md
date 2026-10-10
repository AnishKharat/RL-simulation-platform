# Project Status

> Read this FIRST every session. Keep under ~20 lines. Overwrite, don't append history (git log is the history).

- **Current phase:** 3 tagged `phase-3-complete` (22f8d16; tag deliberately left in place). Follow-up branch `phase-3/arm-followup` merged. No Phase 4 work started. Architecture FROZEN
- **Follow-up done:** TD3 wrapper (`rcl/controllers/rl/td3.py`, exploration-noise param), headline metrics `success_rate` and `time_to_success` (reports and dashboard lead with them; `return` is flagged as not the headline), 30k-step arm benchmark PPO/SAC/TD3 x 5 seeds (~67 min), honest Phase 3 benchmark README
- **Exit criterion ("2 PyBullet robots, each by >= 3 controllers"):** met BY COUNT ONLY. `balance_bot`: 4 controllers, LQR succeeds 25/25. `planar_arm`: 3 controllers (ppo, sac, td3) but **0/75 episodes succeed**; only SAC clearly beats the zero-torque baseline (0.47 vs 0.89 m RMS error)
- **Remaining (user to decide):** make an arm controller actually reach the target (more steps, reward shaping or tuning; not attempted), or accept as is. Phase 4 is next only when you say so
- **Env:** project `.venv` (`--system-site-packages`); PyBullet only in `.venv`; checks via `.venv/Scripts/python`
- **Checks:** pytest 299 pass (venv) | ruff clean | mypy clean | lint-imports 11 kept
- **Last updated:** 2026-10-10 by orchestrator
