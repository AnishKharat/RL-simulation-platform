# Project Status

> Read this FIRST every session. Keep under ~20 lines. Overwrite, don't append history (git log is the history).

- **Current phase:** 3 (PyBullet robots): tagged `phase-3-complete` (22f8d16, pushed), but the exit criterion is only half met (see below). Architecture FROZEN (re-frozen 2026-10-10 after review 4)
- **Done:** `rcl.physics` + PyBullet backend (ADR 0019), optional `physics`/`render` extras and Linux/3.11 CI job (ADR 0020), `balance_bot`, 2-DOF `planar_arm`, controllers run unchanged on both (test), PyBullet version in run records, trajectory signal selection, GIF CLI `python -m rcl.benchmark.gif`, benchmarks in `docs/benchmarks/phase3/`
- **Exit criterion ("2 PyBullet robots, each benchmarked by >= 3 controllers"):** `balance_bot` MET (pid, lqr, ppo, sac). `planar_arm` NOT MET: only ppo and sac are registry-compatible (PID hidden by action_dim 2, LQR needs a linearisation), and at 10k steps success is ppo 0/25, sac 1/25
- **Remaining (user to decide):** add a third arm-compatible controller (proposal: SB3 TD3 wrapper, one file like SAC; no frozen change), re-run arm benchmarks with a larger RL budget, and decide whether the tag stays or moves after that. Success rate is stored per episode but has no report metric yet
- **Env:** project `.venv` (`--system-site-packages`); PyBullet only in `.venv`; run checks with `.venv/Scripts/python`
- **Checks:** pytest 283 pass (venv) | ruff clean | mypy clean | lint-imports 11 kept | GitHub CI green incl. physics job
- **History check:** no force-push or rewritten pushed history; all 46 pushes to origin/main were fast-forwards (amends were on unpushed local commits)
- **Last updated:** 2026-10-10 by orchestrator
