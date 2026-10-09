# 0001. Gymnasium API as environment base

- Status: Accepted
- Date: 2026-10-10

## Context
Environments must work with Stable-Baselines3 (RL) and with classical controllers, and must be deterministic for
fair comparison. The SB3 spike (`docs/architecture/spike-findings.md` section 1) showed a plain `gymnasium.Env`
trains under PPO unchanged and that the same `obs -> action` callable serves PID and `model.predict`.

## Decision
`rcl.core.BaseEnvironment` subclasses `gymnasium.Env` and keeps its API: `reset(seed=, options=) -> (obs, info)`,
`step(a) -> (obs, reward, terminated, truncated, info)`, `render()`, `close()` (idempotent).
Additional benchmarking hooks are abstract methods: `get_env_spec()`, `get_state()` (true state, not observation),
`get_task_spec()`, and optional `get_linearization()`.
- All randomness comes from `self.np_random`, seeded only via `reset(seed=)`; time is simulated (fixed `dt`).
- `terminated` = failure/natural end; `truncated` = `TaskSpec.horizon_steps` limit; never both.
- Envs are passed to SB3 as instances; no Gymnasium global registration.

## Consequences
- SB3 and any Gymnasium wrapper (`TimeLimit`, perturbation wrappers) work without adapters.
- Core depends on `gymnasium` and `numpy` (third-party only, allowed by the core import rule).
- Observations are float32 at the SB3 boundary while dynamics run in float64; casting is the env's job.
- Windows `SubprocVecEnv` needs importable top-level classes; MVP uses `DummyVecEnv`.

## Alternatives
- Own minimal env interface plus an SB3 adapter: more code, two APIs to keep in sync, loses ecosystem wrappers.
- Gymnasium global registry (`gym.register`): duplicates our own registry (ADR 0003) and hides metadata we need.
- PettingZoo / dm_env: no current multi-agent need; revisit only if one arises.
