# 0013. Rollout lives in the experiments layer, not on BaseController

- Status: Accepted
- Date: 2026-10-10

## Context
`BaseController.evaluate(env, n_episodes, seed)` made every controller implement its own rollout loop. Fair
comparison (same seeds, same stopping rules, same recording of true states) then depends on each implementation
being identical, and RL wrappers would duplicate the loop around `model.predict`.

## Decision
- Remove abstract `evaluate` from `BaseController`. Controllers expose `configure`, `reset` (per-episode state clear, default no-op, called by `rollout` after `env.reset`), `predict`, `train`,
  `save`, `load` only.
- The experiments layer owns one shared function `rollout(env, controller, n_episodes, seed) -> list[EpisodeTrajectory]`.
  Contract: episode `i` calls `env.reset(seed=seed + i)`, then loops `controller.predict(obs)` / `env.step(action)`
  until `terminated or truncated`; states are recorded from `env.get_state()` (true state); `success` is judged
  with `env.get_task_spec()`. It is documented in `docs/architecture/sequence-run-experiment.md`; implementation
  is a Phase 1 task (not implemented in Phase 0).

## Consequences
- One evaluation loop for all controllers; the fair-comparison protocol is enforced in one place.
- Controllers become simpler and need no `BaseEnvironment` knowledge except in `train`.
- `project_context.md` section 5 still lists `evaluate` on `BaseController`; it needs the matching edit.

## Alternatives
- Keep `evaluate` with a default implementation in the base class: invites per-controller overrides that diverge.
