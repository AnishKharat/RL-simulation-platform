# 0005. Metrics as pure functions over EpisodeTrajectory

- Status: Accepted
- Date: 2026-10-10

## Context
Controllers are compared by numbers, so definitions must be unambiguous, recomputable, and independent of how a
run was produced. Full definitions: `rcl/metrics/DEFINITIONS.md`.

## Decision
- Every metric is a pure function `metric(traj: EpisodeTrajectory, task: TaskSpec) -> MetricValue`: no I/O, no
  global state, no RNG. Anything not in DEFINITIONS.md is not a metric.
- `MetricValue = (value, status, unit)` with status `ok | censored | undefined`. Non-finite input gives NaN /
  `undefined` and never raises; `T = 0` (no steps) raises `ValueError`. NaN is never coerced to 0 or dropped.
- Settling-band definition: band half-width `b = 0.02 * max(S, S_min)` where `S = |y_ref - y(x_0)|` and `S_min`
  is `TaskSpec.step_floor`; settling index is the first `k` after which `|e_j| <= b` for all `j >= k`. Never
  settling yields `T_max` with `censored`. The floor keeps the band meaningful for balance tasks that start at the
  setpoint. Signal `y = state[error_state_indices]`, `y_ref` come from `TaskSpec`.
- Aggregation unit is the seed (run), not the episode. Default CI: 95% Student t; percentile bootstrap (10,000
  resamples, seeded) for settling/rise time; `low_n` flag when fewer than 5 valid seeds.
- Seed derivation (config contract, `ExperimentConfig`): one root `seed`; env, train and eval seeds are spawned
  from `numpy.random.SeedSequence(seed)`; evaluation episode `i` resets the env with `eval_seed + i`, so every
  controller sees the same initial states.

## Consequences
- Metrics can be recomputed from stored trajectories when a definition changes (ADR 0008 stores trajectories as
  artifacts).
- Censored values enter aggregates (biasing toward `T_max`) rather than hiding failures; counts are reported.
- `TaskSpec` carries metric-relevant fields, so adding a robot requires no metric code.

## Alternatives
- Metrics computed online inside the env/runner: not recomputable, couples definitions to execution.
- Dropping non-settled episodes: survivorship bias.
- Episode-level CIs: treat one policy's episodes as independent training samples.

## Revisions (architecture review 1)
- `TaskSpec`: `time_limit_s` removed (T_max = `horizon_steps * dt`); `y_ref` is now the derived property `setpoint[error_state_index]` instead of a stored field; `angle_wrap: bool = False` added; tolerance/setpoint lengths and index range are validated.

## Revisions (architecture review 2)
- Multi-output tasks: `TaskSpec.error_state_index` / `angle_wrap` replaced by `error_state_indices: tuple[int, ...]` (min 1, unique, in range) and `wrap_indices` (subset of those, the angular components). `y_ref` is now a vector property. Error `e_k` is the Euclidean norm of the per-component wrapped differences; step size `S = ||wrap(y_ref - y_0)||`; step progress `p_k = <y_k - y_0, y_ref - y_0> / S^2`. A single index reduces exactly to the previous scalar definitions.
- Indexing aligned with `EpisodeTrajectory`: T = `n_steps`, states `x_0..x_T`, actions/rewards `k = 0..T-1`; error, progress and settling run over `k = 0..T`, action/reward sums over `k < T` (DEFINITIONS.md).
