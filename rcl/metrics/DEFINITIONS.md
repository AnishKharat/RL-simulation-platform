# Metric Definitions

Each metric is a pure function `metric(traj: EpisodeTrajectory, task: TaskSpec) -> MetricValue` with no I/O, no global state and no RNG. Source rule: project_context §6.7. Anything not defined here is not a metric.

## Inputs and notation

An `EpisodeTrajectory` with `T = n_steps` steps at fixed step `dt` (s) has T+1 state samples `x_0..x_T` (true state, `env.get_state()`; `x_0` is the post-reset state) at times `t_k = k*dt`, and T transitions: applied action `u_k` (physical units; discrete actions are mapped to their applied value by the environment) and reward `r_k` for `k = 0..T-1`, where `u_k` is applied at `x_k` and produces `x_{k+1}`. Flags: `terminated`, `truncated`. **Indexing rule:** error, progress and settling quantities range over the state samples `k = 0..T`; sums over actuation or reward (`u_k`, `r_k`) range over `k < T` (k = 0..T-1). `TaskSpec` supplies:

- `y(x) = x[error_state_indices]`: the controlled output vector (at least one component, e.g. pole angle in rad) and `y_ref = setpoint[error_state_indices]`: its reference (derived, not stored separately).
- `wrap_indices` (subset of `error_state_indices`): output components that are angles.
- `T_max = horizon_steps * dt` (s; there is no separate time-limit field).
- `step_floor` (`S_min`): smallest step size treated as a real step response.

Per-component difference `d_k = wrap(y(x_k) - y_ref)`, where `wrap` maps the components in `wrap_indices` to (-pi, pi] and is the identity for the others. Error `e_k = ||d_k||_2` (Euclidean norm; for one component, `|y - y_ref|`). Step size `S = ||wrap(y_ref - y(x_0))||_2`. Episode duration `T*dt` (not to be confused with `T_max`).

A metric returns `(value, status, unit)`. `status` is `ok`, `censored` (value is a defined bound, e.g. never settled) or `undefined` (value is NaN).

## Phase 1 metrics

| Metric | Definition | Unit |
|---|---|---|
| **return** | `sum_{k=0}^{T-1} r_k` (undiscounted). Episode length `T` is reported separately as `n_steps`. | reward |
| **angle_error** | RMS error `sqrt( (1/(T+1)) * sum_{k=0}^{T} e_k^2 )`. `max_abs_angle_error = max_{k=0..T} e_k` is a companion value. | rad |

An episode terminated early by failure keeps the metrics of its `T` recorded steps; the termination is visible through `terminated` and (Phase 4) failure count, so truncation of the sum is not hidden.

## Phase 2 metrics

All use the error signal above. Define the normalized step progress `p_k = <w_k, w_ref> / S^2` with `w_k = wrap(y(x_k) - y(x_0))` and `w_ref = wrap(y_ref - y(x_0))` (0 at start, 1 at the setpoint, the projection of the motion onto the commanded step), only when `S >= S_min`. For one component this equals `(y(x_k) - y(x_0)) / (y_ref - y(x_0))`.

**settling_time (2% band).** Band half-width `b = 0.02 * max(S, S_min)`. Settling index `k_s = min { k in 0..T : e_j <= b for all j in k..T }`. Value `t_{k_s}`.
- Already inside the band for the whole episode: `0`.
- Never settles (`e_T > b`, so no such `k` exists), including episodes that terminate by failure: value `T_max`, status `censored`. The companion boolean `settled = false`.
- The `S_min` floor makes the band meaningful for balance tasks that start near the setpoint (band becomes `0.02 * S_min`).

**rise_time (10-90%).** Over `k = 0..T`. Defined only if `S >= S_min`; otherwise `undefined`. `t_10` and `t_90` are the first times `p_k` reaches 0.1 and 0.9, with linear interpolation between samples. Value `t_90 - t_10`.
- `p` never reaches 0.9: value `T_max - t_10` if 0.1 was reached, else `T_max`; status `censored`.
- Already beyond 0.1 at `k = 0`: `t_10 = 0`.

**overshoot.** Defined only if `S >= S_min`. `100 * max(0, max_k p_k - 1)`, i.e. percent of the step size beyond the setpoint. Value `0` if the setpoint is never exceeded; `undefined` if `S < S_min`.

**steady_state_error.** Mean error over the final window `W` = last `max(1, ceil(0.1*(T+1)))` state samples (indices `k <= T`): `(1/|W|) * sum_{k in W} e_k`. Unit follows `y`. For episodes terminated early the window is the last 10% of recorded samples and status is `censored` (the system did not reach steady state).

**energy_usage.** Actuation-energy proxy `sum_{k=0}^{T-1} ||u_k||_2^2 * dt` (action-unit^2 s, proportional to electrical energy `I^2 R t`). It is a proxy, not mechanical work; true mechanical energy `sum |u.v| dt` needs actuator velocity and is deferred until `TaskSpec` exposes it.

**control_effort.** `sum_{k=0}^{T-1} ||u_k||_2 * dt` (action-unit s), the time-integrated action magnitude. Also reported: `control_effort_rate = sum_{k=1}^{T-1} ||u_k - u_{k-1}||_2` (smoothness; 0 when T < 2), same units without `dt`.

## Phase 4 metrics (future, not implemented before Phase 4)

- **failure_count:** number of episodes in a perturbation suite with `terminated = true` (task failure per `TaskSpec`), reported as a count and as a rate over episodes.
- **recovery_time:** after a perturbation applied at `t_p`, the time until `e` re-enters and stays in the 2% band: `t_{k_r} - t_p`, with `k_r` defined as in settling_time restricted to `k >= index(t_p)`. Never recovers: `T_max - t_p`, `censored`.
- **degradation curve:** a metric value as a function of perturbation magnitude; not a single-episode metric.

## NaN and edge-case policy

1. A trajectory with any non-finite state, action or reward is invalid: every metric returns `value = NaN`, `status = undefined`, and the episode counts as a failure in bookkeeping. Metrics never raise on well-formed but degenerate input.
2. **T = 0 rule:** `T = 0` (no step taken) is a programming error and every metric raises `ValueError`; the runner never emits it. (`x_0` alone is not an episode.)
3. NaN is never converted to 0 and never silently dropped. Storage keeps `value = NULL`, `status = undefined` (see `docs/architecture/data-model.md`).
4. `censored` values are real numbers and are included in aggregates, with the censoring count reported alongside. This biases time metrics toward `T_max` rather than hiding failures.

## Aggregation across seeds

- The unit of replication is the **seed** (run). Per-seed value = mean of the metric over that run's evaluation episodes, so confidence intervals do not treat episodes of one policy as independent samples of training randomness.
- `undefined` runs are excluded from the mean; the table always reports `n_valid` and `n_total`. If `n_valid < 2` the CI is NaN.
- **Default: two-sided 95% Student t-interval**, `mean +/- t_{0.975, n-1} * s / sqrt(n)`, with `s` the sample standard deviation (ddof = 1). Used for return, angle_error, steady_state_error, energy_usage, control_effort, overshoot.
- **Percentile bootstrap (10,000 resamples, 95%, seeded RNG)** for settling_time and rise_time, whose censored, skewed distributions violate the t-interval assumptions, and, for any metric, a `low_n` warning flag is set when `n_valid < 5`. The method used is recorded in `ci_method`.
- Comparisons between controllers use the same seeds (paired), and report per-seed differences with the same interval rules. No significance claim is made from `n < 5`.
