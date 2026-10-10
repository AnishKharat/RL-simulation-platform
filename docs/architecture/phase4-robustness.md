# Phase 4 robustness design (review iteration 5)

Status: proposal, awaiting user approval (ADR 0021). Nothing here is implemented.

## 1. Perturbations as wrappers

Two stages, one registry kind `perturbation` (additive, see section 5):

| stage | what | how | examples |
|---|---|---|---|
| `step` | acts during an episode | `EnvWrapper` subclass in `rcl/perturbations/<id>.py` | `obs_noise`, `action_noise`, `action_delay`, `disturbance` (force/wind pulse) |
| `params` | acts on the plant definition | `ParamPerturbation.apply(env_params, params) -> env_params`, run before the env is constructed | `param_scale` (mass, length, ...) |

**Order.** `ExperimentConfig.perturbations` is the wrapping order: the first item wraps the bare
environment, the last is outermost. The action therefore passes last -> first, the observation first -> last.
`params`-stage items are applied first (in list order) and never wrap. Recommended order for a stack:
`[param_scale, disturbance, action_delay, action_noise, obs_noise]` (sensor noise outermost, so the controller
sees noise; actuator effects close to the plant). The order is the user's, recorded in the config.

**Seeding.** A wrapper owns an RNG built in `reset(seed=s)` as
`default_rng(SeedSequence([s, position_in_stack]))`, where `s` is the rollout's per-episode seed (`eval_seed + i`,
ADR 0010). It never touches the global RNG, the environment's RNG or the controller's. Rules that make runs
reproducible and fair: (a) a fixed number of draws per step regardless of action or state; (b) event timing is
step-indexed (e.g. a pulse at step `k_p`, drawn from the RNG at reset), never state-triggered. A wrapper reset
without a seed is an error (`ValueError`), so no perturbation is ever unseeded.

**Truth vs sensors.** `get_state()` stays the true unperturbed state (EnvWrapper contract); metrics are therefore
comparable across levels. `info["applied_action"]` is what reached the plant, including `disturbance`/noise
(contract already says so). `info["perturbed"] = True` on every step at which a discrete event was injected.

**Recording.** `rollout` (ADR 0013, pinned) gains one additive line: collect the step indices where
`info.get("perturbed")` into the existing `EpisodeTrajectory.perturbation_steps`. Nothing else in the loop changes.
The resolved `perturbations` list is stored verbatim in the config JSON (`RUN.config`), part of `config_hash`
(empty list still excluded, so old hashes are unchanged). `schema_version` stays 1: the field already exists and
adding entries is not a schema change. The manager stops answering `unsupported` and builds the stack:
`train` / `eval` / `both` via `apply_to` (default `eval`). The controller is always `configure`d on the NOMINAL
environment (that is the point of a robustness test: model mismatch for LQR, unseen conditions for RL).

## 2. Mass / parameter variation (the deferred hook)

No `BaseEnvironment` or `PhysicsBackend` change. Every environment already takes its physical parameters as
constructor arguments validated by its registered param model (`masscart`, `masspole`, `chassis_mass`, ...).
`param_scale` with params `{targets: [name, ...], scale: float > 0}` multiplies the named numeric params of the
environment's own params (taken from the resolved config), then the environment is constructed with them.
Gymnasium-native envs pick them up directly. PyBullet robots rebuild their world from the constructor
(`BalanceBot._build` reads `self._mass`), so `rcl.physics` is untouched. Names are validated against the
environment's registered params schema (unknown or non-numeric target -> 422), so nothing is hardcoded.
Limits (documented, deferred): one value per run, not re-sampled per episode (domain randomisation would need a
reset-time hook or env rebuild per episode); parameters not exposed as constructor args cannot be varied.

## 3. Robustness metric definitions

Written into `rcl/metrics/DEFINITIONS.md` (Phase 4 section). Summary of the edge-case decisions:
`failure_count` counts every `terminated` episode, including failure before the perturbation; `recovery_time`
is `undefined` when no discrete event was injected in the episode (continuous perturbations, event beyond the
episode end, episode ended before the event) and `censored` (`T_max - t_p`) when the system never re-enters
the band or fails after the event; degradation curves are per-seed values vs magnitude with CIs across seeds.

## 4. Fairness

Within a robustness benchmark every controller sees identical perturbation sequences: the same seeds list, the same
per-episode eval seeds (`derive_seeds`), perturbation RNG a function of `(episode seed, stack position)` only
(draws independent of the controller), time-indexed events, same level grid, same env params. Training happens
ONCE per (environment, controller, seed) on the nominal environment (`apply_to: eval` perturbations only, unless the
user marks `train`/`both`), then the same trained policy is evaluated at every level, via the existing
`save`/`load`. A level-0 (nominal) point is always included so degradation is relative to the same policy.

## 5. API, dashboard and registry impact

- Registry: Kind `"perturbation"`, decorator `register_perturbation(id, params=Model)`, discovery already scans
  `rcl.perturbations`. `PerturbationMetadata(label, stage, magnitude_param, magnitude_unit, nominal_magnitude,
  action_spaces)` in core. `magnitude_param` names the scalar the degradation sweep varies (noise std, delay
  steps, pulse force, `scale`), so the frontend builds sweeps from metadata alone.
- API: `GET /perturbations` (same shape as `/environments`: id, metadata, params_schema);
  `POST /experiments` accepts `perturbations`; `BenchmarkSpec` gains optional
  `perturbation: {id, params, levels: [float]}` (levels replace `params[magnitude_param]`);
  `GET /benchmarks/{id}/robustness?env_id=&metric=` returns degradation curves
  `{controller_id, level, mean, ci_low, ci_high, n_valid}` plus failure counts.
- Dashboard: a Robustness page: pick env, perturbation (from `/perturbations`), levels; plots curves per controller.
- `rcl/perturbations/CLAUDE.md` rule: imports `core` and `registry` only (already in the import-linter plugin tier).

## 6. Not decided here (needs user)

True external force on a body (pitch push on the balance bot, wind on a link) needs an environment hook (an
optional `apply_disturbance`). Proposal: do NOT add it; `disturbance` acts through the actuated input channel
(CartPole cart force, balance-bot wheel torque), saturating at the actuator limit. See OPEN_QUESTIONS.
