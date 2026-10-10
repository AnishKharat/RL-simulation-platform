# 0021. Perturbation registry kind, wrapper stack and parameter variation

- Status: Accepted (architecture review 5, user approved 2026-10-11)
- Date: 2026-10-11

## Context
Phase 4 needs sensor/actuator noise, delay, disturbances and mass/parameter variation, selectable from the API
and dashboard without hardcoded names. `EnvWrapper`, `PerturbationConfig`, `perturbation_steps` and the
`perturbations` config field exist (reviews 1-2). The registry knows only `environment` and `controller`.

## Decision
1. Add Kind `perturbation` to the registry (additive): `register_perturbation`, `PerturbationMetadata`
   (stage, magnitude_param, magnitude_unit, nominal_magnitude, action_spaces, label) in `rcl.core`.
2. Step perturbations subclass `EnvWrapper`; parameter perturbations subclass a new tiny core ABC
   `ParamPerturbation` (`apply(env_params, params)`), applied before construction. This reaches Gymnasium and
   PyBullet environments through their constructor params; `BaseEnvironment` and `PhysicsBackend` are unchanged.
3. Stack order = list order; wrapper RNG = `SeedSequence([episode_seed, stack_position])`; fixed draws per step;
   time-indexed events; unseeded reset is an error.
4. `rollout` records `info["perturbed"]` steps into `EpisodeTrajectory.perturbation_steps` (additive).
5. Controllers are configured on the nominal environment; robustness suites train once and evaluate at all levels.

## Consequences
Frozen items touched, all additively: registry Kind literal, core gains two types, rollout reads one info key,
manager drops the `unsupported` error, `BenchmarkSpec` and the API gain optional fields. No existing
signature changes; existing configs, hashes and tests are unaffected. Rejected: a wrapper that rebuilds its inner
environment (hidden construction), and adding `set_mass` to `BaseEnvironment` (every robot would implement it).
