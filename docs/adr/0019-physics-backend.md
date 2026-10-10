# 0019. PhysicsBackend interface and the physics layer

- Status: Proposed (architecture review 4). Interface stub applied (additive); the dependency-rule change below awaits user approval.
- Date: 2026-10-10

## Context
Phase 3 adds PyBullet robots and must allow MuJoCo later "without controller changes" (project_context 7). Principle 6
already draws physics as the bottom layer: Frontend -> API -> Experiment Manager -> Controllers/Environments ->
Physics. Today there is no physics package; CartPole and the pendulum integrate their own analytic dynamics in numpy
(`EnvMetadata.physics_backend = "analytic"`), which stays valid: a backend is optional.

## Decision
1. **Interface** `rcl.core.physics.PhysicsBackend` (a `typing.Protocol`, stub applied, see the code docstring for the
   full contract): `name`, `timestep`, `version()`, `reset()`, `step(n)`, `get_state()` / `set_state()`,
   `apply_joint_torques()`, `render(width, height)`, `close()`.
   - **Ownership:** an environment creates, drives and closes its own backend instance. Controllers, experiments and
     metrics never see it.
   - **No RNG in the backend:** initial conditions come from the environment's `np_random` via `set_state`, so
     ADR 0010 seeding is unchanged.
   - **Determinism:** same initial state, same torques and same step counts give the same states. The claim covers
     the same machine, library version and headless mode only.
   - **Linearisation:** `get_state`/`set_state` make finite-difference `get_linearization` possible for LQR.
2. **Where implementations live (PROPOSED, needs approval):** a new package `rcl.physics` (e.g. `rcl/physics/pybullet.py`)
   as its own layer between the plugin tier and `registry`/`core`: `environments -> physics -> core`.
   - `rcl.physics` imports only `rcl.core` (+ its simulator, lazily). Controllers, perturbations, experiments, api,
     analytics and frontend must not import it.
   - This **changes a frozen rule**: "environments import only `core` and `registry`" becomes "`core`, `registry` and
     `physics`" (CLAUDE.md 1.3, `rcl/environments/CLAUDE.md`, project_context 4, `pyproject.toml` contracts: new
     layer `rcl.physics`, plus a forbidden contract keeping controllers, perturbations and everything above
     experiments away from it).
3. **Backend selection:** one backend per environment class, fixed in code. `EnvMetadata.physics_backend` (existing
   field) names it (`"analytic"`, `"pybullet"`) and stays truthful because it is class-level. A robot on two simulators
   is two registered classes (as `cartpole` / `cartpole_discrete` are two classes today). No backend registry and no
   new global state.
4. **Versions:** the experiment manager adds the simulator package to `RunRecord.versions` (non-frozen code change in
   `rcl.experiments.manager._LIBS`).

## Consequences
- Frozen core types (`BaseEnvironment`, `EnvSpec`, `TaskSpec`, `EpisodeTrajectory`) are unchanged. Backends are an
  implementation detail of the environments that use them.
- Phase 4 mass/friction perturbations will need to reach the backend from an `EnvWrapper`. That would be a new
  optional `BaseEnvironment` hook (a frozen-interface change) and is deliberately deferred to Phase 4's own review.
- Analytic environments keep working untouched.

## Alternatives
- **Backends inside `rcl/environments/_backends/`:** needs no rule change, since the import stays within one
  package. Rejected as the default because it hides a layer that principle 6 names explicitly, and later Phase 7
  hardware and Phase 4 perturbations want a shared, separately-contracted physics package. This is the fallback if
  the rule change is refused.
- **`PhysicsBackend` as an ABC with shared code:** a Protocol keeps `core` free of behaviour (ADR 0004).
- **Selecting the backend per experiment (config field):** would make `EnvMetadata.physics_backend` instance-level and
  add a config axis to every benchmark. Not needed for Phase 3.

## Amendment after the determinism spike (same review)
`spikes/pybullet_determinism/README.md`: seeded runs are bit-identical across fresh clients, `resetSimulation` reuse
and processes, but writing coordinates back (`resetJointState` / `resetBase*`) clears contact and solver caches. The
continuation then diverges from the run the state was read from (max diff 63 under wheel contact), while native
`saveState`/`restoreState` is exact. The stub contract therefore says:
- `set_state` writes a `get_state`-layout vector and clears caches. It is deterministic as a procedure (fine for
  seeded initial states) and NOT an exact continuation.
- `save()` / `restore(handle)` give exact native snapshots.
- Finite-difference linearisation evaluates each perturbation with the same procedure: `restore(base)`,
  `set_state(x +/- eps)`, then step. Contact-rich robots may give noisy Jacobians, so the balance robot's LQR is
  validated at implementation time.
