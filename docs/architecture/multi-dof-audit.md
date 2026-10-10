# Multi-DOF audit (architecture review 4)

Question: do the frozen types and the Phase 1-2 code handle robots with several actuators and long observation /
state vectors (Phase 3: self-balancing robot, robot arm)? Method: read every place that touches action,
observation, state or output shapes. "Fine" = works unchanged; "change" = needs work before or during Phase 3.

## Frozen core types

| Item | Verdict | Notes |
|---|---|---|
| `EnvSpec.action_shape`, `action_low/high` (length = prod(shape)), `obs_shape` | fine | Already n-dimensional, with validators. |
| `EnvSpec.n_actions` (single `Discrete`) | fine for Phase 3 | `MultiDiscrete` is not representable. Both new robots are continuous; revisit only if a multi-discrete robot appears. |
| `EnvMetadata.dof`, `action_dim`, state/action labels and units | fine | ADR 0014 axes already cover multi-input. Labels give per-axis plot titles. |
| `TaskSpec` multi-output (`error_state_indices`, per-component `tolerance`, `wrap_indices`) | fine | Error is the Euclidean norm over the output components. Task-space goals (arm end-effector) are handled by convention C1 below, with no type change. |
| `TaskSpec` fixed per instance (rollout reads it once per evaluation) | fine with C1 | A target that changes every episode cannot be a different `TaskSpec`. Use C1 (error coordinates) or a target fixed by env params. |
| `EpisodeTrajectory` arrays `(T+1, ...)` / `(T, ...)` | fine | Shape-agnostic; actions stored with their full shape. |
| `LinearModel` (n x n, n x m) | fine | Multi-input B is supported. |

## Code that consumes shapes

| Place | Verdict | Notes |
|---|---|---|
| `experiments.rollout` (records, success check) | fine | Vector tolerance check; actions stacked whatever their shape. |
| `metrics` (`angle_error`, step-response, effort) | fine | Norms over output components and action rows (`reshape(T, -1)`). The name `angle_error` is historical: for a position output its unit comes from `TaskSpec.output_unit` (e.g. m), so values are correct; the label is the only oddity (doc note, no rename). |
| `controllers.classical.pid` | fine (by declaration) | Scalar by design: `max_action_dim=1`, so the registry hides it on multi-input robots. A per-axis PID would be a new one-file controller, not an edit. |
| `controllers.classical.lqr` | fine | Vector gain, multi-input B, per-component clipping. Needs a linearisation (C2). |
| `controllers.rl.ppo` / `sac` | fine | Box actions/observations of any shape (SB3 `MlpPolicy`). Discrete PPO needs a single `Discrete`. |
| `registry` compatibility | fine | Action kind, linearisation and action dimension are already checked. |
| `persistence` (npz trajectories, scalar metric rows, task-spec JSON) | fine | Shape-agnostic. |
| `api` `TrajectoryOut` | fine | `actions` sent as `(T, action_dim)`; labels copied from metadata. |
| `frontend.plots.trajectory_figure` | change (UX, non-frozen) | One panel per state and action component. A 7-joint arm gives about 25 stacked panels. Proposal: group panels by label prefix or a "select signals" widget. Frontend-only. |
| `experiments.manager._LIBS` (versions) | change (non-frozen) | Must add the simulator package (e.g. `pybullet`) so `RunRecord.versions` records it (ADR 0019). |
| Environment modules importing a simulator at top level | change (convention) | `discover()` imports every module, so a missing optional dependency would break discovery for everyone. See ADR 0020. |

## Conventions (no interface change)

- **C1 Task-space outputs live in the true state.** `get_state()` may append derived quantities after the generalised
  coordinates, e.g. end-effector position, or end-effector error `ee - target` when the target is sampled per episode.
  `TaskSpec.error_state_indices` points at them and `state_labels/units` name them. Metrics, success and plots then
  work unchanged, and `get_state()` stays "the true state" because it is computed from simulator truth, not
  observations.
- **C2 Linearisation of simulator robots.** `get_linearization()` is a central-difference Jacobian of the env's own
  control-step map through `PhysicsBackend.set_state`/`step`, in observation coordinates, as CartPole and the
  pendulum already do analytically. `EnvMetadata.linearizable` must match (existing per-env test pattern).
- **C3 Control period vs physics step.** `EnvSpec.dt` = k x `PhysicsBackend.timestep` with integer k fixed by the
  environment.

## Can existing controllers run on new robots with zero code changes?

Yes, with these requirements on the robot (all already in the frozen interfaces):

| Controller | Needs from the robot |
|---|---|
| PPO, SAC | Continuous `Box` action (SAC) or `Box`/`Discrete` (PPO), finite action bounds in `EnvSpec`, a reward, and `TaskSpec` for evaluation. **Phase 3 exit criterion: met by design.** |
| LQR | `linearizable=True` and a stabilisable linearisation about the task equilibrium (C2); continuous actions. |
| PID | `action_dim == 1` (registry-enforced) and an observation index to regulate. Runs on a balance robot driven by one symmetric wheel torque; correctly hidden on the arm. |
