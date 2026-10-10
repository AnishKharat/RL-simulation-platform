# Phase 3 robots: specifications (architecture review 4, docs only)

Both robots are PyBullet environments behind `PhysicsBackend` (ADR 0019), headless (`DIRECT`), one file each with one
registry decorator (ADR 0020 covers optional-dependency registration). Conventions C1-C3 refer to
`multi-dof-audit.md`. Numbers are starting values for implementation, exposed as params where sensible.

## Self-balancing robot (`balance_bot`)

Two-wheeled inverted pendulum, balancing in the sagittal plane: both wheels get the same torque. A steering variant
with 2 torques can be a second class later. The model is built in code with `createMultiBody`, so the "one file"
rule needs no asset files (see Open questions).

| Aspect | Specification |
|---|---|
| True state | `[pitch, pitch_rate, x, x_dot]` (rad, rad/s, m, m/s); `x` is the wheel travel. `dof = 2`, `state_dim = 4`. |
| Observation | Same as state, `Box(4,)` float64 (full state, like CartPole). |
| Action | `Box(1,)` symmetric wheel torque, clipped to `[-max_torque, max_torque]` and reported in `info["applied_action"]`. `action_dim = 1`. |
| Timing | Physics step 1/240 s; control `dt = 0.02` s (C3, k = ~5); horizon 500 steps (10 s). |
| Initial state | pitch ~ U(-0.1, 0.1) rad, the rest ~ U(-0.05, 0.05), from `np_random` via `set_state`. |
| Reward | `1 - cost` with `cost = (pitch/0.5)^2 + 0.01 x^2 + 0.001 u^2`. Positive per step, so falling early is never attractive. |
| Termination (failure) | `abs(pitch) > 0.5` rad (fallen). Truncation at the horizon. |
| Success (TaskSpec) | `setpoint = 0`, tolerance `pitch <= 0.02` rad (others 1e9), held 100 steps; `error_state_indices = (0,)`, `wrap_indices = (0,)`, `output_unit = "rad"`, `step_floor = 0.05`. |
| Linearisable | Yes (C2, finite differences through `set_state`). `linearizable = True`. |
| Controllers | PID (scalar action), LQR, PPO, SAC all compatible, with no code change. |
| Metrics | All apply: `angle_error` = RMS pitch error, settling time, steady-state error, effort, energy. Rise time and overshoot are defined only when `abs(pitch0) >= step_floor` and are `undefined` otherwise (DEFINITIONS.md), which is correct for a regulation task. |

## Robot arm (`planar_arm`)

A 2-DOF planar arm (two revolute joints, links 0.5 m and 0.4 m) moving in the **horizontal plane**, so gravity puts
no load on the joints. It reaches a 2-D target with joint torques. The model is built in code with `createMultiBody`;
asset files are allowed but not needed (`rcl/environments/assets/<robot_id>/`, project_context 3). User decision,
review 4: a 2-3 DOF planar arm instead of the KUKA iiwa.

| Aspect | Specification |
|---|---|
| True state | `[q1, q2, q1_dot, q2_dot, ee_err_x, ee_err_y]` with `ee_err = ee_pos - target` (C1). `dof = 2`, `state_dim = 6`. |
| Observation | Same 6-vector, `Box(6,)` float64. The target enters only through `ee_err`, so a per-episode target needs no `TaskSpec` change. |
| Action | `Box(2,)` joint torques, clipped to `[-max_torque, max_torque]` per joint and reported in `info["applied_action"]`. `action_dim = 2`. |
| Timing | Physics 1/240 s, control `dt = 0.02` s, horizon 200 steps (4 s). |
| Initial state / target | `q` ~ U(-pi, pi) per joint, `q_dot = 0`. The target is sampled from `np_random` in the reachable annulus (radius 0.15 to 0.85 m). Both are seeded per episode, so every controller sees the same starts and targets (fairness). |
| Reward | `1 - (norm(ee_err)/0.9)^2 - 0.001 norm(u)^2 - 0.0001 norm(q_dot)^2`. |
| Termination (failure) | None in normal operation; a non-finite state terminates. Truncation at the horizon. |
| Success (TaskSpec) | `setpoint = 0` for `ee_err`, tolerance 0.02 m per axis (joints 1e9), held 25 steps; `error_state_indices = (4, 5)`, `output_unit = "m"`, `step_floor = 0.05`. |
| Linearisable | No in Phase 3 (`linearizable = False`), so LQR is hidden. Possible later about a fixed target configuration. |
| Controllers | PPO and SAC, with no code change (exit criterion). PID is hidden by `action_dim = 2`, by design. |
| Metrics | `angle_error` = RMS end-effector distance in m (name historical, unit correct). Settling time, rise time and overshoot are meaningful because the initial distance usually exceeds `step_floor`. Steady-state error, effort and energy all apply. |

## Rendering
`render_mode="rgb_array"` maps to `PhysicsBackend.render`. Rendering is never used inside benchmark runs (it must not
change state and costs time). A dashboard or video export is a separate ROADMAP item.

## Decisions (user, review 4)
1. Arm: 2-DOF planar arm (above), not the KUKA iiwa.
2. Assets: allowed in `rcl/environments/assets/<robot_id>/`; both robots here are built in code.
