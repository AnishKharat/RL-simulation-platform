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

## Robot arm (`robot_arm`)

KUKA iiwa (7 revolute joints) from `pybullet_data` (shipped inside the PyBullet package, no repo assets). It reaches a
3-D target with torque control. The plant includes gravity compensation computed by the environment (like a real
arm's firmware), so actions are residual torques. This is documented as part of the robot, not the controller.

| Aspect | Specification |
|---|---|
| True state | `[q(7), q_dot(7), ee_err(3)]` with `ee_err = ee_pos - target` (C1). `dof = 7`, `state_dim = 17`. |
| Observation | Same 17-vector, `Box(17,)` float64. The target enters only through `ee_err`, so a per-episode target needs no `TaskSpec` change. |
| Action | `Box(7,)` residual joint torques, clipped per joint to `max_torque[i]`. `action_dim = 7`. |
| Timing | Physics 1/240 s, control `dt = 0.02` s, horizon 250 steps (5 s). |
| Initial state / target | `q` near a home pose + U(-0.1, 0.1); target sampled from `np_random` in a reachable box in front of the robot. Both are seeded per episode (same initial states and targets for every controller, fairness). |
| Reward | `1 - (norm(ee_err)/0.5)^2 - 0.0005 norm(u)^2`. |
| Termination (failure) | None in normal operation (an arm does not "fall"); a non-finite state terminates. Truncation at the horizon. |
| Success (TaskSpec) | `setpoint = 0` for `ee_err`, tolerance 0.02 m per axis (joints 1e9), held 25 steps; `error_state_indices = (14, 15, 16)`, `output_unit = "m"`, `step_floor = 0.05`. |
| Linearisable | No in Phase 3 (`linearizable = False`), so LQR is hidden. Possible later about a fixed target. |
| Controllers | PPO and SAC, with no code change (exit criterion). PID is hidden by `action_dim = 7`, by design. |
| Metrics | `angle_error` = RMS end-effector distance in m (name is historical, unit correct). Settling time, rise time and overshoot are meaningful because the initial distance exceeds `step_floor`. Steady-state error, effort and energy all apply. |

## Rendering
`render_mode="rgb_array"` maps to `PhysicsBackend.render`. Rendering is never used inside benchmark runs (it must not
change state and costs time). A dashboard or video export is a separate ROADMAP item.

## Open questions (also in `docs/OPEN_QUESTIONS.md`)
1. Arm model: KUKA iiwa 7-DOF, which is realistic but needs large RL budgets, or a simpler 2-3 DOF planar arm, which
   learns in minutes?
2. Asset files: is a robot allowed to ship a URDF next to its one `.py` file (`rcl/environments/assets/`), or must
   models be built in code or come from `pybullet_data`? The specs above avoid repo assets.
