# Robotics Control Lab (RCL) — Project Context

> Source of truth for all agents (orchestrator, architect, implementer, reviewer).
> Read this fully before doing anything. Read `CLAUDE.md` for working rules and the git protocol.

---

## 1. Vision

RCL is a **modular robotics simulation, control, reinforcement learning and benchmarking platform**.

A user can:
1. Select a robotic system (environment)
2. Select a control strategy (controller)
3. Train or configure the controller
4. Run simulations
5. Compare performance across controllers
6. Analyze metrics and visualize behavior

Supported controller families:
- **Classical:** PID, LQR, MPC (future)
- **RL:** PPO, SAC, TD3, A2C, DQN, future algorithms

### This project is NOT
- A CartPole demo
- A single RL project
- A toy simulator

### This project IS
A **robotics control framework** and benchmarking platform, eventually supporting simulated robots, real robots (ROS2), sim-to-real transfer, robotics education and research experimentation.

---

## 2. Resume / Portfolio Goals

The project should demonstrate: software architecture, AI engineering, reinforcement learning, robotics, control systems, simulation, experiment design and performance evaluation. It should look closer to a research platform than a student assignment.

---

## 3. Architectural Principles (non-negotiable)

| # | Principle | Consequence |
|---|-----------|-------------|
| 1 | **Environment and controller are independent** | Controllers only see the `BaseEnvironment` interface, never a concrete robot. |
| 2 | **Frontend has zero environment-specific logic** | Dropdowns, parameter forms and plots are generated from registry metadata. |
| 3 | **New robot = 1 file + 1 registry entry** | No edits to the experiment manager, API, metrics or frontend. |
| 4 | **Benchmarking is first-class** | Every run produces standardized, comparable, reproducible metrics. |
| 5 | **Reproducibility by default** | Every run stores seed, config, code version, and library versions. |
| 6 | **Layers depend downward only** | Frontend → API → Experiment Manager → Controllers/Environments → Physics. No upward imports. |

---

## 4. Modularity Map

Each module has one responsibility, a defined public interface, and its own tests. Modules communicate only through those interfaces.

```
rcl/
├── core/            # Interfaces & types ONLY (no heavy deps). BaseEnvironment, BaseController, specs, configs
├── registry/        # Environment + controller registries, discovery, metadata
├── environments/    # One file per robot. Each self-registers.
│   ├── cartpole.py
│   ├── pendulum.py
│   └── (later) balance_bot.py, robot_arm.py, drone.py
├── controllers/
│   ├── classical/   # pid.py, lqr.py, (mpc.py)
│   └── rl/          # ppo.py, sac.py, td3.py, dqn.py (thin wrappers over Stable Baselines3)
├── experiments/     # Experiment Manager: run creation, execution, seeding, artifact saving
├── metrics/         # Metrics Engine: pure functions over episode trajectories
├── analytics/       # Cross-run comparison, tables, reports
├── persistence/     # DB models + repositories (SQLite → PostgreSQL via SQLAlchemy)
├── api/             # FastAPI backend (thin; delegates to experiments/analytics)
├── frontend/        # Streamlit (Phase 1–4) → Next.js (Phase 5)
├── perturbations/   # (Phase 4) disturbances, noise, wind, mass variation as env wrappers
├── ros_adapter/     # (Phase 7) ROS2 bridge
└── tests/
```

**Dependency rules**
- `core` imports nothing from the project.
- `environments` and `controllers` import only `core` and `registry` (the registration decorator) (+ third-party).
- `registry` imports only `core`; it loads environments/controllers by name (discovery), never statically.
- `experiments` imports `core`, `registry`, `metrics`, `persistence`.
- `api` imports `experiments`, `analytics`, `registry`, `persistence`.
- `frontend` talks to `api` only (HTTP), never imports backend internals.
- Enforce with an import-linter test (see Phase 0).

---

## 5. Core Interfaces (proposed — refine in Phase 0)

### 5.1 BaseEnvironment
Follow the **Gymnasium API** so SB3 and the ecosystem work out of the box.

```python
class BaseEnvironment(gymnasium.Env):
    metadata: EnvMetadata            # name, description, dof, obs/action spec, default params, physics backend

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Obs, Info]
    def step(self, action) -> tuple[Obs, float, bool, bool, Info]   # obs, reward, terminated, truncated, info
    def render(self) -> RenderOutput | None
    def close(self) -> None
    def get_observation_space(self) -> gymnasium.Space
    def get_action_space(self) -> gymnasium.Space

    # Needed for benchmarking and classical control:
    def get_state(self) -> np.ndarray            # true state (for metrics, LQR)
    def get_task_spec(self) -> TaskSpec          # setpoint, tolerances, what "success" and "failure" mean
    def get_linearization(self) -> LinearModel | None   # optional, for LQR/MPC
```

### 5.2 BaseController
```python
class BaseController(ABC):
    metadata: ControllerMetadata     # name, family (classical|rl), supported action spaces, hyperparameter schema
    requires_training: bool

    def configure(self, env_spec: EnvSpec, params: dict) -> None
    def predict(self, obs, deterministic: bool = True) -> Action
    def train(self, env, config: TrainConfig) -> TrainResult      # no-op for classical
    def evaluate(self, env, n_episodes: int, seed: int) -> list[EpisodeTrajectory]
    def save(self, path) -> None
    def load(self, path) -> None
```

### 5.3 Registry
- Decorator-based self-registration: `@register_environment("cartpole")`, `@register_controller("pid")`.
- Registry returns **metadata + a Pydantic parameter schema** per entry, so the frontend renders forms dynamically.
- Compatibility is declared, not hardcoded: a controller declares supported action-space types (`discrete`, `continuous`) and the registry filters the dropdown accordingly.
- Discovery via package scanning (`pkgutil`) so adding a file is enough ("1 file + 1 registry entry" — the entry being the decorator).

### 5.4 Experiment config (single source of truth)
```yaml
schema_version: 1
name: pid_vs_ppo_cartpole
seed: 42                                   # default root seed; each Run may override it
environment:   { id: cartpole, params: {...} }
controller:    { id: pid, params: {...} }
perturbations: []                          # optional; [{ id, params, apply_to: train|eval|both }]
training:      { total_timesteps: 100000 } # RL only
evaluation:    { episodes: 20 }
```
Implemented as `rcl.core.config.ExperimentConfig` (flat, no `experiment:` wrapper). Validated with Pydantic, stored verbatim (params resolved) in the DB, and sufficient to reproduce a run. `config_hash` excludes `seed` and `name` (and an empty `perturbations`). Fairness: same eval seeds across controllers for a run seed; training env uses the same wrapper stack as evaluation.

---

## 6. Known Weaknesses in the Original Design → Resolutions

1. **Discrete vs continuous mismatch.** Gymnasium CartPole is discrete; PID/LQR output continuous force. → Provide a continuous-action CartPole variant (own implementation), and let controllers declare supported action spaces.
2. **"Inverted Pendulum" is ambiguous.** → Define two tasks explicitly: *balance* (start near upright) and *swing-up* (start hanging). Both live in `pendulum.py` as task variants.
3. **Controller interface assumes training everywhere.** → `requires_training` flag; classical `train()` is a no-op.
4. **Classical control needs model access.** → `get_linearization()` and `get_state()` on the environment (optional capability).
5. **Streamlit calling backend code directly breaks Principle 2 / Phase 5 migration.** → Introduce the FastAPI layer in Phase 1 (thin), so Streamlit and later Next.js use the same API.
6. **MLflow and SQLite roles overlap.** → MLflow = training/run artifacts and curves; DB = experiment catalog, benchmark results and comparisons. Use SQLAlchemy from day one so SQLite → PostgreSQL is a config change.
7. **Metrics defined too loosely.** → Metrics are pure functions over an `EpisodeTrajectory`, each with a precise written definition (e.g., settling time = first time after which |error| stays within 2% band), in `metrics/DEFINITIONS.md`.
8. **Fairness of comparison.** → Benchmark protocol: same seeds, same episode count, same initial-state distribution, same time limit for every controller.
9. **Long RL training blocks the API.** → Run experiments as background jobs with a status field (queued/running/done/failed); simple in-process worker first, queue later.
10. **Perturbations bolted on late.** → Design perturbations as **environment wrappers** from the start (Phase 4 just adds them).

---

## 7. Phases (with modularity deliverables)

> **Phases 1–3 determine ~80% of future extensibility. Architect agents must iterate on them multiple times before implementation starts.**
> A phase is only "done" when its exit criteria pass, its docs are updated, and it is tagged in git (`phase-N-complete`).

### Phase 0 — Research & Design (architecture freeze)
**Deliverables**
- Architecture diagrams (component, sequence for "run experiment", data model) in `docs/architecture/`
- Interface definitions (`core/`) as stubs with docstrings
- Registry design doc
- Experiment config schema
- Metric definitions doc
- Technology validation spikes (SB3 + custom env works; FastAPI + background job works; registry auto-discovery works)
- ADRs (Architecture Decision Records) in `docs/adr/`
- Import-linter config enforcing dependency rules

**Exit criteria:** architecture reviewed at least 3 iterations; interfaces frozen; ADR for every major choice.

### Phase 1 — MVP (end-to-end vertical slice)
- **Environments:** CartPole (continuous + discrete), Inverted Pendulum (balance)
- **Controllers:** PID, PPO
- **Frontend:** Streamlit (dropdowns built from API/registry)
- **Backend:** FastAPI (`/environments`, `/controllers`, `/experiments`, `/experiments/{id}`)
- **Metrics:** Reward, Angle Error
- **Persistence:** SQLite via SQLAlchemy
- **Modularity checks:** adding a dummy third environment requires only one new file; frontend unchanged.

**Exit criteria:** user picks env + controller in the dashboard, runs it, sees the plot and metrics; reproducible with same seed.

### Phase 2 — Benchmarking Platform
- **Add controllers:** LQR, SAC
- **Add metrics:** Settling Time, Rise Time, Overshoot, Steady-State Error, Energy Usage, Control Effort
- **Add:** benchmark runner (N controllers × M environments × K seeds), comparison dashboard, summary tables, MLflow integration
- **Modularity checks:** new controller added without touching existing controller or environment code.

**Exit criteria:** one command/button produces a PID vs LQR vs PPO vs SAC comparison table and plots, with confidence intervals across seeds.

### Phase 3 — Advanced Robotics (PyBullet)
- **Environments:** Self-Balancing Robot, Robot Arm
- **Physics backend abstraction:** `PhysicsBackend` interface so environments can use Gymnasium-native, PyBullet (and later MuJoCo) without controller changes
- **Add:** richer rendering/visualization, multi-DOF action spaces
- **Modularity checks:** existing controllers (PPO/SAC) run on new robots with zero code changes.

**Exit criteria:** at least two PyBullet robots benchmarked by ≥3 controllers.

### Phase 4 — Research Features (Robustness)
- Disturbances, sensor noise, wind, mass variation (as env wrappers in `perturbations/`)
- Robustness metrics: Failure Count, Recovery Time, performance degradation curves
- Robustness benchmark suite

### Phase 5 — Professional Platform
- Next.js + TypeScript frontend (consumes the existing API)
- PostgreSQL, MLflow server, Docker Compose deployment, CI/CD
- API versioning, authentication (basic)

### Phase 6 — Research-Grade Robotics
- MuJoCo backend via `PhysicsBackend`
- Advanced robot models, higher-fidelity simulation, MPC controller

### Phase 7 — Sim-to-Real
- ROS2 adapter implementing the same `BaseEnvironment` interface for real hardware
- Policy export/deployment, safety limits layer
- Architecture: Simulation → Policy → ROS2 → Robot

### Future Research Directions
Domain randomization, sim-to-real transfer, adaptive control, meta-RL, multi-agent robotics, hierarchical RL, safe RL, imitation learning, world models, MPC.

---

## 8. Technology Stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | |
| Backend | FastAPI, Pydantic v2, NumPy, SciPy | |
| RL | Stable-Baselines3 | PPO, SAC, TD3, A2C, DQN |
| Sim | Gymnasium → PyBullet → MuJoCo → (Isaac Sim) | via `PhysicsBackend` |
| Frontend | Streamlit → Next.js + TypeScript | |
| Viz | Plotly | |
| Tracking | MLflow | |
| DB | SQLite → PostgreSQL | SQLAlchemy + Alembic |
| Quality | pytest, ruff, mypy, import-linter, pre-commit | |
| CI | GitHub Actions | lint + tests on every push |

---

## 9. Success Criteria

1. New environments added in under one hour (1 file).
2. New controllers added without modifying existing code.
3. Controllers benchmarked automatically with one command.
4. Experiments are reproducible (seed + config + versions stored).
5. Platform can eventually connect to ROS2 and real robots.
6. Demonstrates competence in robotics, RL, control theory, architecture and AI engineering.

---

## 10. Agent Workflow

### 10.1 Architecture review (before any implementation)
1. Review this document and identify weaknesses.
2. Suggest improvements; refine interfaces, registry design and experiment workflow.
3. Validate technology choices with small spikes.
4. Produce architecture diagrams and an implementation roadmap.
5. **Iterate at least 3 times on Phases 1–3**, recording each iteration as an ADR or a revision note in `docs/architecture/`.
6. Implementation starts only when the architecture is marked **FROZEN** in `docs/architecture/STATUS.md`.

### 10.2 Roles
- **Orchestrator:** owns the roadmap, assigns tasks, tracks phase status, ensures commit protocol is followed.
- **Architect:** owns interfaces, ADRs, diagrams, dependency rules.
- **Implementer:** writes code + tests for one task at a time.
- **Reviewer:** checks modularity rules, tests, docs, and commit hygiene.

### 10.3 Git workflow
Full rules in `CLAUDE.md` §3 (remote: https://github.com/AnishKharat/RL-simulation-platform.git). Per-phase commit checkpoints: `docs/ROADMAP.md`.

### 10.4 Definition of Done (per task)
- Code + tests pass locally
- Lint/type checks pass
- Docs/ADR updated if an interface or decision changed
- Modularity rules respected (import-linter green)
- Committed with a clear message
