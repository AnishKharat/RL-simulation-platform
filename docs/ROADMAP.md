# Roadmap

> Read only the current phase's section. Scope and rationale: `project_context.md` §7.
> Each `☐` is a **separate commit** (or small group) when finished. Tick it in the same commit.

| Phase | Goal | Exit criteria | Status |
|-------|------|---------------|--------|
| 0 | Architecture freeze | 3 review iterations, interfaces frozen, ADRs, import-linter | ☑ |
| 1 | MVP vertical slice | Pick env + controller in dashboard, run, see plot + metrics; same seed reproduces | ☐ |
| 2 | Benchmarking | One command gives PID/LQR/PPO/SAC table + plots with CIs across seeds | ☐ |
| 3 | PyBullet robots | ≥2 PyBullet robots benchmarked by ≥3 controllers, no controller changes | ☐ |
| 4 | Robustness | Perturbation wrappers, robustness metrics, robustness suite | ☐ |
| 5 | Professional platform | Next.js UI, Postgres, MLflow server, Docker Compose, CI/CD, basic auth | ☐ |
| 6 | Research-grade | MuJoCo backend, MPC, higher-fidelity models | ☐ |
| 7 | Sim-to-real | ROS2 adapter behind `BaseEnvironment`, policy export, safety layer | ☐ |

Tag each finished phase `git tag -a phase-N-complete -m "Phase N complete: <summary>"`.

---

## Phase 0 — Architecture
- ☑ Repo scaffold: `pyproject.toml`, folder structure, `.gitignore`, pre-commit config
- ☑ CI workflow (lint + tests)
- ☑ Component architecture diagram
- ☑ "Run experiment" sequence diagram
- ☑ Data model diagram
- ☑ `core/` interface stubs (`BaseEnvironment`)
- ☑ `core/` interface stubs (`BaseController`)
- ☑ Spec/config types (`EnvSpec`, `TaskSpec`, `EpisodeTrajectory`, experiment config schema)
- ☑ Registry design doc + stub
- ☑ Metric definitions doc
- ☑ Import-linter rules for dependency layers
- ☑ Spike: SB3 trains on a custom env
- ☑ Spike: FastAPI background job
- ☑ ADRs (one commit each)
- ☑ Architecture review iteration 1 → revisions
- ☑ Architecture review iteration 2 → revisions
- ☑ Architecture review iteration 3 → mark FROZEN
- ☑ Tag `phase-0-complete`

## Phase 1 — MVP
- ☑ Registry implementation + tests
- ☑ Auto-discovery of environments/controllers + tests
- ☑ CartPole (continuous) + tests
- ☑ CartPole (discrete variant) + tests
- ☑ Inverted Pendulum (balance) + tests
- ☐ PID controller + tests
- ☐ PPO wrapper (SB3) + tests
- ☐ Metrics: reward, angle error + tests
- ☐ Experiment Manager (config → run → trajectory) + tests
- ☐ Seeding / reproducibility test
- ☐ Persistence models + repository (SQLAlchemy/SQLite) + tests
- ☐ FastAPI endpoints (registry listing)
- ☐ FastAPI endpoints (create/run/get experiment)
- ☐ Streamlit app skeleton (dropdowns from API)
- ☐ Streamlit: parameter forms from schemas
- ☐ Streamlit: results + Plotly charts
- ☐ Modularity proof (commit as `test(env): prove 1-file extensibility`): a test writes a dummy environment (one file, one decorator) into a temporary package outside `rcl/environments` and calls `discover(packages=[tmp_pkg])` on an isolated `Registry`; at API level, with `TestClient`, the new id must appear in `GET /environments`, `POST /experiments` with a compatible controller must return 202, and the run must reach `done`. No dummy environment is ever added to `rcl/environments`.
- ☐ README quickstart + screenshots
- ☐ Tag `phase-1-complete`

## Phase 2 — Benchmarking
- ☐ LQR controller (+ linearization on environments) + tests
- ☐ SAC wrapper + tests
- ☐ Metrics: settling time, rise time, overshoot, steady-state error (one commit each, with tests)
- ☐ Metrics: energy usage, control effort
- ☐ Benchmark runner (controllers × envs × seeds) + tests
- ☐ Benchmark result persistence
- ☐ Analytics: comparison tables
- ☐ Analytics: statistical summaries / confidence intervals
- ☐ MLflow integration
- ☐ Comparison dashboard page
- ☐ Report export (Markdown/HTML)
- ☐ Tag `phase-2-complete`

## Phase 3 — Advanced Robotics
- ☐ `PhysicsBackend` abstraction + tests
- ☐ PyBullet backend
- ☐ Self-balancing robot environment + tests
- ☐ Robot arm environment + tests
- ☐ Rendering/visualization support
- ☐ Verify existing controllers run unchanged on new robots (test)
- ☐ Benchmark results on new robots + docs
- ☐ Tag `phase-3-complete`

## Phases 4–7
Break each deliverable (see `project_context.md` §7) into commit-sized steps: interface/ADR → implementation → tests → docs → tag.
