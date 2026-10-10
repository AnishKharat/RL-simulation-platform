# RCL

Robotics Control Lab: a modular platform to run, compare and benchmark classical and RL
controllers on simulated robots, with reproducible experiments. See `project_context.md`.

## Quickstart

```bash
pip install -e ".[dev]"                          # Python >= 3.11; extras: rl, api, frontend
uvicorn rcl.api.main:app                         # terminal 1: API on http://localhost:8000
streamlit run rcl/frontend/app.py                # terminal 2: dashboard on http://localhost:8501
```

In the dashboard: pick an environment, pick a controller (the list only shows compatible ones),
adjust parameters (forms are generated from each component's schema), press **Run**. When the run
finishes you get the metric table, the training curve (RL) and a per-episode trajectory plot.
The same seed always reproduces the same result (CPU, same library versions).

Notes:
- PID regulates one observation component (`obs_index`) with `e = obs[obs_index] - setpoint`. The
  defaults suit CartPole; for the torque-driven Inverted Pendulum use negative gains, e.g.
  `kp=-30, kd=-6, obs_index=0`.
- The API stores runs in `rcl.db` and trajectories under `artifacts/` (both git-ignored). Override
  with `RCL_DB_URL` and `RCL_ARTIFACTS`; point the dashboard elsewhere with `RCL_API_URL`.
- API reference: `docs/architecture/api.md`; interactive docs at `http://localhost:8000/docs`.

## Benchmark in one command

```bash
python -m rcl.benchmark --envs cartpole --controllers pid lqr ppo sac --seeds 5 --episodes 5     --timesteps 20000 --out report/
```

Runs every compatible (environment, controller) pair for seeds 0..4 under one fairness protocol
(same seeds, episodes, training budget and environment definition for every controller) and writes
`report/report.md`, `report/report.html`, `report/summary.csv` and `report/plots/*.svg`: means with
95% intervals across seeds, per metric. The same seeds reproduce the same files byte for byte (CPU,
same library versions). Omit `--envs` / `--controllers` to use everything registered; pass `--spec
spec.json` for per-controller parameters. The dashboard's **Benchmark** page does the same
interactively. Optional: `--mlflow-uri` (or `RCL_MLFLOW_URI` for the API) logs training curves and
checkpoints to MLflow.

## PyBullet robots (optional)

`balance_bot` (self-balancing robot) and `planar_arm` (2-DOF reaching arm) need the optional `physics` extra:

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -e ".[dev,physics]"
```

PyBullet 3.2.7 publishes a wheel only for Linux CPython 3.11. On other platforms or Python 3.12 pip builds it from
source, which needs a C++ compiler (on Windows, the Visual Studio Build Tools). Without the extra these robots simply
do not appear in the registry, and everything else works. Results and specs are in `docs/benchmarks/phase3/`.

Render one evaluation episode (episode 0 of a benchmark run with the same seed) to a GIF (needs the `render` extra):

```bash
pip install -e ".[physics,render]"
python -m rcl.benchmark.gif --env balance_bot --controller lqr --seed 0 --out balance_bot_lqr.gif
```

## Add a robot or controller (one file)

Drop a module into `rcl/environments/` (or `rcl/controllers/`) with a registry decorator:

```python
@register_environment("my_robot", params=MyRobotParams)   # Pydantic params model -> UI form
class MyRobot(BaseEnvironment):
    env_metadata = EnvMetadata(name="My Robot", ...)      # labels/units drive the plots
    ...
```

Nothing else changes: the API lists it, the dashboard builds its form, compatible controllers are
filtered from the declared metadata. `tests/api/test_modularity.py` proves this end to end.

## Commands

```bash
pytest -q
ruff check . && ruff format --check .
mypy rcl
lint-imports
```
