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
