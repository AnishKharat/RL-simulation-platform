# RCL

Robotics Control Lab: a modular platform to run, compare and benchmark classical and RL
controllers on simulated robots, with reproducible experiments. See `project_context.md`.

## Install

```bash
pip install -e ".[dev]"      # extras: rl, api, frontend
```

## Commands

```bash
pytest -q
ruff check . && ruff format --check .
mypy rcl
lint-imports
streamlit run rcl/frontend/app.py       # Phase 1+
uvicorn rcl.api.main:app --reload       # Phase 1+
```
