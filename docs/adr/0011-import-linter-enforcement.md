# 0011. import-linter enforcement

- Status: Accepted
- Date: 2026-10-10

## Context
Modularity ("one file + decorator" extensibility, frontend via API only) erodes quickly if only documented.
We want violations to fail in CI/pre-commit (`lint-imports`).

## Decision
Dependency rules are encoded in `pyproject.toml` under `[tool.importlinter]` (root package `rcl`):
- `core` imports no other `rcl` package.
- `environments` and `controllers` may not import experiments, metrics, analytics, persistence, api, frontend
  (so they import only `core` and `registry`, ADR 0003) and are `independence`-isolated from each other.
- `registry` imports only `core`.
- `experiments` may not import environments, controllers, analytics, api, frontend (it reaches them through the
  registry); `api` may not import environments, controllers, metrics, frontend.
- `frontend` imports no other `rcl` package (HTTP only).
- A `layers` contract: frontend > api > experiments > (analytics | persistence | metrics) >
  (environments | controllers) > registry > core, downward only.
- `lint-imports` runs with tests and ruff before every commit.

## Consequences
- Architecture violations are caught mechanically, not in review.
- Contracts reference packages by name, so new top-level packages must be added to the config.
- The `metrics`, `analytics`, `persistence` packages share a layer, so they may not import each other
  (a `layers` rule with `|`); if one needs another, split or move shared code to `core`.
- `project_context.md` section 4 wording still says env/controllers import only `core`; it must be updated.

## Alternatives
- Convention plus code review: not reliable across agents.
- `tach` / custom AST scripts: more setup, less standard.
- Runtime checks: too late and incomplete.
