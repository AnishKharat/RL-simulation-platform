# 0018. Benchmark CLI and report export

- Status: Accepted
- Date: 2026-10-10

## Context
Phase 2 exit criterion: one command produces the PID vs LQR vs PPO vs SAC comparison table and plots with confidence
intervals, reproducibly. The dashboard and API already cover the interactive path; a scriptable path is needed for
CI, papers and re-runs.

## Decision
- New top-level package `rcl.benchmark` (`python -m rcl.benchmark`), in the **api tier** of the layer contract: it may
  import `experiments`, `analytics`, `registry` and `persistence` (not environments, controllers, metrics directly or
  the frontend), exactly like `rcl.api`. `pyproject.toml` gains `rcl.benchmark` next to `rcl.api` in the layers
  contract and a matching forbidden-imports contract; the frontend contract also forbids it. No frozen interface
  changes (import-linter config is tooling).
- The CLI runs the benchmark in-process through the same `run_stored_benchmark` the API job manager uses, so CLI and
  API results are identical for the same spec. It stores into a SQLite file under `--out` by default.
- Report export lives in `rcl.analytics.report`: Markdown, self-contained HTML, SVG bar charts with 95% interval
  whiskers (no plotting dependency) and `summary.csv`. Output is deterministic (no timestamps), so the same seeds
  give byte-identical files and a re-run can be diffed.
- Grid defaults come from the registry (all registered ids), never from names in code. Incompatible pairs are skipped
  and reported.

## Consequences
- Reproducibility is checkable with `diff` on `report.md` / `summary.csv` (CPU, same library versions, ADR 0010).
- SVG charts are plain; interactive Plotly charts stay in the dashboard.

## Alternatives
- A CLI inside `rcl.experiments` or `rcl.analytics`: blocked by the layer contract (experiments may not import
  analytics and vice versa).
- Matplotlib/kaleido for static plots: heavy dependencies for simple bar charts.
