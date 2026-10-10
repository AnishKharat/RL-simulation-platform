"""Plotly figures and result tables built only from API payloads (labels come from the API)."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _names(labels: list[str], units: list[str], n: int, fallback: str) -> list[str]:
    out = []
    for i in range(n):
        label = labels[i] if i < len(labels) else f"{fallback} {i}"
        unit = units[i] if i < len(units) else ""
        out.append(f"{label} [{unit}]" if unit else label)
    return out


def trajectory_figure(t: dict[str, Any]) -> go.Figure:
    """One stacked, x-shared panel per state component, per action component and the reward."""
    n_state, n_action = len(t["states"][0]), len(t["actions"][0]) if t["actions"] else 0
    titles = [
        *_names(t["state_labels"], t["state_units"], n_state, "state"),
        *_names(t["action_labels"], t["action_units"], n_action, "action"),
        "reward",
    ]
    fig = make_subplots(rows=len(titles), cols=1, shared_xaxes=True, subplot_titles=titles)
    time = t["time"]
    for i in range(n_state):
        fig.add_scatter(x=time, y=[s[i] for s in t["states"]], row=i + 1, col=1)
    for j in range(n_action):  # actions/rewards have T samples: step k acts at time[k]
        fig.add_scatter(
            x=time[:-1], y=[a[j] for a in t["actions"]], line_shape="hv", row=n_state + j + 1, col=1
        )
    fig.add_scatter(x=time[:-1], y=t["rewards"], row=len(titles), col=1)
    fig.update_xaxes(title_text="time [s]", row=len(titles), col=1)
    fig.update_layout(showlegend=False, height=170 * len(titles) + 60, margin={"t": 40, "b": 30})
    return fig


def curve_figure(curve: list[list[float]]) -> go.Figure:
    """Learning curve: mean episode return versus training timesteps."""
    fig = go.Figure(
        go.Scatter(x=[p[0] for p in curve], y=[p[1] for p in curve], mode="lines+markers")
    )
    fig.update_layout(xaxis_title="timesteps", yaxis_title="mean episode return", height=300)
    return fig


def run_metrics_table(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Run-level metric rows (``episode_index is None``): the per-run means."""
    return [
        {"metric": m["metric_name"], "value": m["value"], "unit": m["unit"], "status": m["status"]}
        for m in run["metrics"]
        if m["episode_index"] is None
    ]


def episode_table(run: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per episode: outcome flags plus that episode's metric values."""
    per_episode: dict[int, dict[str, Any]] = {e["index"]: dict(e) for e in run["episodes"]}
    for m in run["metrics"]:
        if m["episode_index"] is not None:
            per_episode[m["episode_index"]][m["metric_name"]] = m["value"]
    return [per_episode[i] for i in sorted(per_episode)]


def comparison_figure(
    rows: list[dict[str, Any]], metric: str, unit: str | None = None
) -> go.Figure:
    """Bars of the cross-seed mean per controller with 95% interval whiskers.

    ``rows`` are API ``SummaryOut`` dicts of one environment; only ``metric`` is drawn.
    Controllers whose interval is undefined (fewer than two seeds) are drawn without whiskers.
    """
    sel = sorted((r for r in rows if r["metric_name"] == metric), key=lambda r: r["controller_id"])
    means = [r["mean"] for r in sel]
    up = [r["ci_high"] - r["mean"] if r["ci_high"] is not None else 0.0 for r in sel]
    down = [r["mean"] - r["ci_low"] if r["ci_low"] is not None else 0.0 for r in sel]
    fig = go.Figure(
        go.Bar(
            x=[r["controller_id"] for r in sel],
            y=means,
            error_y={"type": "data", "symmetric": False, "array": up, "arrayminus": down},
        )
    )
    unit = unit or (sel[0]["unit"] if sel else "")
    fig.update_layout(
        yaxis_title=f"{metric} [{unit}]" if unit else metric, xaxis_title="controller", height=380
    )
    return fig


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.4g}"


def summary_table(rows: list[dict[str, Any]], metric: str | None = None) -> list[dict[str, Any]]:
    """One display row per (controller, metric): mean, interval, seed counts and warnings."""
    out = []
    for r in rows:
        if metric is not None and r["metric_name"] != metric:
            continue
        notes = []
        if r["low_n"]:
            notes.append("fewer than 5 seeds")
        if r["n_censored"]:
            notes.append(f"{r['n_censored']} censored")
        out.append(
            {
                "controller": r["controller_id"],
                "metric": r["metric_name"],
                "mean": _fmt(r["mean"]),
                "95% interval": f"[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}] ({r['ci_method']})",
                "seeds (valid/total)": f"{r['n_valid']}/{r['n_total']}",
                "unit": r["unit"],
                "notes": ", ".join(notes),
            }
        )
    return out
