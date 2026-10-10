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
