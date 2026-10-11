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


def _signals(t: dict[str, Any]) -> list[tuple[str, list[float], bool]]:
    """Every plottable signal as (title, values, is_step_input); actions and rewards have T
    samples (step k acts at time[k]), states T+1."""
    n_state, n_action = len(t["states"][0]), len(t["actions"][0]) if t["actions"] else 0
    states = _names(t["state_labels"], t["state_units"], n_state, "state")
    actions = _names(t["action_labels"], t["action_units"], n_action, "action")
    return [
        *[(name, [s[i] for s in t["states"]], False) for i, name in enumerate(states)],
        *[(name, [a[j] for a in t["actions"]], True) for j, name in enumerate(actions)],
        ("reward", list(t["rewards"]), False),
    ]


def signal_names(t: dict[str, Any]) -> list[str]:
    """Titles of all plottable signals, in panel order (states, actions, reward)."""
    return [name for name, _, _ in _signals(t)]


def default_signals(t: dict[str, Any], limit: int = 9) -> list[str]:
    """All signals if there are at most ``limit``; otherwise the task outputs are not known
    here, so: the first four states, every action and the reward (many-joint robots)."""
    names = signal_names(t)
    if len(names) <= limit:
        return names
    n_state = len(t["states"][0])
    return names[: min(4, n_state)] + names[n_state:]


def trajectory_figure(t: dict[str, Any], signals: list[str] | None = None) -> go.Figure:
    """One stacked, x-shared panel per selected signal (default: all)."""
    chosen = [s for s in _signals(t) if signals is None or s[0] in signals]
    rows = max(len(chosen), 1)
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, subplot_titles=[c[0] for c in chosen] or [""]
    )
    time = t["time"]
    for row, (_, values, step_input) in enumerate(chosen, start=1):
        x = time[: len(values)]
        fig.add_scatter(x=x, y=values, line_shape="hv" if step_input else "linear", row=row, col=1)
    fig.update_xaxes(title_text="time [s]", row=rows, col=1)
    fig.update_layout(showlegend=False, height=170 * rows + 60, margin={"t": 40, "b": 30})
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


def degradation_figure(curves: list[dict[str, Any]], x_title: str = "level") -> go.Figure:
    """One line per controller: cross-seed mean vs sweep level with 95% interval whiskers.

    ``curves`` are API ``CurveOut`` dicts (one metric, one environment)."""
    fig = go.Figure()
    for c in sorted(curves, key=lambda c: c["controller_id"]):
        pts = c["points"]
        up = [p["ci_high"] - p["mean"] if p["ci_high"] is not None else 0.0 for p in pts]
        down = [p["mean"] - p["ci_low"] if p["ci_low"] is not None else 0.0 for p in pts]
        fig.add_scatter(
            x=[p["level"] for p in pts],
            y=[p["mean"] for p in pts],
            name=c["controller_id"],
            mode="lines+markers",
            error_y={"type": "data", "symmetric": False, "array": up, "arrayminus": down},
        )
    fig.update_layout(
        xaxis_title=x_title, yaxis_title=curves[0]["metric_name"] if curves else "", height=400
    )
    return fig


def degradation_table(curves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One display row per (controller, level); the breakdown level repeats on each row."""
    return [
        {
            "controller": c["controller_id"],
            "level": p["level"],
            "mean": _fmt(p["mean"]),
            "95% interval": f"[{_fmt(p['ci_low'])}, {_fmt(p['ci_high'])}] ({p['ci_method']})",
            "relative to nominal": _fmt(p["relative"]),
            "seeds (valid/total)": f"{p['n_valid']}/{p['n_total']}",
            "breakdown level": _fmt(c["breakdown"]),
        }
        for c in sorted(curves, key=lambda c: c["controller_id"])
        for p in c["points"]
    ]
