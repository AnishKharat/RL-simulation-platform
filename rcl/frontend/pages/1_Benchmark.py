"""Benchmark comparison page: run controllers x environments x seeds, compare with intervals."""

from __future__ import annotations

import time
from typing import Any

import httpx
import streamlit as st

from rcl.frontend import client, plots

NEW = "(new benchmark)"


def names(entries: list[dict[str, Any]]) -> dict[str, str]:
    return {e["id"]: e["metadata"]["name"] for e in entries}


def start(envs: list[dict[str, Any]], ctls: list[dict[str, Any]]) -> None:
    """Sidebar form that creates a benchmark from registry data only."""
    env_names, ctl_names = names(envs), names(ctls)
    with st.form("bench-form"):
        name = st.text_input("Benchmark name", value="comparison", key="bench-name")
        env_ids = st.multiselect(
            "Environments", list(env_names), format_func=env_names.__getitem__, key="bench-envs"
        )
        ctl_ids = st.multiselect(
            "Controllers", list(ctl_names), format_func=ctl_names.__getitem__, key="bench-ctls"
        )
        n_seeds = int(st.number_input("Seeds", 1, 50, 5, 1, key="bench-seeds"))
        episodes = int(st.number_input("Evaluation episodes", 1, 100, 5, 1, key="bench-episodes"))
        steps = int(
            st.number_input(
                "Training timesteps (RL)", 64, 5_000_000, 20_000, 1000, key="bench-steps"
            )
        )
        go = st.form_submit_button("Run benchmark", type="primary")
    if not go:
        return
    spec: dict[str, Any] = {
        "name": name,
        "environments": [{"id": i} for i in env_ids],
        "controllers": [{"id": i} for i in ctl_ids],
        "seeds": list(range(n_seeds)),
        "episodes": episodes,
    }
    if any(c["metadata"]["requires_training"] for c in ctls if c["id"] in ctl_ids):
        spec["training"] = {"total_timesteps": steps}
    status, body = client.create_benchmark(spec)
    if status == 202:
        st.session_state.update(suite_id=body["suite_id"], skipped=body["skipped"], errors=[])
        st.session_state["bench-open"] = body["suite_id"]
    else:
        detail = body.get("detail")
        items = detail if isinstance(detail, list) else [{"loc": [], "msg": str(detail)}]
        st.session_state["errors"] = [
            f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}".lstrip(": ") for e in items
        ]


def watch(suite_id: str) -> dict[str, Any]:
    bar, text = st.progress(0.0), st.empty()
    while True:
        suite = client.benchmark(suite_id)
        run = suite["benchmark_runs"][0]
        bar.progress(min(float(run["progress"]), 1.0))
        text.caption(f"Benchmark {run['status']}")
        if run["status"] in ("done", "failed"):
            return suite
        time.sleep(0.5)


def show(suite: dict[str, Any]) -> None:
    runs = [r for e in suite["experiments"] for r in e["runs"]]
    failed = sum(r["status"] == "failed" for r in runs)
    if failed:
        st.warning(f"{failed} of {len(runs)} runs failed; they are excluded from the statistics.")
    for k in st.session_state.get("skipped", []):
        st.caption(f"Skipped {k['controller_id']} on {k['env_id']}: {k['reason']}")
    env_ids = sorted({e["env_id"] for e in suite["experiments"]})
    if not env_ids:
        return
    env = st.selectbox("Environment", env_ids, key="bench-env")
    rows = client.benchmark_summary(suite["suite_id"], env)
    present = {r["metric_name"] for r in rows}
    catalogue = [m for m in client.metrics() if m["name"] in present]
    metric = st.selectbox("Metric", [m["name"] for m in catalogue], key="bench-metric")
    info = next((m for m in catalogue if m["name"] == metric), None)
    if info is not None:
        st.caption(info["description"] + ("" if info["higher_is_better"] else " Lower is better."))
    st.plotly_chart(plots.comparison_figure(rows, metric))
    st.dataframe(plots.summary_table(rows, metric), hide_index=True)
    if any(r["low_n"] for r in rows):
        st.info("Fewer than 5 seeds: intervals are wide and no significance claim should be made.")
    with st.expander("All metrics"):
        st.dataframe(plots.summary_table(rows), hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Benchmark comparison", layout="wide")
    st.title("Benchmark comparison")
    try:
        with st.sidebar:
            start(client.environments(), client.controllers())
            suites = {s["suite_id"]: s["name"] for s in client.benchmarks()}
            chosen = st.selectbox(
                "Open benchmark",
                [NEW, *suites],
                format_func=lambda i: suites.get(i, i),
                key="bench-open",
            )
        for msg in st.session_state.get("errors", []):
            st.error(msg)
        if chosen != NEW:
            suite = watch(chosen)
            if suite["benchmark_runs"][0]["status"] == "failed" and not suite["experiments"]:
                st.error("Benchmark failed before any run started.")
            else:
                show(suite)
    except httpx.HTTPError as exc:
        st.error(f"API error: {exc}")


if __name__ == "__main__":
    main()
