"""Robustness page: sweep one perturbation over levels and plot degradation per controller."""

from __future__ import annotations

import time
from typing import Any

import httpx
import streamlit as st

from rcl.frontend import client, plots

NEW = "(new sweep)"


def names(entries: list[dict[str, Any]]) -> dict[str, str]:
    return {e["id"]: e["metadata"]["name"] for e in entries}


def parse_levels(text: str) -> list[float]:
    """Comma-separated numbers; raises ValueError naming the bad token."""
    return [float(t) for t in text.replace(";", ",").split(",") if t.strip()]


def start(envs: list[dict[str, Any]], perts: list[dict[str, Any]]) -> None:
    """Sidebar form built from registry metadata only (nominal level, unit, action spaces)."""
    env_names, pert_names = names(envs), names(perts)
    env_id = st.selectbox(
        "Environment", list(env_names), format_func=env_names.__getitem__, key="rob-env-pick"
    )
    env = next((e for e in envs if e["id"] == env_id), None)
    kind = env["metadata"]["action_space_kind"] if env else None
    ok = [p for p in perts if kind is None or kind in p["metadata"]["action_spaces"]]
    ctls = client.compatible_controllers(env_id) if env_id else []
    ctl_names = names(ctls)
    pert_id = st.selectbox(
        "Perturbation", [p["id"] for p in ok], format_func=pert_names.__getitem__, key="rob-pert"
    )
    meta = next((p["metadata"] for p in ok if p["id"] == pert_id), None)
    with st.form("rob-form"):
        name = st.text_input("Sweep name", value="robustness", key="rob-name")
        ctl_ids = st.multiselect(
            "Controllers", list(ctl_names), format_func=ctl_names.__getitem__, key="rob-ctls"
        )
        if meta:
            unit = f" [{meta['magnitude_unit']}]" if meta["magnitude_unit"] else ""
            st.caption(
                f"Levels vary {meta['magnitude_param']}{unit}; nominal "
                f"{meta['nominal_magnitude']:g} is always included."
            )
        text = st.text_input("Levels (comma-separated)", key="rob-levels")
        n_seeds = int(st.number_input("Seeds", 1, 50, 5, 1, key="rob-seeds"))
        episodes = int(st.number_input("Evaluation episodes", 1, 100, 5, 1, key="rob-episodes"))
        steps = int(st.number_input("Training timesteps (RL)", 64, 5_000_000, 20_000, 1000))
        go = st.form_submit_button("Run sweep", type="primary")
    if not go:
        return
    try:
        levels = parse_levels(text)
    except ValueError as exc:
        st.session_state["errors"] = [f"levels: {exc}"]
        return
    spec: dict[str, Any] = {
        "name": name,
        "environments": [{"id": env_id}],
        "controllers": [{"id": i} for i in ctl_ids],
        "seeds": list(range(n_seeds)),
        "episodes": episodes,
        "perturbation": {"id": pert_id, "levels": levels},
    }
    if any(c["metadata"]["requires_training"] for c in ctls if c["id"] in ctl_ids):
        spec["training"] = {"total_timesteps": steps}
    status, body = client.create_benchmark(spec)
    if status == 202:
        st.session_state.update(errors=[], skipped=body["skipped"])
        st.session_state["rob-open"] = body["suite_id"]
        return
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
        text.caption(f"Sweep {run['status']}")
        if run["status"] in ("done", "failed"):
            return suite
        time.sleep(0.5)


def show(suite: dict[str, Any], perts: list[dict[str, Any]]) -> None:
    for k in st.session_state.get("skipped", []):
        st.caption(f"Skipped {k['controller_id']} on {k['env_id']}: {k['reason']}")
    sweep = suite["spec"].get("perturbation")
    if sweep is None:
        st.info("This benchmark has no perturbation sweep; open it on the Benchmark page.")
        return
    env_ids = sorted({e["env_id"] for e in suite["experiments"]})
    if not env_ids:
        return
    env = st.selectbox("Environment", env_ids, key="rob-env")
    shown = [m["name"] for m in client.metrics()]
    metric = st.selectbox(
        "Metric",
        shown,
        index=shown.index("success_rate") if "success_rate" in shown else 0,
        key="rob-metric",
    )
    status, curves = client.robustness(suite["suite_id"], env, metric)
    if status != 200:
        st.error(str(curves.get("detail", curves)))
        return
    meta: dict[str, Any] = next((p["metadata"] for p in perts if p["id"] == sweep["id"]), {})
    unit = f" [{meta['magnitude_unit']}]" if meta.get("magnitude_unit") else ""
    x_title = f"{meta.get('magnitude_param', 'level')}{unit}"
    st.plotly_chart(plots.degradation_figure(curves, x_title))
    st.dataframe(plots.degradation_table(curves), hide_index=True)
    if any(p["n_valid"] < 5 for c in curves for p in c["points"]):
        st.info("Fewer than 5 seeds: intervals are wide and no significance claim should be made.")
    st.caption(
        "Training is nominal; the same trained policy is evaluated at every level. The breakdown "
        "level is the first level above nominal where the mean falls below half its nominal value "
        "(meaningful for success_rate)."
    )


def main() -> None:
    st.set_page_config(page_title="Robustness", layout="wide")
    st.title("Robustness")
    try:
        perts = client.perturbations()
        with st.sidebar:
            start(client.environments(), perts)
            suites = {s["suite_id"]: s["name"] for s in client.benchmarks()}
            chosen = st.selectbox(
                "Open sweep",
                [NEW, *suites],
                format_func=lambda i: suites.get(i, i),
                key="rob-open",
            )
        for msg in st.session_state.get("errors", []):
            st.error(msg)
        if chosen != NEW:
            suite = watch(chosen)
            if suite["benchmark_runs"][0]["status"] == "failed" and not suite["experiments"]:
                st.error("Sweep failed before any run started.")
            else:
                show(suite, perts)
    except httpx.HTTPError as exc:
        st.error(f"API error: {exc}")


if __name__ == "__main__":
    main()
