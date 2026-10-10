"""Streamlit dashboard. Run with: streamlit run rcl/frontend/app.py"""

from __future__ import annotations

import time
from typing import Any

import httpx
import streamlit as st

from rcl.frontend import client, forms

NO_ERRORS: dict[str, dict[str, str]] = {"environment": {}, "controller": {}}


def pick(label: str, entries: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Dropdown over registry entries, labelled with their metadata name."""
    names = {e["id"]: e["metadata"]["name"] for e in entries}
    chosen = st.selectbox(label, list(names), format_func=names.__getitem__, key=key)
    entry = next(e for e in entries if e["id"] == chosen)
    if entry["metadata"].get("description"):
        st.caption(entry["metadata"]["description"])
    return entry


def field_errors(detail: Any) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Split a 422 ``detail`` into per-parameter messages ``{component: {param: msg}}`` and
    general messages (everything that is not under ``body.<component>.params``)."""
    per_param: dict[str, dict[str, str]] = {"environment": {}, "controller": {}}
    general: list[str] = []
    items: list[dict[str, Any]] = (
        detail if isinstance(detail, list) else [{"loc": [], "msg": str(detail)}]
    )
    for item in items:
        loc: list[str] = [str(p) for p in item.get("loc", [])]
        if len(loc) >= 4 and loc[1] in per_param and loc[2] == "params":
            per_param[loc[1]][loc[3]] = item["msg"]
        else:
            general.append(f"{'.'.join(loc[1:])}: {item['msg']}" if len(loc) > 1 else item["msg"])
    return per_param, general


def sidebar() -> dict[str, Any] | None:
    """Render the configuration form; return the experiment config when Run is pressed."""
    errors = st.session_state.get("errors", NO_ERRORS)
    env = pick("Environment", client.environments(), "env")
    controllers = client.compatible_controllers(env["id"])
    if not controllers:
        st.warning("No controller is compatible with this environment.")
        return None
    ctl = pick("Controller", controllers, f"ctl-{env['id']}")
    name = st.text_input("Experiment name", value=f"{env['id']}-{ctl['id']}", key="name")
    seed = int(st.number_input("Seed", min_value=0, value=0, step=1, key="seed"))
    episodes = int(st.number_input("Evaluation episodes", 1, 100, 5, 1, key="episodes"))
    config: dict[str, Any] = {"name": name, "seed": seed, "evaluation": {"episodes": episodes}}
    if ctl["metadata"]["requires_training"]:
        steps = st.number_input("Training timesteps", 64, 5_000_000, 20_000, 1000, key="steps")
        config["training"] = {"total_timesteps": int(steps)}
    problems: list[str] = []
    for label, entry, kind in (
        ("Environment", env, "environment"),
        ("Controller", ctl, "controller"),
    ):
        st.subheader(f"{label} parameters")
        params, bad = forms.render_params(
            entry["params_schema"], f"{kind}-{entry['id']}", errors[kind]
        )
        problems += bad
        config[kind] = {"id": entry["id"], "params": params}
    for p in problems:
        st.error(p)
    if st.button("Run", type="primary", disabled=bool(problems)):
        return config
    return None


def submit(config: dict[str, Any]) -> None:
    status, body = client.create_experiment(config)
    if status == 202:
        st.session_state.update(run_id=body["run_ids"][0], errors=NO_ERRORS)
        return
    per_param, general = field_errors(body.get("detail"))
    st.session_state.update(errors=per_param, general_errors=general, run_id=None)


def watch(run_id: str) -> dict[str, Any]:
    """Poll the run until it is done or failed, showing progress."""
    bar, text = st.progress(0.0), st.empty()
    while True:
        run = client.run(run_id)
        bar.progress(min(float(run["progress"]), 1.0))
        text.caption(f"Run {run['status']}" + (f" - {run['stage']}" if run["stage"] else ""))
        if run["status"] in ("done", "failed"):
            return run
        time.sleep(0.5)


def main() -> None:
    st.set_page_config(page_title="Robotics Control Lab", layout="wide")
    st.title("Robotics Control Lab")
    try:
        with st.sidebar:
            config = sidebar()
    except httpx.HTTPError as exc:
        st.error(f"Cannot reach the API: {exc}")
        return
    if config is not None:
        submit(config)
        st.rerun()  # redraw the sidebar so server-side field errors show next to their widgets
    for msg in st.session_state.pop("general_errors", []):
        st.error(msg)
    run_id = st.session_state.get("run_id")
    if run_id:
        run = watch(run_id)
        if run["status"] == "failed":
            st.error(f"Run failed in stage {run['error']['stage']}: {run['error']['message']}")
        else:
            st.success(f"Run {run_id} done")


if __name__ == "__main__":  # streamlit and AppTest execute the script as __main__
    main()
