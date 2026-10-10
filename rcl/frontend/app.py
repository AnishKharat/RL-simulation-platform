"""Streamlit dashboard. Run with: streamlit run rcl/frontend/app.py"""

from __future__ import annotations

from typing import Any

import httpx
import streamlit as st

from rcl.frontend import client


def pick(label: str, entries: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Dropdown over registry entries, labelled with their metadata name."""
    names = {e["id"]: e["metadata"]["name"] for e in entries}
    chosen = st.selectbox(label, list(names), format_func=names.__getitem__, key=key)
    entry = next(e for e in entries if e["id"] == chosen)
    if entry["metadata"].get("description"):
        st.caption(entry["metadata"]["description"])
    return entry


def main() -> None:
    st.set_page_config(page_title="Robotics Control Lab", layout="wide")
    st.title("Robotics Control Lab")
    try:
        envs = client.environments()
    except httpx.HTTPError as exc:
        st.error(f"Cannot reach the API: {exc}")
        return
    with st.sidebar:
        env = pick("Environment", envs, "env")
        controllers = client.compatible_controllers(env["id"])
        if not controllers:
            st.warning("No controller is compatible with this environment.")
            return
        pick("Controller", controllers, f"ctl-{env['id']}")


main()
