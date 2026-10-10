"""Parameter forms rendered from the registry's JSON schemas (api.md, "Frontend notes").

Handles exactly the subset the registry allows: scalars with bounds, enums (incl. ``$ref`` to an
enum definition), optionals (``anyOf`` with null) and arrays of numbers/strings (``items`` or
``prefixItems``). No environment- or controller-specific code.
"""

from __future__ import annotations

import math
from typing import Any

import streamlit as st


def field_spec(name: str, prop: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """Normalise one schema property into ``{kind, title, description, default, ...}``.

    ``kind`` is one of bool, int, float, str, enum, array, optional (with ``inner``).
    """
    if "$ref" in prop:
        target = defs[prop["$ref"].rsplit("/", 1)[-1]]
        prop = {**target, **{k: v for k, v in prop.items() if k != "$ref"}}
    spec: dict[str, Any] = {
        "name": name,
        "title": prop.get("title") or name,
        "description": prop.get("description"),
        "default": prop.get("default"),
    }
    if "anyOf" in prop:
        branches = [b for b in prop["anyOf"] if b.get("type") != "null"]
        inner = field_spec(name, {**branches[0], "default": prop.get("default")}, defs)
        if len(branches) == len(prop["anyOf"]):
            return inner
        return spec | {"kind": "optional", "inner": inner}
    if "enum" in prop or "const" in prop:
        return spec | {"kind": "enum", "options": prop.get("enum") or [prop["const"]]}
    kind = {"boolean": "bool", "integer": "int", "number": "float", "string": "str"}.get(
        prop.get("type", "")
    )
    if kind:
        return spec | {"kind": kind, "min": prop.get("minimum"), "max": prop.get("maximum")}
    if prop.get("type") == "array":
        items = prop["items"] if "items" in prop else (prop.get("prefixItems") or [{}])[0]
        length = len(prop.get("prefixItems", [])) or (
            prop.get("minItems") if prop.get("minItems") == prop.get("maxItems") else None
        )
        return spec | {"kind": "array", "item_type": items.get("type", "number"), "length": length}
    raise ValueError(f"unsupported schema for {name!r}: {prop}")  # registry bug, not a UI case


def parse_array(text: str, item_type: str, length: int | None) -> list[Any]:
    """Parse comma-separated text into numbers/strings; ValueError with a user-facing message."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if length is not None and len(parts) != length:
        raise ValueError(f"expected {length} comma-separated values")
    conv = {"integer": int, "number": float}.get(item_type, str)
    try:
        return [conv(p) for p in parts]
    except ValueError:
        raise ValueError(f"values must be of type {item_type}") from None


def _step(default: float | None) -> float:
    """Number-input step about one order of magnitude below the default's scale."""
    if not default or not math.isfinite(default):
        return 0.1
    return float(10 ** (math.floor(math.log10(abs(default))) - 1))


def _widget(spec: dict[str, Any], key: str, problems: list[str]) -> Any:
    kind, label, help_, default = spec["kind"], spec["title"], spec["description"], spec["default"]
    if kind == "bool":
        return st.checkbox(label, value=bool(default), key=key, help=help_)
    if kind == "int":
        lo, hi = spec["min"], spec["max"]
        first = int(default) if default is not None else (int(lo) if lo is not None else 0)
        return st.number_input(
            label, value=first, min_value=lo, max_value=hi, step=1, key=key, help=help_
        )
    if kind == "float":
        lo, hi = spec["min"], spec["max"]
        start = float(default) if default is not None else (float(lo) if lo is not None else 0.0)
        return st.number_input(
            label,
            value=start,
            min_value=lo,
            max_value=hi,
            step=_step(start),
            format="%g",
            key=key,
            help=help_,
        )
    if kind == "str":
        return st.text_input(label, value=default or "", key=key, help=help_)
    if kind == "enum":
        opts = spec["options"]
        index = opts.index(default) if default in opts else 0
        return st.selectbox(label, opts, index=index, key=key, help=help_)
    if kind == "array":
        shown = ", ".join(str(v) for v in default) if default else ""
        text = st.text_input(f"{label} (comma-separated)", value=shown, key=key, help=help_)
        try:
            return parse_array(text, spec["item_type"], spec["length"])
        except ValueError as exc:
            problems.append(f"{label}: {exc}")
            return None
    # optional: a toggle that yields None when off
    if st.checkbox(f"Set {label}", value=default is not None, key=f"{key}.set"):
        return _widget(spec["inner"], key, problems)
    return None


def render_params(
    schema: dict[str, Any], key: str, errors: dict[str, str] | None = None
) -> tuple[dict[str, Any], list[str]]:
    """Render a form for ``schema`` (a params JSON schema); return ``(values, problems)``.

    ``errors`` maps a parameter name to a server-side message shown under its widget;
    ``problems`` lists client-side parse errors (the caller must block submission on them).
    """
    defs: dict[str, Any] = schema.get("$defs", {})
    values: dict[str, Any] = {}
    problems: list[str] = []
    for name, prop in schema.get("properties", {}).items():
        values[name] = _widget(field_spec(name, prop, defs), f"{key}.{name}", problems)
        if errors and name in errors:
            st.caption(f":red[{errors[name]}]")
    if not schema.get("properties"):
        st.caption("No parameters.")
    return values, problems
