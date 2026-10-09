"""THROWAWAY: show that dropping one file into a package auto-registers it."""

import sys
import tempfile
from pathlib import Path

from rcl.registry import Registry, discover, use_registry

MODULE = """
from rcl.registry import register_{kind}

@register_{kind}("{id}")
class C:
    metadata = {{"action_space_kind": "continuous", "supported_action_spaces": ["continuous"]}}
"""

with tempfile.TemporaryDirectory() as tmp, use_registry(Registry()) as reg:
    pkg = Path(tmp) / "demo_robots"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    sys.path.insert(0, tmp)
    (pkg / "cartpole.py").write_text(MODULE.format(kind="environment", id="cartpole"))
    discover(["demo_robots"])
    print("before adding file:", [e.id for e in reg.entries("environment")])
    (pkg / "pendulum.py").write_text(MODULE.format(kind="environment", id="pendulum"))
    discover(["demo_robots"])
    print("after adding pendulum.py:", [e.id for e in reg.entries("environment")])
