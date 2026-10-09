import ast
from pathlib import Path

import rcl.core


def test_core_imports_no_other_rcl_subpackage() -> None:
    core_dir = Path(rcl.core.__file__).parent
    bad = []
    for py in core_dir.glob("*.py"):
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            bad += [
                (py.name, m) for m in mods if m.startswith("rcl") and not m.startswith("rcl.core")
            ]
    assert not bad
