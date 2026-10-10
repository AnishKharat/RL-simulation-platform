"""Modularity proof (controllers): a new controller is ONE file with ONE decorator, discovered
outside rcl/controllers, and the benchmark runner compares it fairly with existing controllers
without any existing file being edited."""

import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from rcl.controllers.classical.pid import PID, PIDParams
from rcl.environments.cartpole import CartPole, CartPoleParams
from rcl.experiments import BenchmarkSpec, run_benchmark
from rcl.registry import Registry, discover, get_registry, use_registry

DUMMY_CONTROLLER = '''
    from pathlib import Path
    from typing import Any, ClassVar

    import numpy as np

    from rcl.core.controller import BaseController
    from rcl.core.types import ActionSpaceKind, ControllerFamily, ControllerMetadata, EnvSpec
    from rcl.registry import register_controller


    @register_controller("bang_bang")
    class BangBang(BaseController):
        """Pushes with full force toward the side the pole leans to."""

        metadata: ClassVar[ControllerMetadata] = ControllerMetadata(
            name="Bang-bang", version="0.1.0", family=ControllerFamily.CLASSICAL,
            supported_action_spaces=(ActionSpaceKind.CONTINUOUS,), requires_training=False,
        )

        def configure(self, env_spec: EnvSpec, params: dict[str, Any], linear_model: Any = None):
            self._hi = np.asarray(env_spec.action_high)
            self._shape = env_spec.action_shape

        def predict(self, obs: np.ndarray, deterministic: bool = True):
            lean = obs[2] + 0.5 * obs[3]
            return np.full(self._shape, self._hi[0] if lean > 0 else -self._hi[0])

        def save(self, path: Path) -> None: ...
        def load(self, path: Path) -> None: ...
'''


def test_one_file_one_decorator_joins_the_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pkg = tmp_path / "fx_controllers"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "bang_bang.py").write_text(textwrap.dedent(DUMMY_CONTROLLER))  # the ONLY new file
    monkeypatch.syspath_prepend(str(tmp_path))

    reg = Registry()  # isolated; existing classes are added by hand, nothing is edited
    reg.register("environment", "cartpole", CartPole, CartPoleParams)
    reg.register("controller", "pid", PID, PIDParams)
    try:
        with use_registry(reg):
            discover(["fx_controllers"])
        assert [c.id for c in reg.compatible_controllers("cartpole")] == ["bang_bang", "pid"]

        spec = BenchmarkSpec.model_validate(
            {
                "name": "modularity",
                "environments": [{"id": "cartpole"}],
                "controllers": [{"id": "pid"}, {"id": "bang_bang"}],
                "seeds": [0, 1],
                "episodes": 2,
            }
        )
        result = run_benchmark(spec, registry=reg)
    finally:
        sys.modules.pop("fx_controllers.bang_bang", None)
        sys.modules.pop("fx_controllers", None)

    cells = {c.controller_id: c for c in result.cells}
    assert set(cells) == {"pid", "bang_bang"} and result.skipped == []
    for cell in cells.values():
        assert [r.record.seed for r in cell.runs] == [0, 1]
    # fairness holds for the newcomer: identical initial states as the existing controller
    for new, old in zip(cells["bang_bang"].runs, cells["pid"].runs, strict=True):
        for tn, to in zip(new.trajectories, old.trajectories, strict=True):
            assert np.array_equal(tn.states[0], to.states[0])
    assert {m.name for m in cells["bang_bang"].runs[0].metrics} == {
        m.name for m in cells["pid"].runs[0].metrics
    }
    # nothing leaked into the global registry or into rcl/controllers
    assert "bang_bang" not in [e.id for e in get_registry().entries("controller")]
    assert not list((Path(__file__).parents[2] / "rcl" / "controllers").rglob("bang_bang*"))
