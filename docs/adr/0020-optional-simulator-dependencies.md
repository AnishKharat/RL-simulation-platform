# 0020. Optional simulator dependencies (PyBullet extra, conditional registration, CI)

- Status: Accepted (architecture review 4, user approved 2026-10-10)
- Date: 2026-10-10

## Context
PyBullet is heavy and, as of 3.2.7, ships a prebuilt wheel **only for Linux CPython 3.11**. There is none for 3.12,
none for Windows and none for macOS, so everywhere else `pip` builds it from source (a C++ compiler, several minutes;
see `spikes/pybullet_determinism/README.md`). `discover()` imports every module under `rcl.environments` and turns any
import error into a loud `RegistryError` (registry.md 5). A top-level `import pybullet` in a robot file would
therefore break discovery, and with it the API, the CLI and the dashboard, on every machine without the simulator.

## Decision
1. **Optional extra** `physics = ["pybullet"]` in `pyproject.toml`. It is not in the base dependencies and not in
   `dev`, so a plain dev install stays fast and portable.
2. **Lazy import + conditional registration** (convention for every simulator-backed robot file): the module never
   imports the simulator at top level. It registers only if the simulator is importable:
   ```python
   _AVAILABLE = importlib.util.find_spec("pybullet") is not None
   _register = (
       register_environment("balance_bot", params=BalanceBotParams) if _AVAILABLE else (lambda c: c)
   )


   @_register
   class BalanceBot(BaseEnvironment): ...  # imports pybullet inside __init__ / the backend
   ```
   - On machines without the extra, the robot simply is not in the registry: not listed in the API or dashboard, and
     `unknown_id` (422) if requested.
   - A broken robot file (a real bug) still fails discovery loudly.
   - The same applies to `rcl.physics.pybullet` (ADR 0019).
3. **Tests** for simulator robots use `pytest.importorskip("pybullet")`. The suite stays green without the extra.
4. **CI:** a separate job `physics` on Linux with Python 3.11 installs `.[dev,physics]` (prebuilt wheel, no compiler)
   and runs the whole suite, so the PyBullet tests really run. The existing 3.11/3.12 matrix stays without the extra.
   If wheels for 3.12 appear, the job moves to 3.12.
5. **Import-linter:** the contracts for the `rcl.physics` layer are those of ADR 0019. No contract mentions
   `pybullet` (it is third-party).
6. **Versions:** `pybullet` is added to the experiment manager's recorded libraries (`RunRecord.versions`), so a run
   on a different simulator build is visible.

## Consequences
- The registry contents depend on the installed extras. A saved config that uses a simulator robot fails with a clear
  `unknown_id` where the extra is missing.
- Windows/macOS developers who want PyBullet need a C++ toolchain, or a Linux/WSL environment with Python 3.11.
  README will say so.
- Determinism is claimed per machine and build (ADR 0010 already limits it to the same platform and versions).

## Alternatives
- **PyBullet in base or dev dependencies:** forces a source build on Windows/macOS and on Python 3.12 everywhere.
- **Registering anyway and failing at run time:** the dashboard would offer robots that cannot run.
- **An `available` flag in `EnvMetadata`:** a frozen-type change for something conditional registration already
  expresses.
