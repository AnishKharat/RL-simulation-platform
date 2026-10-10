# Spike: PyBullet install and determinism (THROWAWAY, architecture review 4)

Not part of the package. Feeds ADR 0019 (PhysicsBackend contract) and ADR 0020 (optional dependency, CI).

## Install (Windows 11, CPython 3.12.2, 2026-10-10)

- PyPI `pybullet` 3.2.7 has **no wheel for CPython 3.12 on any platform, and none for Windows or macOS**. The only
  prebuilt wheel among those checked is Linux (manylinux2014) x86_64 for CPython 3.11
  (`pip install --dry-run --only-binary=:all: --platform ... --python-version ...`).
- Here it was built from the sdist (80.5 MB) with the **already installed** Visual Studio 2019 Build Tools (MSVC); no
  system tools were installed. The build took ~5.5 min and succeeded: `pybullet 3.2.7` (API 202010061).
- Pitfall: pip's download of the sdist hung once on a slow link (0 bytes for 49 min). Downloading with
  `curl -C - --speed-limit ...` (resumable) and `pip install <file>.tar.gz` worked. SHA-256 verified against PyPI.
- Without MSVC on Windows (or gcc/clang elsewhere), installing on Python 3.12 **fails**. Options: Linux or WSL with
  Python 3.11 (wheel), or a C++ toolchain.

## Determinism (`python spikes/pybullet_determinism/determinism.py`)

Scenario: r2d2 from `pybullet_data` on a plane (wheel contacts), `DIRECT` (headless), fixed 1/240 s step, 600 steps,
random initial joint angles and random wheel torques from a seeded numpy RNG. State: base pose and twist plus joint
positions and velocities.

| Check | Result |
|---|---|
| Two fresh clients, same seed | identical (bit for bit) |
| One client reused with `resetSimulation` between runs | identical, and equal to fresh-client runs |
| `deterministicOverlappingPairs=1` (two runs) | identical; changes the trajectory versus the default |
| Separate process, same seed | identical (sha256 of the state array) |
| Different seed | differs |
| Exact continuation after `saveState` / `restoreState` | identical (max diff 0.0) |
| Continuation after a manual restore (`resetBasePositionAndOrientation`/`resetBaseVelocity`/`resetJointState`) | **diverges** (max diff 63) |
| `restoreState` followed by a manual reset to the *same* values | **diverges** (max diff 78): the reset calls themselves clear contact/solver warm-start caches |
| Manual restore repeated with the same procedure | repeatable (same result each time) |

## Conclusions for the architecture

1. **ADR 0010's claim holds for PyBullet**: same seed, same procedure, same machine/build, headless gives
   bit-identical trajectories, across clients, `resetSimulation` reuse and processes. Set the engine parameters
   explicitly and identically every reset (they change trajectories).
2. **`set_state` is not an exact continuation.** Writing generalised coordinates resets internal caches. That is
   deterministic as a procedure (seeded initial states: fine), but it is not the same as the run the state was read
   from. The `PhysicsBackend` contract is amended: `set_state` = "set coordinates, caches cleared", plus native
   `save()` / `restore(handle)` for exact snapshots (ADR 0019 amendment).
3. **Finite-difference linearisation** (convention C2) must evaluate every perturbation with the same procedure
   (`restore(base)` then `set_state(x +/- eps)` then step). For contact-rich robots the Jacobian may be noisy. The
   balance robot's LQR must be validated at implementation time.
4. Limits: one machine, one build, CPU only, `DIRECT` mode. `GUI` mode, other platforms and other PyBullet builds are
   outside the claim (as in ADR 0010).
