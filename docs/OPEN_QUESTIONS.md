# Open Questions

> Agents: add ambiguities or architecture conflicts here and ask the user. Don't guess.
> When answered, move the entry to "Resolved" with the decision (and an ADR link if one exists).

## Open

- **Note (user decision, Phase 3 follow-up):** Arm RL controllers do not reach target; optional time-boxed reward-shaping attempt later. | **Raised by:** user, 2026-10-10 | **Affects:** planar_arm, Phase 3 benchmarks (not blocking)

- **Q:** Approve ADR 0021 (review 5): new registry Kind `perturbation` + core `PerturbationMetadata`/`ParamPerturbation`, rollout recording `info["perturbed"]`, optional `perturbation` sweep in `BenchmarkSpec`/API? All additive, no existing signature changes. Also: should `disturbance` act only through the actuated input (proposed), or do you want an environment-level external-force hook (`apply_disturbance`, changes BaseEnvironment)? | **Raised by:** orchestrator, review 5, 2026-10-11 | **Affects:** core, registry, rollout, manager, API, Phase 4 (docs/architecture/phase4-robustness.md)

## Resolved

- **Q:** Approve a new physics layer `rcl.physics` (PhysicsBackend implementations) between the plugin tier and core, which changes the frozen rule "environments import only `core` and `registry`" to "`core`, `registry` and `physics`" (CLAUDE.md 1.3, rcl/environments/CLAUDE.md, project_context 4, import-linter contracts)? Fallback without a rule change: backends as private modules inside `rcl/environments/`. | **Raised by:** orchestrator, review 4, 2026-10-10 | **Affects:** ADR 0019, pyproject contracts, Phase 3
  **Resolved 2026-10-10 (user approved):** `rcl.physics` layer, imports core only, only environments may import it; rule files, project_context 4 and import-linter contracts updated. ADR 0019 Accepted.

- **Q:** Robot arm model: KUKA iiwa 7-DOF from `pybullet_data` (realistic, large RL budgets) or a 2-3 DOF planar arm (fast to learn)? | **Raised by:** orchestrator, review 4, 2026-10-10 | **Affects:** docs/architecture/phase3-robots.md, Phase 3 benchmarks
  **Resolved 2026-10-10 (user):** a 2-3 DOF planar arm, not the KUKA iiwa. Chosen: 2-DOF arm in the horizontal plane (no gravity load, fast RL), see docs/architecture/phase3-robots.md.

- **Q:** May a robot ship a model file (URDF) next to its single `.py` file (e.g. `rcl/environments/assets/`), or must models be built in code / come from `pybullet_data`? | **Raised by:** orchestrator, review 4, 2026-10-10 | **Affects:** "one robot = one file" rule, Phase 3
  **Resolved 2026-10-10 (user):** yes, in a sibling folder `rcl/environments/assets/<robot_id>/`; the robot stays one `.py` file. Documented in project_context 3.

- **Q:** Approve ADR 0020 (PyBullet as optional extra `physics`, conditional registration, separate Linux/3.11 CI job)? | **Raised by:** orchestrator, review 4, 2026-10-10 | **Affects:** pyproject, CI, discovery behaviour
  **Resolved 2026-10-10 (user approved):** optional `physics` extra, conditional registration, Linux/CPython 3.11 CI job. PyBullet installed in the project `.venv` only. ADR 0020 Accepted.

- **Q:** Param models accept NaN/inf (e.g. PID kp=NaN yields a "done" run). Fix requires a change to frozen core config validation (reject non-finite floats). Approve as an architecture change with an ADR? | **Raised by:** Phase 1 orchestrator, 2026-10-10 | **Affects:** rcl/core config, all param models
  **Resolved 2026-10-10 (user approved):** reject non-finite floats in `Entry.validate_params` (422 `finite_number`); no interface shape changed. ADR 0017.

- **Q:** Root `CLAUDE.md` §1.3, `rcl/core|controllers|environments/CLAUDE.md` and project_context §3 still say env/controllers import "only `core`"; registry decision (a) allows `core` + `rcl.registry` (ADR 0003/0011, enforced in pyproject). Approve rewording these rule files? | **Raised by:** orchestrator, 2026-10-10 | **Affects:** CLAUDE.md, rcl/*/CLAUDE.md
  **Resolved 2026-10-10 (user approved):** reworded to `core` + `registry`.

- **Q:** Environments/controllers may import only `core`, yet must call `@register_environment` from `rcl.registry`. Options: (a) allow env/controller -> `rcl.registry` (registry imports only core); (b) put decorator + store primitives in `core`. | **Raised by:** orchestrator/scaffold agent, 2026-10-10 | **Affects:** registry, import-linter, project_context.md §4. 
  **Decision (architecture review 1, 2026-10-10): (a).** Details and rationale: `docs/architecture/registry.md` section 1, ADR 0003.  `core` stays types-only (store, discovery and JSON-schema export are behaviour); the registry imports only `core` and loads env/controller modules by name (no static cycle). import-linter has a "registry imports only core" contract and a registry layer between env/controller and core. `project_context.md` section 4 updated.



<!-- Entry format:
- **Q:** <question>  | **Raised by:** <agent/date> | **Affects:** <module/phase>
-->
