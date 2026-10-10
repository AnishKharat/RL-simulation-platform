# Open Questions

> Agents: add ambiguities or architecture conflicts here and ask the user. Don't guess.
> When answered, move the entry to "Resolved" with the decision (and an ADR link if one exists).

## Open
## Resolved

- **Q:** Root `CLAUDE.md` §1.3, `rcl/core|controllers|environments/CLAUDE.md` and project_context §3 still say env/controllers import "only `core`"; registry decision (a) allows `core` + `rcl.registry` (ADR 0003/0011, enforced in pyproject). Approve rewording these rule files? | **Raised by:** orchestrator, 2026-10-10 | **Affects:** CLAUDE.md, rcl/*/CLAUDE.md
  **Resolved 2026-10-10 (user approved):** reworded to `core` + `registry`.

- **Q:** Environments/controllers may import only `core`, yet must call `@register_environment` from `rcl.registry`. Options: (a) allow env/controller -> `rcl.registry` (registry imports only core); (b) put decorator + store primitives in `core`. | **Raised by:** orchestrator/scaffold agent, 2026-10-10 | **Affects:** registry, import-linter, project_context.md §4. 
  **Decision (architecture review 1, 2026-10-10): (a).** Details and rationale: `docs/architecture/registry.md` section 1, ADR 0003.  `core` stays types-only (store, discovery and JSON-schema export are behaviour); the registry imports only `core` and loads env/controller modules by name (no static cycle). import-linter has a "registry imports only core" contract and a registry layer between env/controller and core. `project_context.md` section 4 updated.


<!-- Entry format:
- **Q:** <question>  | **Raised by:** <agent/date> | **Affects:** <module/phase>
-->
