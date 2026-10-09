# Open Questions

> Agents: add ambiguities or architecture conflicts here and ask the user. Don't guess.
> When answered, move the entry to "Resolved" with the decision (and an ADR link if one exists).

## Open

_None._

## Resolved

- **Q:** Environments/controllers may import only `core`, yet must call `@register_environment` from `rcl.registry`. Options: (a) allow env/controller -> `rcl.registry` (registry imports only core); (b) put decorator + store primitives in `core`. | **Raised by:** orchestrator/scaffold agent, 2026-10-10 | **Affects:** registry, import-linter, project_context.md §4. 
  **Decision (architecture review 1, 2026-10-10): (a).** Details and rationale: `docs/architecture/registry.md` section 1, ADR 0003.
  **Recommendation (registry agent):** (a). `core` stays types-only; store/discovery/JSON-schema export are behaviour. Registry imports only `core` and loads env/controller modules by name (no static cycle). import-linter updated: registry layer sits between env/controller and core, plus a "registry imports only core" contract. `project_context.md` section 4 updated. Details: `docs/architecture/registry.md` section 1.


<!-- Entry format:
- **Q:** <question>  | **Raised by:** <agent/date> | **Affects:** <module/phase>
-->
