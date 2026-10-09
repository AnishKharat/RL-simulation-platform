# 0003. Decorator registry, pkgutil discovery, registry import rule

- Status: Accepted (design); import rule is option (a), listed in `docs/OPEN_QUESTIONS.md` pending review
- Date: 2026-10-10

## Context
New robots/controllers must plug in with one file and a decorator, and the API/frontend must list them without
naming any. Environments/controllers may import only `core`, yet must call a decorator from somewhere.
Full design: `docs/architecture/registry.md`.

## Decision
- `@register_environment(id, params=PydanticModel)` and `@register_controller(...)` in `rcl.registry` store the
  class unchanged. Validation happens at decoration time: id matches `^[a-z][a-z0-9_]*$`, class has the metadata
  attribute, `params` is a `BaseModel`. Errors: `RegistryError`, `DuplicateIdError`, `UnknownIdError`.
- `discover()` walks `rcl.environments` and `rcl.controllers` with `pkgutil.walk_packages`, importing every module
  so decorators run. An import failure raises `RegistryError` naming the module (loud, not silently missing).
  Called once at API/experiment-manager startup, never on registry import.
- `Entry.info()` returns `{id, kind, metadata, params_schema}` (JSON schema derived from the params model, the single source of truth; metadata carries no schema; params models must set `extra="forbid"`; `validate_params` returns resolved JSON-safe params) so the
  frontend renders forms generically.
- Import rule, option (a): `environments|controllers -> registry -> core`. Registry imports only `core` and loads
  modules by name via `importlib`, so there is no static cycle. Rejected (b): moving decorator and store into `core`.
- The default `Registry` instance is the only global state; tests swap it with `use_registry(Registry())`.

## Consequences
- One allowed extra edge; `project_context.md` section 4 needs the matching wording (env/controllers import
  `core` and `registry`). `pyproject.toml` already encodes it (ADR 0011).
- Optional heavy dependencies (SB3/torch) must be lazy-imported inside methods so discovery stays fast.
- Duplicate ids fail at import time, which makes broken plugins visible at startup.

## Alternatives
- (b) store and decorator in `core`: puts global mutable state and discovery logic in an interface-only package.
- Entry points / plugin packaging: more machinery than needed for in-repo components; revisit for third-party plugins.
- Explicit central list of components: every new robot edits it, violating the one-file rule.
