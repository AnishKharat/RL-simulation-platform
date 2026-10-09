# Registry design (Phase 0)

Status: proposed. Code: `rcl/registry/`. Spec source: `project_context.md` 5.3.

## 1. Import-rule decision

Environments/controllers must call a decorator, so they must import something. Options:

- (a) `environments`/`controllers` may import `rcl.registry`; `registry` imports only `core`.
- (b) Move decorator + store primitives into `core`.

**Decision: (a).** Reasons: `core` stays "interfaces and types only" (its CLAUDE.md), whereas a store, discovery and
the JSON-schema export are behaviour, not types. (b) would make `core` hold global mutable state and `pkgutil`
logic, and the registry module would then be an empty facade. (a) costs one allowed edge, which keeps layering
downward-only: `experiments/api -> environments|controllers -> registry -> core`. Registry never imports
env/controller modules statically; it loads them by name via `importlib` (discovery), so no cycle exists.
Enforced in `pyproject.toml`: the `registry` layer now sits between `environments|controllers` and `core`, and a new
contract says registry imports only core. `project_context.md` 4 needs the matching one-line edit
("environments and controllers import only `core` and `registry`"); that file is not mine to edit.

## 2. Decorator API

```python
@register_environment("cartpole", params=CartPoleParams)  # params: optional Pydantic model class
class CartPole(BaseEnvironment): ...


@register_controller("pid", params=PIDParams)
class PID(BaseController): ...
```

The decorator returns the class unchanged. Validation happens at decoration time (fail at import, not at run time):
id matches `^[a-z][a-z0-9_]*$`; the class has the metadata attribute, of the right type (environments: `cls.env_metadata` must be an `EnvMetadata`; controllers: `cls.metadata` must be a `ControllerMetadata`, checked with `isinstance`, so gymnasium's own `metadata` dict can never be mistaken for it, ADR 0012); `params` (if given) is a `BaseModel` subclass.
Violations raise `RegistryError`.

## 3. Entry contents

`Entry(kind, id, cls, metadata, params_model)`; `metadata` is `cls.env_metadata` for environments and `cls.metadata` for controllers (core's `EnvMetadata` /
`ControllerMetadata`). `requires_training` is part of `ControllerMetadata` (no class attribute), so it appears in `info()["metadata"]`. Training hyperparameters live in the controller's own `params` model (one schema per controller); `TrainConfig` / `TrainingSection` carry only the budget (`total_timesteps`) and seed. `Entry.info()` returns the API-facing dict:
`{id, kind, metadata: <model_dump>, params_schema: params_model.model_json_schema() or {}}` so the frontend renders
forms from `params_schema` with no per-robot code. `Registry.entries(kind)` lists; `Entry.validate_params(raw)` validates through the model.

**From core:** the registry imports `EnvMetadata` / `ControllerMetadata` from `rcl.core.types` (allowed edge registry -> core) and reads `action_space_kind` (env) and `supported_action_spaces` (controller) from them directly.

## 4. Compatibility filtering

Declared, never hardcoded: `compatible_controllers(env_id)` returns controllers whose `supported_action_spaces`
contains the env's `action_space_kind`; `compatible_environments(controller_id)` is the inverse. The API exposes
both so the dropdown filters from registry data only.

## 5. Discovery

`discover(packages=("rcl.environments", "rcl.controllers"))` walks each package with `pkgutil.walk_packages` and
imports every module; importing runs the decorators. Subpackages (`controllers.classical`, `controllers.rl`) are
covered by the recursive walk. Idempotent (Python caches imports). A module that fails to import raises
`RegistryError` naming the module and chaining the cause: a broken robot file is loud, not silently missing.
Optional heavy deps (SB3) are the module's problem: lazy-import them inside methods, not at top level.
Called once by the API/experiment manager at startup, never from the registry import itself.

## 6. Errors

| Situation | Behaviour |
|---|---|
| Duplicate id within a kind | `DuplicateIdError` at decoration (the same class re-registered, compared by `__module__` + `__qualname__` so a reloaded module is fine, is a no-op) |
| Same id across kinds | allowed (kinds are separate namespaces) |
| Bad id / missing metadata / non-Pydantic params | `RegistryError` at decoration |
| Unknown id on lookup | `UnknownIdError` (a `KeyError` subclass) listing known ids |
| Import failure in discovery | `RegistryError` naming the module |

## 7. Global state and test isolation

The only global state in the project is the default `Registry` instance in `rcl.registry`. Decorators write to
`get_registry()`. Tests use `use_registry(Registry())`, a context manager that swaps in a fresh instance and restores
the previous one, so nothing leaks between tests. Library code that needs a registry takes it by parameter
(defaulting to `get_registry()`).

## 8. How the API lists entries

`GET /environments`, `GET /controllers` return `[entry.info() ...]`;
`GET /environments/{id}/compatible-controllers` uses the filter. The API imports `rcl.registry` only, calls
`discover()` in its startup hook, and never names a concrete robot or controller.
