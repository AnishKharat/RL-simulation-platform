# Registry design (Phase 0)

Status: accepted (architecture review 1). Code: `rcl/registry/`. Spec source: `project_context.md` 5.3.

## 1. Import-rule decision

Environments/controllers must call a decorator, so they must import something. Options:

- (a) `environments`/`controllers` may import `rcl.registry`; `registry` imports only `core`.
- (b) Move decorator + store primitives into `core`.

**Decision: (a).** Reasons: `core` stays "interfaces and types only" (its CLAUDE.md), whereas a store, discovery and
the JSON-schema export are behaviour, not types. (b) would make `core` hold global mutable state and `pkgutil`
logic, and the registry module would then be an empty facade. (a) costs one allowed edge, which keeps layering
downward-only: `experiments/api -> environments|controllers -> registry -> core`. Registry never imports
env/controller modules statically; it loads them by name via `importlib` (discovery), so no cycle exists.
Enforced in `pyproject.toml`: the `registry` layer sits between `environments|controllers` and `core`, and a
contract says registry imports only core. `project_context.md` section 4 states the same rule.

## 2. Decorator API

```python
@register_environment("cartpole", params=CartPoleParams)  # params: optional Pydantic model class
class CartPole(BaseEnvironment): ...


@register_controller("pid", params=PIDParams)
class PID(BaseController): ...
```

The decorator returns the class unchanged. Validation happens at decoration time (fail at import, not at run time):
id matches `^[a-z][a-z0-9_]*$`; the class has the metadata attribute, of the right type (environments: `cls.env_metadata` must be an `EnvMetadata`; controllers: `cls.metadata` must be a `ControllerMetadata`, checked with `isinstance`, so gymnasium's own `metadata` dict can never be mistaken for it, ADR 0012); `params` (if given) is a `BaseModel` subclass with `extra="forbid"` (a typo in a param must fail, not be ignored).
The registered class must also subclass the base of its kind (`BaseEnvironment` for environments, `BaseController` for controllers), so a decorated class that the experiment manager cannot use fails at import. `metadata.name` is NOT required to equal the id: the id is the stable registry key used in configs and URLs, `name` is the display label (it may contain spaces and capitals or be renamed without breaking stored configs).
Violations raise `RegistryError` (a duplicate id raises `DuplicateIdError`).

**Constructor contract.** The experiment manager builds an environment as `cls(**validated_params)` and a controller as `cls()` followed by `configure(env_spec, validated_params, linear_model)`; `validated_params` is `Entry.validate_params(raw)`, i.e. the params model dumped in JSON mode. Consequently (1) every environment's `__init__` must accept exactly the params model's fields as keyword arguments; (2) JSON has no tuples, so the params model (not the class) coerces JSON lists to the tuple/list types it declares (pydantic does this on validation) and the class receives what `model_dump(mode="json")` yields: lists. Environments must therefore tolerate list values for tuple-typed params (convert with `tuple(...)` if immutability matters); (3) no seed is passed to the constructor (seeds enter only through `reset(seed=...)`, ADR 0010).

## 3. Entry contents

`Entry(kind, id, cls, metadata, params_model)`; `metadata` is `cls.env_metadata` for environments and `cls.metadata` for controllers (core's `EnvMetadata` /
`ControllerMetadata`). `requires_training` is part of `ControllerMetadata` (no class attribute), so it appears in `info()["metadata"]`. Training hyperparameters live in the controller's own `params` model (one schema per controller); `TrainConfig` / `TrainingSection` carry only the budget (`total_timesteps`) and seed. `Entry.info()` returns the API-facing dict:
`{id, kind, metadata: <model_dump>, params_schema: params_model.model_json_schema() or {}}` so the frontend renders
forms from `params_schema` with no per-robot code. `Registry.entries(kind)` lists; `Entry.validate_params(raw)` validates through the model and returns the RESOLVED params (defaults filled, `model_dump(mode="json")`); those, not the raw input, are what the experiment manager stores and hashes. An entry without a params model accepts only `{}`.

**Supported params subset (checked at register time):** every field of a params model must be a scalar (`str`, `int`, `float`, `bool`), an `Enum`/`Literal`, an optional (`X | None`) of those, or a tuple/list of numbers or strings (use `Field(ge=, le=, ...)` constraints; they reach the JSON schema and become form limits). Nested models, dicts, lists of objects and nested arrays raise `RegistryError` at decoration: the frontend renders forms from the schema and has no generic object editor, so the subset is what keeps "no per-robot frontend code" true.

**Finite numbers (ADR 0017):** `validate_params` also rejects NaN/inf anywhere in the resolved params (`type: "finite_number"`).

**Param errors:** `Entry.validate_params` wraps pydantic's `ValidationError` into `ParamValidationError` (a `RegistryError` and `ValueError`) whose `.errors` is a JSON-safe list of `{loc, msg, type}` (`loc` a list of field path parts), which the API returns as its 422 body, so the frontend can mark the offending fields.

**Single source of truth for parameters:** `EnvMetadata` and `ControllerMetadata` carry no `params_schema` / `default_params`. The JSON schema and defaults come only from the Pydantic params model passed to the decorator.

**From core:** the registry imports `EnvMetadata` / `ControllerMetadata` from `rcl.core.types` (allowed edge registry -> core) and reads `action_space_kind` (env) and `supported_action_spaces` (controller) from them directly.

## 4. Compatibility filtering

Declared, never hardcoded (ADR 0014): `compatible_controllers(env_id)` returns controllers where the env's
`action_space_kind` is in `supported_action_spaces`, `requires_linearization` implies the env is `linearizable`,
and `EnvMetadata.action_dim <= max_action_dim` (when set); `compatible_environments(controller_id)` is the inverse, and `is_compatible(controller_id, env_id)` is the single-pair check used to reject a bad pair before queuing. The API exposes
the filters so the dropdown is built from registry data only.

## 5. Discovery

`discover(packages=DISCOVERY_PACKAGES)` (one constant: `rcl.environments`, `rcl.controllers`, `rcl.perturbations`, `rcl.ros_adapter`; a test keeps it equal to the import-linter plugin tier) walks each package with `pkgutil.walk_packages` and
imports every module; importing runs the decorators. Subpackages (`controllers.classical`, `controllers.rl`) are
covered by the recursive walk. Idempotent (Python caches imports). A module that fails to import raises
`RegistryError` naming that module (not only the package) and chaining the cause: a broken robot file is loud, not silently missing.
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
`GET /controllers/compatible?env_id=` uses the filter (full surface: `api.md`). The API imports `rcl.registry` only, calls
`discover()` in its startup hook, and never names a concrete robot or controller.
