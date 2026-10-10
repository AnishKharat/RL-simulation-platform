# 0017. Reject non-finite parameters at the registry

- Status: Accepted
- Date: 2026-10-10

## Context
Phase 1 review: `POST /experiments` with `{"kp": NaN}` returned 202 and the run finished `done`.
Pydantic accepts NaN/inf for `float`, JSON parsers in Python accept the `NaN`/`Infinity` literals, and the
resolved params are then stored, hashed (`config_hash`) and used in the simulation. There is no shared base class
for params models (each is a plain `BaseModel` registered with the decorator), so a base-model flag cannot cover
them, and plugin authors should not have to remember one.

## Decision
`Entry.validate_params` (the single place every config passes: API 422 path, experiment manager, job worker) checks the
resolved JSON-mode params recursively (lists/tuples included) and raises `ParamValidationError` with one
`{loc, msg: "must be a finite number", type: "finite_number"}` item per offending field. The API already turns that
into a 422 with `loc` prefixed by `["body", <environment|controller>, "params"]`. Behaviour only: no change to
`rcl/core`, `ExperimentConfig`, the registry decorator, `Entry` or any schema shape.

## Consequences
- NaN/inf never reach a stored config, hash or simulation, whichever params model is used, with no per-model code.
- Model-level `allow_inf_nan=False` is still allowed but unnecessary. The JSON schema does not advertise the rule
  (JSON has no non-finite numbers), so the form renderer needs no change.
- Only params are covered; `ExperimentConfig.seed`/`evaluation` are integers and cannot be non-finite.

## Alternatives
- Require `allow_inf_nan=False` on every params model (register-time check): stricter, but changes the plugin contract.
- Change `ComponentConfig` in core: rejected, core is frozen and it does not know the params types.
