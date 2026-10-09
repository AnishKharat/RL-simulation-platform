# 0012. `env_metadata` naming vs gym `metadata`, and `get_env_spec`

- Status: Accepted (naming); registry reads `env_metadata` for environments with an isinstance check (review 1, R1)
- Date: 2026-10-10

## Context
`gymnasium.Env.metadata` is already a class-level dict (render modes, fps). Our static description of an
environment (name, DOF, action-space kind, params schema) would collide with it if it used the same name.
Controllers have no such clash. Separately, controllers need a runtime description of a built instance.

## Decision
- `BaseEnvironment.env_metadata: ClassVar[EnvMetadata]` holds our static, instantiation-free description.
  Gymnasium's `metadata` dict stays untouched for render modes.
- `BaseController.metadata: ClassVar[ControllerMetadata]` keeps the plain name (no clash).
- `EnvMetadata` (class level, drives registry/forms) is distinct from `EnvSpec` (instance level: shapes, bounds,
  `dt`, `n_actions` or `action_low/high`), returned by the abstract method `get_env_spec()`. The method is named
  `get_*` to match `get_state()`, `get_task_spec()` and avoid clashing with gym attributes `observation_space`
  and `action_space`.
- The registry must read `env_metadata` for environments and `metadata` for controllers.

## Consequences
- No accidental overwrite of gym render metadata; both can coexist on one class.
- Asymmetric attribute names (`env_metadata` vs `metadata`) must be handled in the registry.
- Done in review 1: `rcl/registry/registry.py` reads `env_metadata` (environments) / `metadata` (controllers) and
  rejects classes whose attribute is not an `EnvMetadata` / `ControllerMetadata` instance.

## Alternatives
- Overload gym `metadata` with our keys: mixes concerns, risks breaking render tooling.
- Name both `controller_metadata`/`env_metadata`: symmetric but renames an unambiguous attribute.
- Derive `EnvSpec` from gym spaces automatically: possible helper later, but `state_dim` and `dt` are not gym spaces.
