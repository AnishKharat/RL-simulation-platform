# 0014. Declared compatibility

- Status: Accepted
- Date: 2026-10-10

## Context
ADR 0002 declared compatibility by action-space kind only and left a finer capability set open. Architecture
review 2 found pairs that pass the filter and still fail: LQR on an environment without a linearization, a
scalar-output PID on a multi-input robot.

## Decision
Compatibility stays declared data, now on three axes, all evaluated by the registry (`_supports`):
- action kind: `EnvMetadata.action_space_kind` in `ControllerMetadata.supported_action_spaces` (unchanged);
- linearization: `ControllerMetadata.requires_linearization: bool = False` needs `EnvMetadata.linearizable: bool = False`;
- action dimension: `EnvMetadata.action_dim: int = 1` (number of action components; 1 for one discrete choice)
  must be `<= ControllerMetadata.max_action_dim` (`None` = unlimited).
`EnvMetadata.linearizable` is a declaration that must agree with `get_linearization() is not None`; each
environment's tests assert this (see `tests/core/test_environment.py` for the pattern). The `configure`-time
`ValueError` of ADR 0002 stays as the safety net.

## Consequences
- Dropdown filtering (`compatible_*`) hides pairs that cannot work; no controller or robot names appear anywhere.
- Defaults keep existing metadata valid. `action_dim` is a new fact environments must state for multi-input robots.
- Further axes (observation kind, partial observability) are added the same way when a real component needs them.

## Alternatives
- Free-form capability string sets: untyped, easy to misspell; three booleans/ints cover the known cases.
- Deriving `linearizable` by calling `get_linearization()` at registration: needs instantiation, which metadata
  must not (it is class-level).
