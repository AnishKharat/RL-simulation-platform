# 0002. Environment/controller independence and declared compatibility

- Status: Accepted
- Date: 2026-10-10

## Context
The platform's value is combining any controller with any robot. Hardcoding pairs (or controller names in the
frontend) would force edits across modules for each new robot or controller (project rule: one file + decorator).

## Decision
- Environments and controllers never import each other (import-linter `independence` contract, ADR 0011).
  Controllers see an environment only through `BaseEnvironment` and the runtime `EnvSpec`.
- Compatibility is declared as data, not code:
  `EnvMetadata.action_space_kind` (`discrete | continuous`) and
  `ControllerMetadata.supported_action_spaces` (non-empty tuple).
- `BaseController.configure(env_spec, params)` must raise `ValueError` if the env's kind is unsupported.
- The registry derives `compatible_controllers(env_id)` / `compatible_environments(controller_id)` from the
  declarations; the API exposes them and the frontend filters dropdowns from that data only.
- Controllers needing more than action-space kind (e.g. LQR needs `get_linearization()`) check at `configure`
  and fail loudly; richer capability declaration is not yet decided (see Consequences).

## Revisions (architecture review 1)
- `requires_training` moved from a `BaseController` class attribute into `ControllerMetadata`, so the registry/API expose it without instantiation. Training hyperparameters are controller `params`; `TrainConfig`/`TrainingSection` hold only budget and seed.

## Consequences
- Adding a robot or controller is one file; no existing module changes.
- Action-space kind is a coarse compatibility test: it admits pairs that still fail (LQR on an env without a
  linearization). A finer capability set (e.g. `requires_linearization`) is a possible later extension; not
  decided here.
- Shared logic between the two sides lives in `core` or in a third module, never one importing the other.

## Alternatives
- Explicit compatibility matrix file: central edit per new component, violates the one-file rule.
- Try-and-catch at run time only: poor UX, no filtered dropdowns.
- Controllers subclassing per-robot interfaces: couples the two sides.
