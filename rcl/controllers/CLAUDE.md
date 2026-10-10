# rcl/controllers — local rules

- One controller = one file, with `@register_controller("<id>")`. Layout: `classical/` (pid, lqr, mpc) and `rl/` (thin SB3 wrappers).
- Import only `rcl.core` and `rcl.registry` (+ third-party). Never reference a concrete environment; use `BaseEnvironment`/`EnvSpec` only.
- Declare supported action-space types (`discrete`/`continuous`), `requires_training`, and a Pydantic hyperparameter schema.
- Classical `train()` is a no-op. `save`/`load` must round-trip.
- Seeds passed explicitly; deterministic `predict` when `deterministic=True` (evaluation always uses it). RL controllers: `device` param defaults to `"cpu"` (ADR 0010); the determinism claim is CPU-only.
- Tests per controller: configure → predict shape, save/load round-trip, runs on at least one compatible env.
