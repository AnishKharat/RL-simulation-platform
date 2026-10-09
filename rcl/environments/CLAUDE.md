# rcl/environments — local rules

- One robot = one file, with `@register_environment("<id>")`. Nothing else may need editing; if it does, stop and raise an architecture issue.
- Import only `rcl.core` (+ third-party). Never import controllers, experiments, api or frontend.
- Follow the Gymnasium API. Implement `get_state()`, `get_task_spec()`, and `get_linearization()` (or return `None`).
- Declare metadata: description, dof, obs/action spec, default params, Pydantic param schema, physics backend.
- Seed only via `reset(seed=...)`; no global RNG.
- Tests per environment: reset/step shapes, seeded determinism, registry discovery.
