# rcl/environments — local rules

- One robot = one file, with `@register_environment("<id>")`. Nothing else may need editing; if it does, stop and raise an architecture issue.
- Import only `rcl.core`, `rcl.registry` and `rcl.physics` (+ third-party). Never import controllers, experiments, api or frontend.
- Simulator-backed robots (ADR 0020): never import the simulator at module top level; register only if it is installed (conditional decorator), so discovery works without the optional extra. Tests use `pytest.importorskip`.
- Model assets (URDF, meshes) may live in a sibling folder `rcl/environments/assets/<robot_id>/`; the robot is still one `.py` file.
- Follow the Gymnasium API. Implement `get_state()`, `get_task_spec()`, and `get_linearization()` (or return `None`).
- Declare `env_metadata` (`EnvMetadata`: name, description, dof, action kind/dim, labels/units, linearizable, physics backend). The params schema and defaults come from the Pydantic params model passed to the decorator (`@register_environment("<id>", params=Model)`), not from metadata.
- Seed only via `reset(seed=...)`; no global RNG.
- Tests per environment: reset/step shapes, seeded determinism, registry discovery.
