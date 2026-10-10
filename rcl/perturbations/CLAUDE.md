# rcl/perturbations — local rules

- One perturbation = one file with `@register_perturbation("<id>", params=Model)` (ADR 0021). Nothing else may need editing.
- Import only `rcl.core` and `rcl.registry` (+ third-party). Enforced by import-linter.
- `step` stage: subclass `StepPerturbation`; draw a fixed number of values per step from `self.rng` (seeded in `reset`), schedule events by step index, set `info["perturbed"]` on injection steps. Never use a global RNG.
- `params` stage: subclass `ParamPerturbation`; `apply` returns new env params and never mutates its input.
- Declare `perturbation_metadata` with `magnitude_param` and `nominal_magnitude`; the dashboard sweeps are built from it.
- Disturbances act through the actuated input and saturate at the actuator limit (the environment clips the sum); there is no external-force hook.
