# rcl/frontend — local rules

- Talks to the API over HTTP only. Never import backend modules.
- Zero environment- or controller-specific logic: dropdowns, parameter forms and plots are generated from API/registry metadata and schemas.
- Adding a robot or controller must require no change here. If it does, that is an architecture bug.
- Plotly for charts. Keep pages small; shared helpers in one module.
