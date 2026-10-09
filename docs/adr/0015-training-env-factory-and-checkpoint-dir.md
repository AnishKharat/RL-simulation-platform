# 0015. Training env factory and checkpoint directory

- Status: Accepted
- Date: 2026-10-10

## Context
`train(env, config)` passed one environment instance, but RL training needs several (vectorised SB3
`n_envs`), possibly wrapped like the evaluation environment, and a stateful observation normaliser
(`VecNormalize`) whose statistics must travel with the checkpoint. `save(path)` was ambiguous: a file or a directory.

## Decision
- `BaseController.train(env_factory: Callable[[], BaseEnvironment], config: TrainConfig) -> TrainResult`. The
  experiment manager supplies a factory that builds the environment with the same wrapper stack (perturbations,
  noise) as evaluation; the controller calls it `TrainConfig.n_envs` times (default 1) and owns what it builds.
- `TrainResult.curve: tuple[tuple[int, float], ...] = ()` carries `(timestep, mean_episode_return)` pairs.
- The controller owns observation-normalisation state. It is fitted in `train`, persisted by `save`, restored by
  `load`, and applied inside `predict`, which receives the RAW observation. The rollout never normalises.
- `save(path)` receives a directory created by the caller and fills it (e.g. `model.zip`, `vecnorm.pkl`);
  `load(path)` reads one. Artifact references stored by persistence point at that directory.

## Consequences
- One evaluation loop and one `predict` signature work for every controller, normalised or not.
- Training and evaluation see identical environment dynamics by construction (fairness).
- Controllers cannot assume a single env instance; classical controllers ignore the factory.

## Alternatives
- Pass a gymnasium vector env: forces vectorisation policy on controllers and on the manager.
- Rollout normalises observations: couples evaluation to RL internals and breaks "all controllers evaluated identically".
- `save` to a file path: cannot hold multi-file checkpoints.
