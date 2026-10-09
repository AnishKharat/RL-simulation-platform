# Sequence: Run Experiment

Covers `POST /experiments` through result display. Design rules: project_context §5.4 (config), §6.8-6.9 (fairness, background jobs), principle 5 (reproducibility). Endpoint names follow Phase 1: `/experiments`, `/experiments/{id}`.

## Status states

```mermaid
stateDiagram-v2
    [*] --> queued: POST /experiments (config valid)
    queued --> running: worker picks job
    running --> done: all stages succeed
    running --> failed: any stage raises
    queued --> failed: worker cannot start (e.g. unknown id)
    done --> [*]
    failed --> [*]
```

`failed` rows keep `error` (type, message, traceback) and the stage reached. No automatic retry; a re-run creates a new experiment with the same config and seed.

## Main flow

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant FE as Frontend
    participant API as API
    participant W as Background worker
    participant EM as ExperimentManager
    participant REG as Registry
    participant ENV as Environment
    participant CTL as Controller
    participant MET as Metrics
    participant DB as Persistence

    U->>FE: choose env, controller, params, seed
    FE->>API: POST /experiments (config JSON)
    API->>API: validate config (Pydantic)
    API->>DB: insert Experiment + Run(status=queued), config verbatim
    API->>W: submit(run_id)
    API-->>FE: 202 {run_id, status: queued}

    W->>DB: status = running, started_at
    W->>EM: execute(run_id)
    EM->>DB: load config
    EM->>EM: record versions (python, libs, git sha)
    EM->>REG: get_environment(env.id), get_controller(ctl.id)
    REG-->>EM: classes + parameter schemas
    EM->>EM: validate params against schemas, check action-space compatibility
    EM->>EM: derive seeds from experiment.seed
    EM->>ENV: create(params)
    EM->>CTL: configure(env_spec, params)

    opt controller.requires_training
        EM->>CTL: train(env, TrainConfig, seed)
        CTL->>ENV: reset(seed) / step(action) ...
        CTL-->>EM: TrainResult (curves, checkpoint)
        EM->>DB: save training artifact refs
    end

    EM->>CTL: evaluate(env, n_episodes, eval_seed)
    loop each episode i, reset seed = eval_seed + i
        CTL->>ENV: reset(seed)
        CTL->>ENV: step(predict(obs)) until terminated or truncated
    end
    CTL-->>EM: list of EpisodeTrajectory

    EM->>MET: compute(trajectory, task_spec) per episode
    MET-->>EM: MetricResult values
    EM->>DB: save trajectories as artifacts, metric rows, status = done, finished_at

    loop poll every 1-2 s
        FE->>API: GET /experiments/{id}
        API->>DB: read run + metrics
        API-->>FE: {status, progress, metrics?}
    end
    FE-->>U: plots and metric table (status = done)
```

## Seeding

- One `experiment.seed` is stored in the config. The manager derives independent seeds with `numpy.random.SeedSequence(seed).spawn(...)`: one for env construction, one for training, one for evaluation.
- Evaluation episode `i` resets with `eval_seed + i` (or the i-th spawned child), so every controller in a benchmark sees identical initial states (§6.8).
- No global RNG is used; seeds are passed explicitly to `reset`, `train`, and `evaluate`. The derived seeds are written to the Run row.

## Failure path

```mermaid
sequenceDiagram
    autonumber
    participant W as Background worker
    participant EM as ExperimentManager
    participant CTL as Controller
    participant DB as Persistence
    participant API as API
    participant FE as Frontend

    W->>EM: execute(run_id)
    EM->>CTL: train / evaluate
    CTL--xEM: exception (NaN loss, env error, bad params)
    EM->>EM: close env, release resources
    EM->>DB: status = failed, error, stage, finished_at
    EM-->>W: raises, worker logs with logging and continues
    FE->>API: GET /experiments/{id}
    API-->>FE: {status: failed, error: "..."}
```

Rules:
- Config or registry-lookup errors are caught before work starts: `POST` returns 422 and nothing is queued. Errors discovered inside the worker (unknown id, schema mismatch) produce `queued -> failed`.
- The worker never crashes on one job; every exception is converted to a `failed` row, so polling always terminates.
- Partial artifacts written before the failure are kept and linked to the Run for debugging.
- Server restart while `running`: on startup, runs left in `running` or `queued` are marked `failed` (reason `interrupted`) in the in-process worker design; a real queue replaces this later.
- A single episode that hits the time limit is `truncated`, not a failure. Only exceptions fail a run.
