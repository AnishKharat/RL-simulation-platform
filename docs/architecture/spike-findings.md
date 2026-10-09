# Spike Findings (Phase 0)

Code: `spikes/sb3_custom_env`, `spikes/fastapi_background_job` (throwaway). Run tests with
`pytest spikes/<name>` only; root `pytest` ignores them. Env: Windows 11, Python 3.12.2, CPU only.

Versions: stable-baselines3 2.9.0, torch 2.14.1+cpu, gymnasium 1.4.0, numpy 2.5.3,
fastapi 0.143.0, starlette 1.7.0, uvicorn 0.54.0, httpx 0.28.1, pydantic 2.14.0.

## 1. SB3 on a custom env: PASS

- A plain `gymnasium.Env` (Box obs/action, own dynamics, `reset(seed=)` via
  `super().reset(seed=seed)` + `self.np_random`) works with `PPO("MlpPolicy")` unchanged. No
  registration with Gymnasium needed; the env is passed as an instance.
- 4096 PPO steps on CPU: ~4.6 s (includes torch import warm-up). 4k steps does not learn the
  task (return -612 vs PID -0.11); fine for a pipeline check, so real training needs 100k+.
- A PID controller runs on the same env via the same `obs -> action` callable as
  `model.predict(obs, deterministic=True)[0]`. Suggests the `BaseController.predict(obs)` contract
  can serve both classical and RL controllers; the RL one is a thin wrapper around a loaded model.
- Reproducibility: same env seed -> bit-identical trajectory; same train seed + same versions ->
  bit-identical policy and rollout (CPU only). Save/load (`model.save`/`PPO.load(device="cpu")`)
  reproduces rollouts exactly.
- Recommendation: env stays SB3-agnostic (`core` + numpy/gymnasium only); an adapter in the
  experiments layer wraps `BaseEnvironment` as `gym.Env`. Always pass `seed=` and `device="cpu"`.

Gotchas / risks:
- `torch` is a heavy install (CPU wheel is ~200 MB; default PyPI wheel on Windows is CPU).
  Keep it in the `rl` extra and import lazily so the API/frontend start without it.
- Cold `import stable_baselines3` takes seconds; do not import at API module load.
- Determinism holds on CPU only for fixed versions; save library versions with each run
  (already a project rule). GPU/multi-thread torch can break bit-exactness.
- Observations are float32 in SB3; env dynamics in float64 internally, cast at the boundary.
- Windows: SB3 `SubprocVecEnv` uses spawn; envs must be importable/picklable top-level classes.
  Use `DummyVecEnv` (default) for the MVP.
- Model files are zip (pickle inside): never load untrusted checkpoints.

## 2. FastAPI background job: PASS

Both approaches pass: POST returns 202 + id immediately, GET shows queued -> running -> done,
progress 0..1, failure sets `failed` + error message, unknown id gives 404.

| | `BackgroundTasks` | `ThreadPoolExecutor` (module-level) |
|---|---|---|
| Returns before work | yes on real server; **TestClient blocks until task ends** | yes, also under TestClient |
| Concurrency cap / queue | none (unbounded jobs) | `max_workers` cap; extras stay `queued` |
| Handle for cancel/inspect | none | `Future` available |
| Lifecycle | tied to request/response | owned by app; shut down in lifespan |
| Testability | needs live uvicorn to see non-blocking | works with TestClient |

Recommendation: **ThreadPoolExecutor** (owned by an app-level `JobManager`, shutdown in
lifespan) for training jobs. A job store must be thread-safe (lock used in spike).

Risks / follow-ups:
- Threads share the GIL; SB3/torch release it in native code, but heavy pure-Python
  simulation can starve the API. If that bites, switch the executor to `ProcessPoolExecutor`
  (needs picklable job args; progress via `multiprocessing.Manager` or DB rows). Keep job
  function signature `(job_id, config) -> result` so this swap is local.
- In-memory job store is lost on restart; persist status/progress in the DB (Phase 1) and
  mark orphaned `running` jobs as `failed` on startup.
- Threads cannot be killed: cancellation must be cooperative (a flag checked in the training
  callback). Not spiked.
- Worker exceptions must be caught at the job boundary (done) or they vanish into the Future.
- Starlette 1.7 emits a deprecation warning: `TestClient` with httpx (suggests `httpx2`).
  Harmless now; watch when pinning versions.
