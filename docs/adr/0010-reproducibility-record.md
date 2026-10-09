# 0010. Reproducibility record

- Status: Accepted
- Date: 2026-10-10

## Context
Project rule: every run is reproducible. The SB3 spike showed bit-identical results for the same seeds on CPU
with fixed library versions, and that this can break across versions or on GPU/multi-thread torch.

## Decision
Every run stores a reproducibility block (core type `RunRecord`; DB columns on `RUN`/`EXPERIMENT`):
- `seed`: the authoritative root seed of the run (`config.seed` is only the experiment default); `derived_seeds` (env, train, eval) spawned with `numpy.random.SeedSequence(seed)`;
  evaluation episode `i` uses `eval_seed + i`. Seeds are passed explicitly, never via global RNG.
- `config`: the validated `ExperimentConfig` stored verbatim, with `config_hash` (sha256 of canonical JSON of the resolved config, `seed` and `name` excluded).
- `versions` (core `RunRecord.versions`, DB `RUN.versions`): python, numpy, gymnasium, pydantic,
  stable-baselines3, torch, rcl, platform, via `importlib.metadata`; registry entry versions (metadata `version`)
  are included as `env:<id>` / `controller:<id>`.
- `code_version`: git commit sha, suffixed `-dirty` when the tree had uncommitted changes. The diff is not stored.
- RL runs use `device="cpu"` and a fixed seed; bit-exactness is only claimed for identical versions on CPU.

## Consequences
- Reproduction is "same config + same versions + same commit"; a dirty flag only warns.
- Core `RunRecord` and the data model now use the same names (`versions`, `derived_seeds`, `config_hash`).
- Environment lockfile or container capture is not included; consider if version drift becomes a problem.

## Alternatives
- Seed only: fails silently when libraries change.
- Storing the full diff or a container image per run: heavy and not needed for MVP.
- Per-component seeds in the config: more knobs, easier to misconfigure; derivation from one seed is simpler.
