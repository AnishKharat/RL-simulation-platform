# CLAUDE.md — Working Rules for Coding Agents

**Start of every session:** read `docs/STATUS.md` (current phase, next task). Read `project_context.md` only the sections you need (architecture §3–5, phases §7); it is the source of truth. Task checklists are in `docs/ROADMAP.md` (current phase only). Subfolders have their own short `CLAUDE.md` with local rules.

**Code questions:** if `graphify-out/` exists, query it (graphify) before grepping or reading many files.

**Repo:** https://github.com/AnishKharat/RL-simulation-platform.git (remote `origin`)

---

## 1. Ground Rules

1. **Do not implement until architecture is FROZEN** (`docs/architecture/STATUS.md`). During Phase 0, produce docs, diagrams, interface stubs and spikes only.
2. **One task at a time**, in the current phase only. No later-phase features early.
3. **Respect modularity** (`project_context.md` §3–4):
   - `core/` imports nothing from the project.
   - Environments/controllers import only `core`.
   - Frontend talks to the API only.
   - New robot or controller = 1 file + registry decorator. If your change needs edits elsewhere, stop and raise it as an architecture issue.
4. **No hardcoded environment/controller names** in the frontend, API or experiment manager. Use registry metadata.
5. **Every run is reproducible**: seed, config, library versions saved.
6. **Write tests with the code.** No task is done without tests.
7. **Never commit secrets, large binaries, model checkpoints or datasets.** Use `.gitignore`.
8. Ambiguity or architecture conflict → write it in `docs/OPEN_QUESTIONS.md` and ask. Don't guess.

---

## 2. Commands

```bash
pip install -e ".[dev]"      # install
pytest -q                    # tests
ruff check . && ruff format --check .   # lint
mypy rcl                     # types
lint-imports                 # dependency-rule check
streamlit run rcl/frontend/app.py       # dashboard (Phase 1+)
uvicorn rcl.api.main:app --reload       # API (Phase 1+)
```

Run tests + lint **before every commit**. Never commit failing code to `main`.

---

## 3. Git Protocol (MANDATORY)

Use the terminal for git. Commit **early and often**, after every completed logical step.

**Setup (first session only):**
```bash
git init && git branch -M main            # if not already a repo
git remote add origin https://github.com/AnishKharat/RL-simulation-platform.git   # skip if `git remote -v` shows it
# Create .gitignore (Python, venv, .env, __pycache__, mlruns/, *.db, models/, graphify-out/, .idea/, .vscode/)
git add .gitignore README.md project_context.md CLAUDE.md docs rcl
git commit -m "chore: initialize repository with project context and agent rules"
```

**Branching:** `main` is always green. Work on `phase-N/<short-feature>` (e.g. `phase-1/cartpole-env`). When done and tests pass: `git merge --no-ff <branch> -m "merge: <feature>"` (or PR), then delete the branch.

**When to commit** (one commit per item, never batch unrelated work): a new/changed interface or stub; a new module with first working behavior; a passing test group; a bug fix (+ regression test); a green refactor; a doc/ADR/diagram; a config/CI/tooling change; each `☐` in `docs/ROADMAP.md`.
Rule of thumb: **a coherent change that passes its checks gets committed**, typically every 1–3 steps. Never leave more than ~200 changed lines uncommitted.

**Message format (Conventional Commits):** `<type>(<scope>): <imperative summary, ≤72 chars>` + optional body.
Types: `feat fix docs test refactor perf chore ci build`.
Scopes: `core registry env controller experiments metrics analytics persistence api frontend docs ci`.
Example: `feat(controller): implement PID controller with anti-windup`

**Procedure:**
```bash
git status --short
pytest -q && ruff check .
git add <specific files>          # no blind `git add .`
git diff --cached --stat
git commit -m "<type>(<scope>): <summary>"
```

**Pushing:** push after each feature-branch merge and at the end of every session: `git push origin main --tags` (and `git push -u origin <branch>` for branches you want backed up). Never invent a remote URL.

**Tags:** at each phase exit, `git tag -a phase-N-complete -m "Phase N complete: <summary>"`, then rebuild the graphify graph.

**Hard rules:**
- Every commit has real, meaningful changes. No empty, whitespace-only or placeholder commits.
- Never backdate commits or alter author/committer dates.
- Never `push --force` to `main` unless the user explicitly asks.
- Don't squash many distinct steps into one giant commit.

---

## 4. End-of-Session Routine

1. Run full checks (`pytest`, `ruff`, `mypy`, `lint-imports`).
2. Commit finished work (never leave it uncommitted).
3. Overwrite `docs/STATUS.md` (done / next / blockers / check results) → commit `docs: update project status`.
4. Push to `origin`.
5. Tell the user briefly: commits made, test status, next task.

---

## 5. Code Standards

- Type hints on all public functions; Pydantic for configs and API schemas.
- Docstrings on all interfaces stating the contract.
- Small functions, no god classes, no global state outside the registry.
- Seeds passed explicitly; never rely on global RNG.
- Metrics are pure functions with documented definitions.
- Log with `logging`, not `print`.
