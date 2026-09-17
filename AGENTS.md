# AGENTS.md

Compact orientation for agents working in this repo. Verified against the current tree
(no commits yet; `git branch --show-current` may report `main` even though `master` is
the intended local default — check before assuming, and never commit straight to `main`,
see Git conventions below).

## What this repo is

- An MkDocs **book** (`docs/`, 24 chapter folders + `docs/index.md`) that renders
  `STUDY_PLAN.md` as a course: college algebra -> building/training/operating open LLMs.
  The markdown is the product; `STUDY_PLAN.md` is the single source of truth for chapter
  scope, resources, and URLs.
- NOT a Python package: `[tool.uv] package = false`. `src/llm_for_dummies/` has only
  `__init__.py`, which exists solely to keep pyright/pyrefly green until chapter code lands.
- `books/free/` holds downloaded reference PDFs (gitignored); `books/free/README.md` is the
  tracked manifest of official sources/licenses. Only download from official sources —
  never the z-library/archive mirrors.

## Commands

- The Makefile (`lint`, `format`, `types`, `docs`, `docs-serve`, `precommit`, `clean`) is
  the single interface, but it uses `grep`/`rm` so it only works from Git Bash/WSL, not
  pwsh. On Windows run the equivalent `uv run --frozen <tool> ...` directly.
- Everything runs `uv run --frozen` — the lockfile is authoritative. Keep `uv.lock` in sync
  with `pyproject.toml` (a pre-commit hook enforces `uv lock --check`).
- Full pre-merge check: `make precommit` (runs all hooks incl. `markdown-link-check`, which
  hits the network) — plus `make docs` if `docs/` changed.

## Docs (MkDocs) gotchas

- `strict: true` in `mkdocs.yml`: **warnings are build failures**. `mkdocs build --strict`
  (`make docs`) is effectively this repo's test.
- `mkdocs` is pinned `>=1.6,<2` and `mkdocs-material>=9.7.5` on purpose: MkDocs 2.0 removes
  the plugin system and YAML config. Never bump the cap. Material prints a non-fatal
  "MkDocs 2.0" advisory on every build — ignore it.
- Links that leave `docs/` (e.g. `../books/free/README.md`) fail the strict build. Reference
  sidecar files by path in prose, not as markdown links.
- YAML string gotcha: chapter titles containing `:` ("Chapter 9 - Deep Learning: Concepts
  and Practice", "Chapter 23 - Capstone: Train, Ship and Improve Your Own Model") must stay
  double-quoted in `mkdocs.yml` nav and in `docs/index.md`.
- Social icons only work for bundled sets: `fontawesome/brands/*`, `simple/huggingface`,
  `material/*`. Other forms silently break the build.
- Chapter pages have a fixed skeleton (What you will learn / Topics / Resources / Milestone /
  Estimated time) mirroring `STUDY_PLAN.md`; keep them in sync when editing either.
- Resources is itself structured: a `**Suggested path.**` paragraph, then `###` groups —
  University courses / Online courses (MOOCs) / Books / Lectures, papers and articles /
  Tools and hands-on — omitting empty ones. `STUDY_PLAN.md` Appendix A/B and their
  `docs/index.md` mirror tables are hand-maintained summaries, not generated; reconcile
  them by hand after any multi-chapter resource change.
- A wrapped continuation line starting with a bare `+ ` parses as a new Markdown list item
  (silent rendering bug, not caught by `--strict`); use `and` at a line wrap instead.
- `markdown-link-check` runs URL checks concurrently; in some sandboxed environments that
  produces spurious `Status: 0`/403/429 on live URLs. Re-verify a flagged link with a few
  sequential `curl -A "Mozilla/5.0" -L` retries before treating it as dead.

## Python tooling (when code lands)

- Ruff is strict: Google docstrings (`N`, `D`), bandit (`S`), bugbear, SIM/RET/PIE, single
  quotes, line length 119, max-complexity 15. In `tests/**` only: `S101` allowed and `D`
  off (`ruff.toml` `per-file-ignores`).
- TWO type checkers must pass: `pyright` + `pyrefly check` (different engines on purpose).
  Both are currently green on the empty package.

## Git / commit conventions

- Work on a feature branch, never `main`. Expected pattern: `<type>/<slug>` with type in
  `build|bump|chore|ci|docs|feat|fix|perf|refactor|revert|style|test` (enforced at pre-push
  by `.pre-commit-hooks/check-branch-name.sh`; `main|master|HEAD` are exempt).
- Commits are Conventional Commits, enforced at commit-msg by commitizen. Branch names with
  multiple commits drive the GitHub-defaulted PR title, so a multi-commit branch needs an
  explicit PR title.
- User runs all git commands themselves; propose branch name + PR title + exact commands.
