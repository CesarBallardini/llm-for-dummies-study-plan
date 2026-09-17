# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

An MkDocs Material **book**, not a Python package (`[tool.uv] package = false`). The
markdown under `docs/` is the product. `STUDY_PLAN.md` is the single source of truth for
chapter scope, resources and URLs; `docs/index.md` plus 24 `docs/chapter-NN-*/index.md`
pages render it as a navigable course. Every chapter page follows the same skeleton
(What you will learn / Topics / Resources / Milestone / Estimated time) mirroring the
matching `## Chapter NN` section of `STUDY_PLAN.md` — when you edit one, keep the other
in sync.

The book is published at <https://katra.ballardini.com.ar/llm-for-dummies-study-plan/>.
`.github/workflows/deploy-book.yml` builds it with `mkdocs build --strict` and deploys to
GitHub Pages on every push to `main`. That domain is the custom domain of the user site
CesarBallardini.github.io and is inherited by every project site on the account, so this
repo holds no CNAME file; `site_url` in `mkdocs.yml` must stay equal to that URL, trailing
slash included, because canonical links, the sitemap and `404.html` are generated from it.

`src/llm_for_dummies/` contains only `__init__.py`; it exists so the type-check gates
have something to run against until chapter code lands. `books/free/*.pdf` is gitignored;
`books/free/README.md` is the tracked manifest of official sources and licenses (download
only from those official URLs, never from mirror sites).

The repo is MIT-licensed (`LICENSE`). The notice lives in `docs/license.md`, in the
`copyright` footer of `mkdocs.yml` — which renders on every page, which is why no chapter
file carries its own notice — in `README.md`, and in a paragraph near the top of both
`STUDY_PLAN.md` and `docs/index.md` (keep those two in sync like the chapters). The MIT
terms cover this book only; every linked course, book, paper and model keeps its own
license, which is why each resource is cited through its official page.

Each chapter's `## Resources` (docs page) / `**Resources.**` (matching `STUDY_PLAN.md`
section) follows a fixed shape: a `**Suggested path.**` paragraph naming what to use in
what order and what to skip, then `### University courses`, `### Online courses (MOOCs)`,
`### Books`, `### Lectures, papers and articles`, and (where relevant) `### Tools and
hands-on` groups, omitting any that would be empty. `STUDY_PLAN.md`'s Appendix A (book
shelf) and Appendix B (course shelf), and their mirror tables in `docs/index.md`, are
hand-maintained summaries of what recurs across chapters, not generated from the chapter
Resources sections — reconcile them by hand after any resource-list change spanning
several chapters, or they silently drift (they did, once, in this repo's history).

`AGENTS.md` holds the same orientation for other agents; keep the two consistent.

## Commands

Everything runs through `uv run --frozen` so the committed `uv.lock` is authoritative.
`make` targets are thin wrappers; the `help` and `clean` targets need `grep`/`awk`/`rm`,
so use Git Bash (or run the `uv run --frozen ...` line directly in PowerShell).

```sh
make install       # uv sync --frozen + pre-commit install (wires commit-msg and pre-push stages too)
make docs          # uv run --frozen mkdocs build --strict   <- the effective test suite
make docs-serve    # live reload at http://127.0.0.1:8000
make lint          # ruff check . && ruff format --check .
make format        # ruff format . && ruff check --fix .
make types         # pyright && pyrefly check  (both must pass; different engines on purpose)
make precommit     # pre-commit run --all-files (markdown-link-check hits the network)
make clean
```

There is no pytest suite yet. The pre-merge check is `make precommit`, plus `make docs`
whenever `docs/` or `mkdocs.yml` changed.

## MkDocs gotchas (these break the strict build)

- `strict: true` in `mkdocs.yml`: any warning fails `mkdocs build --strict`.
- Links that leave `docs/` (e.g. `../books/free/README.md`) are warnings. Mention sidecar files
  by path in prose instead of linking them.
- Chapter titles containing `:` (Chapter 9, Chapter 23) must stay double-quoted in the
  `nav` of `mkdocs.yml`.
- Social icons must come from bundled sets (`fontawesome/brands/*`, `simple/*`,
  `material/*`); anything else silently breaks the build.
- `mkdocs` is capped `<2` and `mkdocs-material>=9.7.5` deliberately (MkDocs 2.0 drops the
  plugin system and YAML config). Never raise the cap. Material prints a non-fatal
  "MkDocs 2.0" advisory on every build; ignore it.
- Every chapter cites many external URLs and the `markdown-link-check` pre-commit hook
  fails on dead ones. It runs URL checks concurrently and, in some sandboxed/CI network
  environments, that concurrency produces spurious `Status: 0`/403/429 failures on live
  URLs. Before changing or removing a flagged link, re-check it with a few sequential
  `curl -A "Mozilla/5.0" -L` retries a couple of seconds apart — most flagged links turn
  out to be transient, not dead.
- A wrapped line whose continuation starts with a bare `+ ` (e.g. a link followed by
  `+ assignments` at the wrap point) is parsed by Python-Markdown as a new list item, not
  as running prose — it silently breaks rendering without failing `--strict`. Prefer `and`
  over `+` as a conjunction at a line wrap, or keep the `+` mid-line.

## Python conventions (for code as it lands)

Ruff config in `ruff.toml`: line length 119, single quotes, Google-style docstrings (`D`),
pep8-naming, bandit (`S`), bugbear, SIM/RET/PIE/TCH, max complexity 15. Under `tests/**`
only, `S101` (assert) is allowed and docstring rules are off. `pyrefly.toml` scans `src`
and `tests`; `pyrightconfig.json` excludes `docs` and `site`.

## Git workflow

- The user runs every git command themselves. Make the edits, then hand over the exact
  commands, a `<type>/<slug>` branch name, and an explicit PR title.
- The local default branch is currently `master`; the `no-commit-to-branch` hook blocks
  `main`, and work is expected on a branch named `<type>/<slug>` with type in
  `build|bump|chore|ci|docs|feat|fix|perf|refactor|revert|style|test`
  (`.pre-commit-hooks/check-branch-name.sh`, enforced at pre-push; `main|master|HEAD`
  exempt).
- Commit messages are Conventional Commits, enforced at commit-msg by commitizen. A branch
  with more than one commit gets a GitHub-derived PR title that fails the convention, so
  always propose the PR title explicitly.
- Keep `uv.lock` in sync with `pyproject.toml`; a pre-commit hook runs `uv lock --check`.
