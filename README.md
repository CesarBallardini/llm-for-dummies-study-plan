# LLM for Dummies

A book-style course that starts at college algebra and ends at building, training,
operating and improving open-source large language models. The full syllabus is
described in [STUDY_PLAN.md](STUDY_PLAN.md); this repository renders it as a
navigable MkDocs book.

Read it at **<https://katra.ballardini.com.ar/llm-for-dummies-study-plan/>**. Every
push to `main` rebuilds and republishes it through the `Deploy book to Pages`
workflow; the build runs `mkdocs build --strict`, so a dead nav entry fails the
deploy rather than reaching the site.

## Requirements

- [uv](https://docs.astral.sh/uv/) (manages Python 3.14 and the tooling)
- [GNU make](https://www.gnu.org/software/make/) for the `Makefile` targets
  (Windows: install via `choco install make` or use Git Bash, which ships it)

## Commands

Everything goes through the Makefile:

```powershell
# Create the environment (downloads Python 3.14 if missing) and install git hooks
uv sync
uv run pre-commit install

uv run mkdocs serve        # serve the book locally at http://127.0.0.1:8000
uv run mkdocs build        # build a static site into ./site (for publishing)
uv run mkdocs build --strict  # warnings become errors (also: make docs)
```

The Makefile is the single interface:

```powershell
make                # list every available target
make lint           # ruff check + ruff format --check
make format         # ruff format + ruff check --fix
make types          # pyright + pyrefly
make docs           # mkdocs build --strict
make docs-serve     # mkdocs serve, with live reload
make precommit      # run all pre-commit hooks by hand
make clean          # remove build, cache and coverage artifacts
```

All targets run through `uv run --frozen`, so the exact lockfile is used and
nothing re-resolves on its own. The pre-commit hooks (lint, formatting,
lockfile sync, dead-link check on the book's cited URLs, Conventional Commits)
run automatically on every commit.

## Tooling

Adopted from [localenv-python](https://github.com/CesarBallardini/localenv-python):

- **uv** for dependency and environment management (Python 3.14).
- **ruff** as linter and formatter (`ruff.toml`).
- **pyright** + **pyrefly** as a pair of type checkers (`pyrightconfig.json`,
  `pyrefly.toml`) — active for Python code as it lands in later chapters.
- **pre-commit** hooking lint, format, lockfile, dead-link and commit-message
  checks before every commit.
- **commitizen** for [Conventional Commits](https://www.conventionalcommits.org/).
- **MkDocs + Material** for the book, built with `--strict`.

The full backend scaffold (security scans, pytest gates, GitHub Actions CI) is
deliberately deferred until real code lands.

## Layout

| Path | Purpose |
| --- | --- |
| `docs/` | Book content, one folder per chapter |
| `docs/index.md` | Home page and how-to-use guide |
| `docs/chapter-NN-*/index.md` | The 24 chapters |
| `src/llm_for_dummies/` | Runnable Python code that accompanies the chapters |
| `mkdocs.yml` | Book navigation and theme |
| `pyproject.toml` | uv-managed Python 3.14 project |
| `Makefile` | Single interface for every command |
| `.github/workflows/` | Builds the book and deploys it to GitHub Pages |
| `LICENSE` | MIT License covering the book text and its code |

## License

Released under the [MIT License](LICENSE) — the book text, the study plan and the
code that accompanies the chapters.

The resources the book links to are not covered by it: every course, textbook,
paper, video, dataset and model keeps the license of its own author or publisher,
and each is cited through its official page so the terms can be checked there. The
PDFs downloaded into `books/free/` are covered by the licenses listed in that
folder's `README.md`.
