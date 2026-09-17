.DEFAULT_GOAL := help

.PHONY: help install lint format types docs docs-serve precommit clean

help: ## Show this list of available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Sync the environment (from the committed lockfile) and install the git hooks
	uv sync --frozen
	uv run --frozen pre-commit install

lint: ## Check formatting and lint rules without modifying files
	uv run --frozen ruff check .
	uv run --frozen ruff format --check .

format: ## Auto-fix formatting and lint issues
	uv run --frozen ruff format .
	uv run --frozen ruff check --fix .

types: ## Run both type checkers (pyright + pyrefly)
	uv run --frozen pyright
	uv run --frozen pyrefly check

docs: ## Build the documentation site (--strict: warnings are failures)
	uv run --frozen mkdocs build --strict

docs-serve: ## Serve the documentation locally with live reload
	uv run --frozen mkdocs serve

precommit: ## Run all pre-commit hooks against every file
	uv run --frozen pre-commit run --all-files

clean: ## Remove build, cache and coverage artifacts
	rm -rf site/ dist/ build/ htmlcov/ .coverage coverage.xml \
	       .pytest_cache/ .ruff_cache/ .pyrefly_cache/
