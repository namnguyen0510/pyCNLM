# =============================================================================
#  pycnlm — Makefile
#  ----------------------------------------------------------------------------
#  Self-documenting: run ``make`` (or ``make help``) for the full list.
# =============================================================================
SHELL := /bin/bash

# Treat all targets as phony unless they correspond to real files.
.PHONY: help install dev test test-fast test-slow lint format typecheck \
        coverage docs docs-serve build clean clean-all release-check \
        precommit-install precommit-run

PYTHON ?= python
PIP    ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
RUFF   ?= $(PYTHON) -m ruff
MYPY   ?= $(PYTHON) -m mypy

# ----------------------------------------------------------------------------
#  Default target
# ----------------------------------------------------------------------------
help:  ## Show this help.
	@printf "\n\033[1mpycnlm — development tasks\033[0m\n\n"
	@grep -E '^[a-zA-Z_-]+:.*## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*## "} {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo

# ----------------------------------------------------------------------------
#  Installation
# ----------------------------------------------------------------------------
install:  ## pip install the package (runtime deps only).
	$(PIP) install -e .

dev:  ## Editable install with every dev/optional extra.
	$(PIP) install -e ".[dev,docs,benchmark]"
	@$(MAKE) precommit-install

precommit-install:  ## Install the git pre-commit hooks.
	$(PYTHON) -m pre_commit install --install-hooks || \
	    echo "  (pre-commit not installed — skip)"

# ----------------------------------------------------------------------------
#  Testing
# ----------------------------------------------------------------------------
test: test-fast  ## Run the full default test suite (alias for test-fast).

test-fast:  ## Run tests, skipping anything marked 'slow'.
	$(PYTEST) -m "not slow" -q

test-slow:  ## Run only the slow / integration tests.
	$(PYTEST) -m "slow" -q

coverage:  ## Run tests and produce a coverage report.
	$(PYTEST) -m "not slow" --cov=pycnlm --cov-report=term-missing --cov-report=html
	@echo "HTML report: htmlcov/index.html"

# ----------------------------------------------------------------------------
#  Lint / format / typecheck
# ----------------------------------------------------------------------------
lint:  ## Run ruff lint (no auto-fix).
	$(RUFF) check pycnlm tests

format:  ## Auto-format code and apply safe lint fixes.
	$(RUFF) format pycnlm tests
	$(RUFF) check --fix pycnlm tests

typecheck:  ## Run mypy over the package.
	$(MYPY) pycnlm

precommit-run:  ## Run every pre-commit hook against every file.
	$(PYTHON) -m pre_commit run --all-files

# ----------------------------------------------------------------------------
#  Build / docs / release
# ----------------------------------------------------------------------------
build: clean  ## Build the wheel and source distribution.
	$(PYTHON) -m build

docs:  ## Build the documentation site to ./site/.
	$(PYTHON) -m mkdocs build --strict

docs-serve:  ## Serve the docs locally on http://127.0.0.1:8000/.
	$(PYTHON) -m mkdocs serve

release-check: build  ## Run twine's metadata sanity check.
	$(PYTHON) -m twine check dist/*

# ----------------------------------------------------------------------------
#  House-keeping
# ----------------------------------------------------------------------------
clean:  ## Remove build, dist, cache, and coverage artefacts.
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache \
	       htmlcov .coverage coverage.xml site
	find . -type d -name __pycache__   -exec rm -rf {} +  2>/dev/null || true
	find . -type f -name '*.pyc'       -delete 2>/dev/null || true

clean-all: clean  ## Like clean, plus venvs and pre-commit caches.
	rm -rf .venv venv env .tox
	$(PYTHON) -m pre_commit clean 2>/dev/null || true
