# Contributing to pycnlm

Thanks for thinking about contributing! This document is the short version
of everything you need to make a clean, mergeable change.

> **TL;DR**
> 1. Fork & clone.
> 2. `make dev` — installs the package editable, with all extras and the
>    pre-commit hooks.
> 3. Make your change on a feature branch.
> 4. `make test` and `make lint` are green.
> 5. Open a PR against `main` and fill in the template.

---

## Repository layout

```
pycnlm/                ← the Python package (see README.md for details)
docs/                  ← MkDocs documentation sources
tests/                 ← pytest suite + small DIMACS fixtures
pyproject.toml         ← single source of truth for build / lint / type config
Makefile               ← shortcuts for the common dev tasks
```

## Development setup

```bash
git clone https://github.com/your-org/pycnlm.git
cd pycnlm
python -m venv .venv
source .venv/bin/activate      # or `.\.venv\Scripts\Activate.ps1` on Windows
make dev                       # editable install + dev/docs/benchmark extras
```

`make dev` also installs the pre-commit hooks; from then on, `git commit`
runs `ruff`, `mypy`, and basic hygiene checks automatically.

## Running things

| Goal                              | Command                       |
|-----------------------------------|-------------------------------|
| Run the fast test suite           | `make test`                   |
| Run the slow / integration tests  | `make test-slow`              |
| Coverage report                   | `make coverage`               |
| Lint (no auto-fix)                | `make lint`                   |
| Format + safe auto-fixes          | `make format`                 |
| Type-check                        | `make typecheck`              |
| Build wheel + sdist               | `make build`                  |
| Build docs locally                | `make docs-serve`             |

## Coding conventions

- **Line length:** 100 columns (enforced by `ruff format`).
- **Style:** Black-compatible via `ruff format`. Imports sorted by `ruff`.
- **Type hints:** new code should be typed where reasonable. The existing
  algorithmic surface is gradually being typed.
- **Public API:** anything exported from `pycnlm.__init__` is part of the
  stable surface. Don't add re-exports there without a corresponding
  entry in the docs.
- **Docstrings:** use NumPy or reST style consistently within a file.
- **Tests:** new code should ship with tests. We aim for unit-test coverage
  on the public surface and at least one end-to-end test per top-level
  feature.

## Algorithmic / "core method" changes

The repository ships several research-grade implementations whose
behaviour is part of the contract (e.g. solver dynamics, quadratization
reductions). When you change algorithmic code:

1. Call it out in the PR description — what changed and why.
2. Include a benchmark comparing before/after on a representative
   instance set, or a citation to the paper you're following.
3. Bump at least the MINOR version per SemVer (or MAJOR if it's
   user-visible behaviour change).

Non-algorithmic refactors (renaming a private helper, fixing imports,
docstring polish) need no special treatment — just keep tests green.

## Commit messages

Conventional Commits are preferred but not enforced. Roughly:

```
<type>(<optional scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `ci`,
`build`, `style`. The PR title is what ends up in the release notes —
write it for a reader who doesn't know the rest of the PR.

## Branching & releases

- Feature branches off `main`; PR back into `main`.
- Releases are tagged `vMAJOR.MINOR.PATCH`. Pushing such a tag triggers
  the release workflow (`.github/workflows/release.yml`) which builds,
  validates, and publishes to PyPI via OIDC trusted publishing.
- The package version lives in `pycnlm/_version.py` — the release
  workflow refuses to publish if the tag and the version disagree.

## Reporting bugs and asking questions

- **Bugs:** open an issue using the *Bug report* template; please include
  the output of `pycnlm info` and a minimal repro.
- **Questions / design discussions:** use the
  [Discussions](https://github.com/your-org/pycnlm/discussions) tab.
- **Security vulnerabilities:** see [`SECURITY.md`](SECURITY.md).

## License of contributions

By submitting a contribution you agree that it is released under the
project's [MIT License](LICENSE).

Thanks again — see you in the PR queue!
