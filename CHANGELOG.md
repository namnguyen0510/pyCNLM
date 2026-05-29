# Changelog

All notable changes to **pycnlm** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Type-hints type-marker file (`py.typed`) so downstream packages benefit
  from inline type information without an external stub package.
- `pycnlm` console-script entry point with `solve-sat` / `solve-maxsat`
  / `reduce` / `version` sub-commands.

### Changed
- _N/A_

### Deprecated
- _N/A_

### Fixed
- _N/A_

### Removed
- _N/A_

---

## [0.1.0] — 2026-05-28

First public release.

### Added
- **CNLM-Langevin solver** (`pycnlm.langevin`):
  - DIMACS CNF / WCNF parser handling old-WCNF and MSE-2022 `h` formats.
  - `CNLMLangevinSolver`, `SolverConfig`, and `SolveResult` dataclasses.
  - Folder-level entry point `solve_folder()` with process-level parallelism.
  - Eight visualization functions (`plot_assignment_trajectory`,
    `plot_energy_curve`, `plot_clause_satisfaction`, …) and a one-shot
    `save_all_plots()` dashboard.
- **HOBO quadratization library** (`pycnlm.hobo`):
  - `HOBO` sparse polynomial data structure.
  - Zero-auxiliary reductions: `DeducReduc`, `ELCReduction`, `SplitReduction`.
  - NTR (negative-term) reductions: `NTR_KZFD`, `NTR_ABCG`, `NTR_ABCG2`,
    `NTR_GBP`.
  - PTR (positive-term) reductions: `PTR_BG`, `PTR_Ishikawa`, `PTR_KZ`,
    `PTR_GBP`, `BitFlipping`.
  - Arbitrary-function reductions: `ReductionBySubstitution`,
    `FGBZ_Negative`, `FGBZ_Positive`, `PairwiseCovers`.
  - Fermat-Quotient quadratization: `FERQ`, `create_ferq_evaluator`.
- **AdaptCNLM** (`pycnlm.adapt`):
  - `SymmetryDetector` — variable-orbit discovery from structural signatures.
  - `OrbitBasedEncoder`, `CliqueBasedEncoder`, `ClusterBasedEncoder` for
    qubit-count reduction on annealer hardware.
  - Optional Chimera / Pegasus embedding via the `[dwave]` extra.
- **Shared utilities** (`pycnlm.utils`): `SATInstance`, `parse_cnf_file`.

### Project infrastructure
- PEP 621 `pyproject.toml`, setuptools build, dynamic version sourced from
  `pycnlm/_version.py`.
- Optional dependency groups: `dwave`, `neural`, `benchmark`, `docs`, `dev`,
  `all`, `complete`.
- MIT-licensed.
- Continuous integration via GitHub Actions: lint (ruff), type-check (mypy),
  unit tests on Python 3.9–3.12 across Linux / macOS / Windows.
- Documentation site built with MkDocs Material + mkdocstrings.

[Unreleased]: https://github.com/your-org/pycnlm/compare/v0.1.0...HEAD
[0.1.0]:      https://github.com/your-org/pycnlm/releases/tag/v0.1.0
