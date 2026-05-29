# pyCNLM: Confidence Neural Logic Machines solvers

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</p>

---

`pycnlm` is a unified Python toolkit for **Continuous Non-Linear Manifold**
(CNLM) methods applied to combinatorial optimization. It bundles three
complementary components:

| Component | What it does | Public name |
|-----------|--------------|-------------|
| **CNLM-Langevin** | A fast–slow SDE solver for SAT / MaxSAT in DIMACS `.cnf` / `.wcnf` format, with parallel chains and a logarithmic annealing schedule. | `pycnlm.langevin` |
| **HOBO Reducers** | A library of quadratization methods (NTR, PTR, SFR, FERQ, ELC, FGBZ, …) for compiling polynomial pseudo-Boolean objectives down to QUBO. | `pycnlm.hobo` |
| **AdaptCNLM** | Symmetry-based qubit-count reduction and embedding onto Chimera / Pegasus annealer topologies. | `pycnlm.adapt` |

## Installation

```bash
# Core install (numpy, matplotlib, networkx only)
pip install pycnlm

# With D-Wave embedding support
pip install "pycnlm[dwave]"

# With neural baselines (PyTorch)
pip install "pycnlm[neural]"

# With benchmarking adapters (PySAT, scikit-learn, …)
pip install "pycnlm[benchmark]"

# Everything except the heavy GPU stack
pip install "pycnlm[all]"

# Truly everything
pip install "pycnlm[complete]"
```

Python 3.9 or newer is required.


### Command-line interface

The `pycnlm` console script exposes the solver to the shell:

```bash
# Single instance
pycnlm solve-sat path/to/instance.cnf --chains 32 --steps 3000

# A whole folder, 8 worker processes
pycnlm solve-sat path/to/cnf_folder/  --workers 8  --out results/

# MaxSAT
pycnlm solve-maxsat path/to/instance.wcnf --slow-sde
```

Run `pycnlm --help` for the full list of sub-commands.


## Project layout

```
pycnlm/
├── pyproject.toml
├── LICENSE
├── README.md
├── CHANGELOG.md
├── docs/                          ← MkDocs documentation
├── tests/                         ← pytest suite
└── pycnlm/                        ← the Python package
    ├── __init__.py                ← curated public API
    ├── _version.py
    ├── cli.py                     ← `pycnlm` console script
    ├── py.typed
    ├── core/
    │   ├── AdaptCNLM/             ← symmetry-based encoders
    │   ├── HOBOReducers/          ← quadratization library
    │   └── LangevinCNLM/
    │       ├── cnlm_langevin/     ← the solver itself
    │       ├── benchmark/         ← benchmarking harness + adapters
    │       └── third_party/       ← vendored baselines (not pip-installed)
    ├── scripts/                   ← legacy run scripts (not pip-installed)
    └── utils/                     ← shared helpers (dataloader, …)
```

## Citation

If `pycnlm` contributes to academic work, please cite it via the
[`CITATION.cff`](CITATION.cff) file at the repo root, or copy the BibTeX
entry rendered on the GitHub sidebar.

## Contributing

Bug reports, feature requests, and pull requests are warmly welcome.
See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development workflow,
coding standards, and the contributor licence terms.


## License

`pycnlm` is released under the [MIT License](LICENSE). The `third_party/`
directory vendors reference implementations of competing methods used by the
benchmark harness; each upstream license is retained in-place and applies to
its respective sub-directory.
