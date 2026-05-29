# pycnlm

<p align="center">
  <em>Confidence Neural Logic Machines Toolbox</em>
</p>

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

## Quick start

### Solve a SAT instance with the Langevin solver

```python
import pycnlm

result = pycnlm.solve_sat_file("examples/uf20-01.cnf")

print(f"SAT?           {result.is_sat}")
print(f"Best energy    {result.best_energy:.4f}")
print(f"Assignment     {result.best_assignment[:10]}…")
```

### Quadratize a degree-4 monomial

```python
from pycnlm import HOBO
from pycnlm.hobo import PTR_Ishikawa

# E(x) = x0·x1·x2·x3
poly = {frozenset({0, 1, 2, 3}): 1.0}
hobo = HOBO(poly)

reducer = PTR_Ishikawa()
result  = reducer.quadratize(hobo)

print(f"Original degree:    {hobo.degree}")
print(f"Quadratic degree:   {result.quadratic.degree}")
print(f"Auxiliaries added:  {result.quadratic.n_aux}")
```

### Reduce qubit count with symmetry detection

```python
from pycnlm import parse_cnf_file, SymmetryDetector, ClusterBasedEncoder

sat = parse_cnf_file("instance.cnf")
orbits = SymmetryDetector(sat).find_orbits()

encoder = ClusterBasedEncoder(sat, orbits)
print(f"Qubit reduction: {sat.num_vars} → {encoder.total_qubits}")
```

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

## Documentation

Full documentation, including an API reference and tutorials, lives at
**https://pycnlm.readthedocs.io**.

Within this repository:

- [`docs/`](docs/) — Markdown sources rendered with MkDocs Material.
- [`pycnlm/core/LangevinCNLM/examples/demo.ipynb`](pycnlm/core/LangevinCNLM/examples/demo.ipynb) — end-to-end notebook demo.
- [`pycnlm/core/LangevinCNLM/README.md`](pycnlm/core/LangevinCNLM/README.md) — Langevin solver paper-style write-up.

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

Discussion of the underlying mathematics — Langevin annealing schedules,
quadratization bounds, orbit detection heuristics — is best raised in the
[Discussions](https://github.com/your-org/pycnlm/discussions) tab.

## License

`pycnlm` is released under the [MIT License](LICENSE). The `third_party/`
directory vendors reference implementations of competing methods used by the
benchmark harness; each upstream license is retained in-place and applies to
its respective sub-directory.
