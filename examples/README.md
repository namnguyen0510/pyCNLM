# pycnlm — Tutorial Notebooks

A guided, hands-on tour of the three `pycnlm` components. Every notebook is **self-contained**
(sample instances live in [`data/`](./data)) and runs in seconds on a laptop CPU — no quantum
hardware or GPU required.

## Contents

| Notebook | Component | What you'll learn |
|----------|-----------|-------------------|
| [`00_overview.ipynb`](./00_overview.ipynb) | all | Install, the public API map, a 30-second taste of each component. |
| [`01_langevin_sat_maxsat.ipynb`](./01_langevin_sat_maxsat.ipynb) | `pycnlm.langevin` | Solve SAT & MaxSAT, tune the fast–slow SDE, plot convergence (schedules, free energy, satisfied-clause curves), sweep chain counts. |
| [`02_hobo_reducers.ipynb`](./02_hobo_reducers.ipynb) | `pycnlm.hobo` | Build & inspect HOBOs, the PTR-vs-NTR duality, full quadratization with spectrum verification, SAT→QUBO, a head-to-head reducer benchmark. |
| [`03_adaptcnlm_qubit_reduction.ipynb`](./03_adaptcnlm_qubit_reduction.ipynb) | `pycnlm.adapt` | Symmetry/orbit detection, the three encoders, qubit-compression comparison, bit-flip polishing, adaptive-confidence dynamics, optional D-Wave embedding. |

Start with `00_overview.ipynb`, then read the three deep dives in any order.

## Running them

Install the package plus the notebook extras, then launch Jupyter from this folder:

```bash
pip install "pycnlm[examples]"      # jupyter, ipykernel, matplotlib, tqdm
cd examples
jupyter notebook                    # or: jupyter lab
```

The notebooks are committed **with their outputs** so they render fully on GitHub without
running anything. To re-execute from scratch:

```bash
jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```

## Sample data

| File | Used by | Description |
|------|---------|-------------|
| `data/sample_3sat.cnf` | 00, 01 | Small satisfiable 3-SAT instance (6 vars, 8 clauses). |
| `data/sample_maxsat.wcnf` | 01 | Weighted partial MaxSAT (hard + soft clauses). |
| `data/sample_symmetric.cnf` | 00, 03 | 9-variable instance with structure (three *exactly-one* groups) so the encoders visibly compress qubits to 9 / 6 / 4. |

## Going further

For full-scale batch runs and benchmarks, see the drivers in
[`../pycnlm/scripts/`](../pycnlm/scripts):
`run_LangevinCNLM/`, `run_HOBOReducers/`, and `run_AdaptCNLM.py`.
