# Benchmark suite

Comparison benchmarks pitting `cnlm_langevin` against neural and classical
SAT / MaxSAT solvers from the literature.

```
benchmark/
├── adapters/                # one Adapter per baseline (unified interface)
│   ├── base.py              # BaseAdapter, SolveOutcome
│   ├── adapter_cnlm.py
│   ├── adapter_walksat.py
│   ├── adapter_random_restart.py
│   ├── adapter_satnet.py    # both SATNet-SDP-NumPy and SATNet-official
│   ├── adapter_neurosat.py
│   └── adapter_neural_others.py   # PDP, NSNet, QuerySAT, GMS, SGAT-MS, G4SATBench
├── driver.py                # discovery, run loop, aggregation, plots
└── README.md                # (this file)

benchmark_SAT.py             # CLI for SAT
benchmark_MaxSAT.py          # CLI for MaxSAT
third_party/                 # cloned baseline repos (see below)
```

## What is benchmarked

| Adapter                | Class                          | Problem types | Source / paper |
|------------------------|--------------------------------|---------------|----------------|
| `cnlm_langevin`        | `CNLMAdapter`                  | SAT, MaxSAT   | This package (CNLM-Langevin fast-slow SDE) |
| `walksat`              | `WalkSATAdapter`               | SAT, MaxSAT   | Selman, Kautz, Cohen 1994 (pure-Python SLS) |
| `random_restart_greedy`| `RandomRestartGreedyAdapter`   | SAT, MaxSAT   | trivial reference baseline |
| `satnet_sdp_numpy`     | `SATNetSDPAdapter`             | SAT, MaxSAT   | Wang, Donti, Wilder, Kolter ICML 2019 — pure NumPy port of the SDP "Mixing Method", **no training required** |
| `satnet_official`      | `SATNetOfficialAdapter`        | SAT, MaxSAT   | `pip install satnet` (CUDA C++) |
| `neurosat`             | `NeuroSATAdapter`              | SAT           | Selsam et al. ICLR 2019 — `third_party/neurosat` |
| `pdp_satyr`            | `PDPAdapter`                   | SAT           | Amizadeh, Matusevych, Weimer 2019 — `third_party/PDP-Solver` |
| `nsnet`                | `NSNetAdapter`                 | SAT           | Li & Si NeurIPS 2022 — `third_party/NSNet` |
| `querysat`             | `QuerySATAdapter`              | SAT           | Ozolins et al. IJCNN 2022 — `third_party/QuerySAT` |
| `gms`                  | `GMSAdapter`                   | MaxSAT        | Liu AAAI 2022 — `third_party/GMS` |
| `sgat_ms`              | `SGATMSAdapter`                | SAT, MaxSAT   | NeurIPS 2025 — `third_party/SGAT-MS` |
| `g4satbench`           | `G4SATBenchAdapter`            | SAT           | Li, Guo, Si TMLR 2024 — `third_party/G4SATBench` (umbrella for NeuroSAT, GGNN, GIN, GCN, GAT under unified API) |

The first four (`cnlm_langevin`, `walksat`, `random_restart_greedy`,
`satnet_sdp_numpy`) **always run** — no GPU, no checkpoints, no extra
deps beyond NumPy / SciPy / matplotlib.  The remaining eight need
PyTorch / TensorFlow installs and (mostly) trained checkpoints.

## Cloned third-party repos

Located under `third_party/`:

```
third_party/
├── neurosat/         (dselsam/neurosat,            TF1, ~5.4 MB)
├── SATNet/           (locuslab/SATNet,             PyTorch, ~0.9 MB)
├── PDP-Solver/       (microsoft/PDP-Solver,        PyTorch, ~0.6 MB)
├── G4SATBench/       (zhaoyu-li/G4SATBench,        PyTorch + PyG, ~2.7 MB)
├── NSNet/            (zhaoyu-li/NSNet,             PyTorch + PyG, ~15 MB)
├── QuerySAT/         (LUMII-Syslab/QuerySAT,       TF2, ~9 MB)
├── GMS/              (minghao-liu/GMS,             PyTorch, ~0.3 MB)
└── SGAT-MS/          (sotam2369/SGAT-MS,           PyTorch + PyG, ~54 MB)
```

These were cloned via `git clone --depth 1`.  Each retains its original
LICENSE and README.

## Quick start (CPU, no extra installs)

```bash
cd cnlm_langevin
python benchmark_SAT.py    /your/cnf_folder    out_sat    --timeout 60
python benchmark_MaxSAT.py /your/wcnf_folder   out_maxsat --timeout 60
```

The script auto-detects which adapters can run, prints a summary up-front,
and skips the rest with a clear reason.  Out of the box you'll get a
4-way comparison: `cnlm_langevin` vs `walksat` vs `random_restart_greedy`
vs `satnet_sdp_numpy`.

## Adding the neural baselines

For each baseline you need (a) the right deep-learning framework and
(b) a trained checkpoint.  Most repos do **not** ship checkpoints — you
have to train on a synthetic distribution that matches your test set.

### NeuroSAT
```bash
pip install tensorflow==1.15        # TF1 only
cd third_party/neurosat
bash scripts/toy_train.sh           # trains a small model on random 3-SAT
# afterwards point the adapter at the resulting model.ckpt:
python ../../benchmark_SAT.py /folder /out --neurosat-ckpt /path/to/ckpt
```

### NSNet
```bash
pip install torch torch_geometric
cd third_party/NSNet
bash scripts/sat_data.sh            # generates training data
bash scripts/sat_nsnet_3-sat.sh     # trains the SAT model
# point at the checkpoint:
python ../../benchmark_SAT.py /folder /out --nsnet-ckpt runs/sat_nsnet_3-sat_marginal/checkpoints/model_best.pt
```

### PDP-Solver  (a.k.a. SATYR)
```bash
pip install torch
cd third_party/PDP-Solver && python setup.py install
# PDP supports the *non-learned* Survey-Propagation mode out of the box —
# no checkpoint needed.  Just run:
python ../../benchmark_SAT.py /folder /out --solvers cnlm_langevin pdp_satyr
```

### QuerySAT
```bash
pip install tensorflow                        # TF2
cd third_party/QuerySAT
pip install -r requirements.txt
python main.py                                 # train (long; needs a Tesla T4+)
# afterwards:
python ../../benchmark_SAT.py /folder /out --querysat-ckpt /path/to/model_dir
```

### G4SATBench
```bash
pip install torch torch_geometric
cd third_party/G4SATBench
bash scripts/install.sh
# train e.g. a NeuroSAT-style model on the easy SR distribution:
python train_model.py satisfying_assignment ~/data/sr/train/ \
    --train_splits sat --valid_dir ~/data/sr/valid/ --valid_splits sat \
    --label satisfying_assignment --graph lcg --model neurosat --n_iterations 32 \
    --batch_size 128 --seed 123
python ../../benchmark_SAT.py /folder /out \
    --g4satbench-ckpt runs/.../checkpoints/model_best.pt \
    --g4satbench-model neurosat --g4satbench-graph lcg
```

### GMS (MaxSAT)
```bash
# Tesla V100 + PyTorch 1.5 recommended (per the GMS README).
cd third_party/GMS
./setup.sh && ./generate_raw_data.sh && ./generate_data.sh && ./train.sh
python ../../benchmark_MaxSAT.py /folder /out --gms-ckpt /path/to/model.pt
```

### SGAT-MS (MaxSAT, NeurIPS 2025)
```bash
pip install torch torch_geometric
cd third_party/SGAT-MS
# pretraining
python src/main.py --train --model-id 1
# then benchmark in SGAT mode:
python ../../benchmark_MaxSAT.py /folder /out --sgat-ckpt third_party/SGAT-MS/plots --sgat-model-id 1 --sgat-mode sgat
# or use the bundled SDP-mixing baseline (no training):
python ../../benchmark_MaxSAT.py /folder /out --sgat-mode mixing --solvers sgat_ms
```

### SATNet (official, CUDA)
```bash
pip install satnet                  # needs nvcc + CUDA toolkit
python ../../benchmark_SAT.py /folder /out --solvers satnet_official
```

## Output files

```
<out_dir>/
├── results_per_instance.csv       # one row per (adapter, instance) — raw numbers
├── summary_per_solver.json        # aggregated metrics per solver
├── 01_score_and_solved.pdf        # mean sat-score + fully-solved fraction (bar)
├── 02_runtime_box.pdf             # runtime distribution (log-scale boxplot)
├── 03_cost_box.pdf                # MaxSAT only — cost distribution
├── paper_table.pdf                # publication-style results table (rendered)
└── paper_table.tex                # the same table as LaTeX booktabs source
```

The **paper-style table** mimics the visual style of modern ML-paper benchmark
tables (booktabs rules, vertical column headers with citations, grouped row
labels for solver families, bold + underlined best-per-column entries,
blue-tinted "ours" row, gray-italic rows for unavailable solvers).  If you
arrange your DIMACS files in subfolders (e.g. `easy/`, `med/`, `hard/`) the
table automatically gets one column per group plus an Average column.

The `.tex` file is drop-in for any LaTeX paper — required preamble:

```latex
\usepackage{booktabs,multirow,array,colortbl,xcolor,graphicx}
\definecolor{ourblue}{HTML}{DCE8F4}
\definecolor{citegray}{HTML}{777777}
```

The leaderboard printed at the end ranks solvers by

  1. number of fully-SAT (or fully-hard-SAT for MaxSAT) instances
  2. mean sat-score
  3. mean runtime (lower is better)

For MaxSAT the leaderboard also reports mean cost (lower is better).

## Adding a new baseline

1. Drop the repo under `third_party/<name>/`.
2. Create `benchmark/adapters/adapter_<name>.py` implementing the
   `BaseAdapter` interface (see `adapters/base.py`).
3. Register the class in `benchmark/adapters/__init__.py:ALL_ADAPTERS`.
4. Run `python benchmark_SAT.py /folder /out --solvers <name>` to verify.

## Caveats

* Comparing learned and non-learned solvers is inherently lossy — neural
  solvers are usually trained on a specific synthetic distribution and may
  generalise poorly off-distribution, while non-learned solvers like
  WalkSAT or our CNLM-Langevin are distribution-free.
* For SAT, scoring is `n_satisfied / n_clauses` and "full SAT" is a binary
  pass/fail metric.  For MaxSAT, the primary metric is **cost** (sum of
  unsatisfied soft-clause weights), and "full hard-SAT" indicates whether
  every hard clause is satisfied.
* Runtimes are wall-clock and include all setup / forward-pass overhead.
  This favours light-weight solvers like WalkSAT for tiny instances.
