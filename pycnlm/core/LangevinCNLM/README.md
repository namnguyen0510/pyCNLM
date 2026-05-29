# `cnlm_langevin` — CNLM-Langevin (fast-slow) for SAT and MaxSAT

python aggregate_results.py --results_dir _RESULTS_ --out_dir ./output

A precise, parallelised Python implementation of the **Continuous Non-Linear Manifold
Langevin** solver in its **fast-slow** SDE form, applied to SAT and MaxSAT instances
in DIMACS `.cnf` / `.wcnf` format.

The solver discretises the coupled Itô SDE

```
fast :  dz_t  = -∇_z F̃_λ(z_t; c_t) dt + sqrt(2/β_t)   dW^z_t
slow :  dρ_t  = -η ∇_ρ F̃_λ(z_t; e^{ρ_t}) dt + sqrt(2η/β_c) dW^ρ_t      (c_t = e^{ρ_t})
```

with the lifted free energy

```
F̃_λ(z; c) = -Σ_j w_j ln(1 + exp(c_j s̃_j(z))) + (λ/2)‖z‖²
```

over the continuous embedding `x = σ(z) ∈ (0,1)ⁿ`. For a CNF clause `C_j` with signed
literal row `L_j ∈ {-1,0,+1}^n`,

```
s̃_j(z) = Σ_i L_{j,i} σ(z_i) + (|N_j| - 1 + ε)        with ε ∈ (0,1)
```

so that `s̃_j(z) > 0` ⇔ clause `C_j` is satisfied at `x = σ(z)`.

The temperature `β_t` follows a logarithmic schedule (sufficient for global
convergence by the cooling theorem), and `c_t` follows a polynomial / linear growth
that sharpens the soft-plus indicator into the true clause indicator at
`t → T`.

---

## Package layout

```
cnlm_langevin/
├── pyproject.toml
├── run_SAT.py                 # CLI driver for a folder of .cnf
├── run_MaxSAT.py              # CLI driver for a folder of .wcnf
├── cnlm_langevin/
│   ├── __init__.py
│   └── core/
│       ├── parser.py          # DIMACS CNF + WCNF (old + MSE-2022 'h' formats)
│       ├── instance.py        # SATInstance / MaxSATInstance + signed L matrix
│       ├── dynamics.py        # CNLMLangevinSolver, SolverConfig, SolveResult
│       ├── solver.py          # solve_sat_file / solve_maxsat_file / solve_folder
│       └── viz.py             # 8 plot functions + summary dashboard
├── examples/
│   └── demo.ipynb             # end-to-end demo on one SAT and one MaxSAT instance
└── tests/
    └── smoke_test.py          # 8 self-contained correctness tests
```

## Install

```bash
pip install numpy scipy matplotlib
# package itself is import-from-source:
cd cnlm_langevin
python -c "import cnlm_langevin; print(cnlm_langevin.__version__)"
```

## CLI usage

### SAT

```bash
python run_SAT.py /path/to/cnf_folder results_sat \
    --workers 8 \
    --steps 3000 --chains 32 \
    --slow-sde
```

### MaxSAT

```bash
python run_MaxSAT.py /path/to/wcnf_folder results_maxsat \
    --workers 8 \
    --steps 4000 --chains 32 \
    --hard-scale 1000
```

Both scripts forward through `cnlm.solve_folder`, which dispatches one instance per
worker process via `concurrent.futures.ProcessPoolExecutor`. Within each instance
the solver runs `n_chains` Langevin walkers in vectorised parallel via NumPy/BLAS.

## Per-instance outputs

Each input file `foo.cnf` produces a directory `out/foo/` containing:

| file                    | contents                                                          |
|-------------------------|-------------------------------------------------------------------|
| `summary.json`          | metrics: `is_SAT`, `n_satisfied/n_clauses`, `sat_score`, `cost` (MaxSAT), `runtime`, `converged_step`, configuration snapshot |
| `instance_meta.json`    | `n_vars`, `n_clauses`, `n_hard`, `n_soft`, mean/median clause width, `top` (MaxSAT) |
| `solution.txt`          | DIMACS-style assignment (`v 1 -2 3 …`) plus `o <cost>` for MaxSAT |
| `result_full.npz`       | compressed: schedule histories, per-chain free-energy / n-sat traces, final assignments of every chain, `sat_mask` |
| `00_summary.pdf`        | one-page solution dashboard (assignment + chains + clauses + schedule) |
| `01_energy.pdf`         | free-energy F̃ vs step for every chain                            |
| `02_clauses.pdf`        | best-chain n_satisfied curve and ensemble cloud                   |
| `03_confidence.pdf`     | mean and per-variable `|σ(z) − ½|` (annealing certainty)          |
| `04_schedule.pdf`       | β(t) and c(t) actual trajectories                                 |
| `05_chain_diversity.pdf`| pairwise Hamming distance heatmap of final assignments            |
| `06_score_distribution.pdf` | per-clause SAT/UNSAT broken out by hardness and width         |
| `07_maxsat.pdf`         | (MaxSAT only) hard-vs-soft satisfied weight bar charts            |
| `08_assignment_traj.pdf`| (if `record_assignment_every>0`) `x_i(t)` for every variable      |
| `09_clause_veritron_heatmap.pdf` | (if `record_assignment_every>0`) per-clause $\nu_j(t)=\sigma(c_t \tilde s_j)$ heatmap + aggregate dynamics |
| `10_convergence_to_corners.pdf` | (if `record_assignment_every>0`) $x_i(t)$ vs Boolean corners; SAT colored, UNSAT grey, $c(t)$ on twin axis |
| `11_sde_trajectory_2d.pdf` | (if `record_assignment_every>0`) hexbin density of chains in $(z_i, z_j)$ across annealing epochs |
| `12_loss_landscape_sweep.pdf` | (small instances only) sweep of $c$ on the lifted free-energy slice with $-\nabla\widetilde F$ streamlines |

### Top-level outputs (per folder run)

| file                | contents                                                                |
|---------------------|-------------------------------------------------------------------------|
| `summary.csv`       | one row per instance: name, is_SAT, sat_score, cost, runtime, …         |
| `all_results.json`  | aggregate metrics: fraction solved, mean cost, hard-sat fraction, mean runtime |
| `errors.json`       | (only if any worker raised) per-file traceback                          |

## MaxSAT metrics reported

The solver reports the standard partial-MaxSAT and weighted-MaxSAT competition metrics:

* **`is_SAT`** — all hard clauses satisfied
* **`n_hard_sat / n_hard_total`** — hard-clause satisfaction
* **`n_soft_sat / n_soft_total`** — soft-clause count
* **`soft_weight_satisfied`** — `Σ_{j soft, sat} w_j`
* **`cost`** — `Σ_{j soft, unsat} w_j` (MaxSAT objective; lower is better)
* **`sat_score`** — overall `n_satisfied / n_clauses`

## Paper-style figures (v1.1)

Five additional visualizations / analyses from the paper are exposed as top-level functions:

| function | what it shows |
|---|---|
| `cnlm.plot_clause_veritron_heatmap(result, instance)` | Per-clause $\nu_j(t)=\sigma(c_t\,\tilde s_j(z_t))$ heatmap along the trajectory of one chain, with mean-$\bar\nu(t)$ + per-clause traces + a green band marking timesteps where every clause is satisfied. |
| `cnlm.plot_loss_landscape_sweep(instance, c_values=...)` | Sweeps $c$ over several values and renders $\widetilde F_\lambda(z;c)$ on a 2-D slice with $-\nabla\widetilde F$ streamlines. Shows the landscape morphing from convex (small $c$) to a sharp V-basin (large $c$). For $n>2$ the other variables are anchored at $0.5$. |
| `cnlm.plot_sde_trajectory_2d(result, instance, project=(i,j))` | Hexbin density of the chain ensemble in $(z_i, z_j)$-space across early / mid / late annealing epochs. Visualizes the stochastic concentration onto the satisfying corner as $\beta_t,\,c_t$ grow. |
| `cnlm.plot_convergence_to_corners(result, instance)` | Two-panel "annealed Langevin" trajectory plot: $x_i(t)$ on the left, $\|x(t)-x^*\|_2$ to every Boolean corner on the right (SAT colored, UNSAT grey, $c(t)$ on twin axis). For $n>12$ the right panel falls back to distance-to-best. |
| `cnlm.analyze_gradient_snr(instance, n_samples=2000)` + `cnlm.plot_gradient_snr(...)` | **Pre-solve diagnostic.** Computes $\mathrm{SNR}_j(c)=\|\mathbb E_x[\partial_c F_j]\|/\sigma_x[\partial_c F_j]$ over uniform-random $x$, sweeping $c$ on a log-grid. Useful for choosing a $c$-schedule. |

These all participate in `save_all_plots`, so per-instance directories get them automatically when the trajectory was recorded (`SolverConfig(record_assignment_every>0)`).

## Demo notebook

1. Building / parsing a satisfiable random 3-SAT in DIMACS
2. Running the full **fast-slow** SDE (slow-SDE on `ρ` enabled) with 16 parallel chains
3. Pretty-printed `SolveResult` summary as an HTML table
4. Independent verification of the assignment against every clause
5. All eight diagnostic plots inline
6. Building a partial MaxSAT (`.wcnf`, MSE-2022 `h` format) with conflicting soft clauses
7. Brute-force verification that the solver hits (or comes close to) the true optimum
8. Folder-level multi-process driver demo with `solve_folder`

## Method internals — what is `n_neg − 1 + ε`?

For a CNF clause we encode the row `L_j ∈ {−1,0,+1}^n` so that `+1` ↔ positive
literal, `−1` ↔ negated literal, `0` ↔ variable absent. Define `|N_j|` =
number of negated literals. Then for the embedding `x = σ(z) ∈ (0,1)ⁿ`,

```
Σ_i L_{j,i} σ(z_i) + (|N_j| − 1 + ε)
   = Σ_{i ∈ P_j} σ(z_i) + Σ_{i ∈ N_j} (1 − σ(z_i)) − 1 + ε
   = (#satisfied literals as a soft count) − 1 + ε.
```

When the clause is fully Boolean (all `σ(z_i) ∈ {0,1}`), this is `≥ ε > 0`
iff at least one literal is satisfied, and `≤ ε − 1 < 0` if none is. So
`s̃_j > 0` is exactly the disjunction. Lifting via the soft-plus
`ln(1 + e^{c_j s̃_j})` produces a smooth surrogate of the clause indicator
that becomes sharp as `c_j → ∞`.

## Reproducing a single solve in code

```python
import cnlm_langevin as cnlm

parsed = cnlm.parse_dimacs_cnf("uf20-01.cnf")
inst   = cnlm.SATInstance.from_parsed(parsed)

cfg = cnlm.SolverConfig(
    n_steps=2000, n_chains=32, seed=0,
    use_slow_sde=True, eta=0.05, beta_c=50.0,
    beta_init=1.0, beta_final=80.0, beta_schedule="log",
    c_init=1.0, c_final=80.0, c_schedule="lin",
    early_stop_when_sat=True,
    record_assignment_every=10,
)
res = cnlm.CNLMLangevinSolver(inst, cfg).solve()
print(res.is_SAT, f"{res.n_satisfied}/{res.n_clauses}")
```

## Tests

```bash
python tests/smoke_test.py
```

runs nine self-contained tests covering the parser, the literal matrix, easy SAT,
random uf20-style 3-SAT, old-format WCNF, the slow-SDE branch, the multi-process
folder driver, a trivially UNSAT instance, and the v1.1 paper-style plots.

## Benchmark suite (v1.2)

The package ships a benchmark harness pitting CNLM-Langevin against neural
and classical SAT/MaxSAT solvers from the literature.  See
[`benchmark/README.md`](benchmark/README.md) for full details.

```bash
python benchmark_SAT.py    /your/cnf_folder    out_sat    --timeout 60
python benchmark_MaxSAT.py /your/wcnf_folder   out_maxsat --timeout 60
```

**Always-available adapters** (no GPU, no extra deps, no checkpoints):
`cnlm_langevin`, `walksat`, `random_restart_greedy`, `satnet_sdp_numpy`
(NumPy port of SATNet's mixing-method SDP — Wang et al., ICML 2019).

**Neural baselines** (cloned under `third_party/`, need their own framework
+ trained checkpoints — adapters report unavailability cleanly if missing):
`neurosat`, `pdp_satyr`, `nsnet`, `querysat`, `g4satbench`, `gms`, `sgat_ms`,
`satnet_official`.

The harness writes per-instance CSV, per-solver summary JSON, three
comparison plots (sat-score, runtime, MaxSAT cost), a **publication-style
results table** (`paper_table.pdf` + `paper_table.tex` — booktabs LaTeX
ready to drop into a paper, with vertical column headers, grouped row
families, bold-underlined best entries, and a blue-tinted "ours" row),
and prints a leaderboard ranking solvers by full-solved count, mean
sat-score, and mean runtime.
