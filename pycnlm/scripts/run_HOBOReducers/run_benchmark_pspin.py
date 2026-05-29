#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
HIGH-ORDER BENCHMARK: Random p-Spin Glass Model
═══════════════════════════════════════════════════════════════════════════
GOAL: Benchmark reducers on naturally high-order problems (Degree p >= 3).
PROBLEM: Minimize E(x) = Sum_{hyperedges} J_e * Prod_{i in e} x_i
         Where interactions are naturally of degree p.
CONFIG: Nodes = [10, 12, 14], Degrees (p) = [3, 4, 5], Runs = 10.
OUTPUT: Generated samples, JSON results, CSV summaries, Plots.
═══════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import csv
import time
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable
from itertools import combinations, product
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from math import comb

# Plotting
try:
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    print("⚠ Matplotlib not found. Plots will be skipped.")

# Progress Bar — prefer standard tqdm in script context; notebook tqdm causes
# garbled output when running from the command line.
try:
    from tqdm import tqdm
except ImportError:
    # Graceful no-op fallback if tqdm is not installed at all.
    def tqdm(iterable=None, *args, **kwargs):  # type: ignore[misc]
        return iterable if iterable is not None else range(0)

# Ensure reducers module is accessible
try:
    _current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _current_dir = os.getcwd()

sys.path.insert(0, os.path.dirname(_current_dir))

try:
    from pycnlm.core.HOBOReducers import (
        HOBO, QuadResult,
        DeducReduc, ELCReduction,
        NTR_KZFD, NTR_ABCG, NTR_ABCG2, NTR_GBP,
        PTR_Ishikawa, PTR_KZ, PTR_GBP, BitFlipping,
        ReductionBySubstitution, FGBZ_Negative, FGBZ_Positive, PairwiseCovers,
        FERQ,
    )
    _REDUCERS_AVAILABLE = True
except ImportError as e:
    _REDUCERS_AVAILABLE = False
    print(f"CRITICAL ERROR: Could not import reducers module. Error: {e}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# JSON HELPERS
# ═══════════════════════════════════════════════════════════════════════════
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)

def convert_numpy_types(obj):
    if isinstance(obj, dict):       return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):       return [convert_numpy_types(v) for v in obj]
    if isinstance(obj, np.bool_):   return bool(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating):return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class PSpinInstance:
    n_nodes: int
    degree: int  # The 'p' in p-spin
    interactions: List[Tuple[Tuple[int, ...], float]]  # ((i, j, k...), weight)
    source_id: str

    def to_hobo(self) -> HOBO:
        """Directly map interactions to HOBO terms.  E(x) = Sum J_e * Prod x_i"""
        terms: Dict = {}
        for indices, weight in self.interactions:
            key = frozenset(indices)
            terms[key] = terms.get(key, 0.0) + weight
        terms = {k: v for k, v in terms.items() if abs(v) > 1e-9}
        return HOBO(terms, n_vars=self.n_nodes)

    def compute_energy(self, assignment: List[int]) -> float:
        if len(assignment) < self.n_nodes:
            assignment = assignment + [0] * (self.n_nodes - len(assignment))
        energy = 0.0
        for indices, weight in self.interactions:
            prod = 1.0
            for idx in indices:
                prod *= assignment[idx]
            energy += weight * prod
        return energy

    # FIX #1 — Brute-force is only feasible up to N ≈ 20.  Guard against
    #           accidentally calling it on larger instances.
    BRUTEFORCE_MAX_N: int = field(default=20, init=False, repr=False)

    def get_optimal_bruteforce(self) -> Tuple[float, List[int]]:
        if self.n_nodes > 20:
            raise ValueError(
                f"Brute-force search is infeasible for n_nodes={self.n_nodes} "
                f"(2^{self.n_nodes} ≈ {2**self.n_nodes:,} states). "
                "Reduce n_nodes to ≤ 20, or replace with a heuristic ground-state solver."
            )
        min_val = float('inf')
        best_assign: List[int] = []
        for assign in product([0, 1], repeat=self.n_nodes):
            val = self.compute_energy(list(assign))
            if val < min_val:
                min_val = val
                best_assign = list(assign)
        return min_val, best_assign


@dataclass
class OptimizationRunResult:
    run_id: int
    method_name: str
    n_vars_original: int
    n_vars_quadratized: int
    n_auxiliary: int
    energy: float
    optimal_energy: float
    relative_error: float
    is_optimal: bool
    runtime_ms: float
    assignment: List[int]

    def to_dict(self) -> Dict:
        return {
            'run_id':             int(self.run_id),
            'method_name':        str(self.method_name),
            'n_vars_original':    int(self.n_vars_original),
            'n_vars_quadratized': int(self.n_vars_quadratized),
            'n_auxiliary':        int(self.n_auxiliary),
            'energy':             float(self.energy),
            'optimal_energy':     float(self.optimal_energy),
            'relative_error':     float(self.relative_error),
            'is_optimal':         bool(self.is_optimal),
            'runtime_ms':         float(self.runtime_ms),
            'assignment':         [int(x) for x in self.assignment],
        }


@dataclass
class InstanceBenchmarkResults:
    instance_id: str
    n_nodes: int
    degree: int
    n_interactions: int
    optimal_energy: float
    optimal_assignment: List[int]
    graph_data: Dict
    runs: List[OptimizationRunResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'instance_id':        self.instance_id,
            'n_nodes':            self.n_nodes,
            'degree':             self.degree,
            'n_interactions':     self.n_interactions,
            'optimal_energy':     float(self.optimal_energy),
            'optimal_assignment': [int(x) for x in self.optimal_assignment],
            'graph_data':         self.graph_data,
            'runs':               [r.to_dict() for r in self.runs],
            'metadata':           convert_numpy_types(self.metadata),
        }


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZER (Unified SA)
# ═══════════════════════════════════════════════════════════════════════════
class UnifiedSimulatedAnnealing:
    def __init__(self, num_reads: int = 20, num_sweeps: int = 2000,
                 beta_range: Tuple[float, float] = (0.1, 10.0), seed: Optional[int] = None):
        self.num_reads  = num_reads
        self.num_sweeps = num_sweeps
        self.beta_range = beta_range
        self.seed       = seed
        self.rng        = np.random.default_rng(seed)

    def optimize(
        self,
        energy_fn: Callable,
        n_vars: int,
        delta_fn: Optional[Callable] = None,
    ) -> Tuple[List[int], float, float]:
        start_time     = time.perf_counter()
        beta_min, beta_max = self.beta_range
        T_init = 1.0 / beta_min
        T_min  = 1.0 / beta_max
        decay  = (T_min / T_init) ** (1.0 / max(self.num_sweeps - 1, 1))

        best_energy = float('inf')
        best_assignment: Optional[List[int]] = None

        if delta_fn is not None:
            # Fast path: incremental delta evaluation (e.g. FERQ).
            for _ in range(self.num_reads):
                x = self.rng.integers(0, 2, size=n_vars, dtype=np.int32)
                energy = float(energy_fn(x))
                local_best_energy = energy
                local_best_x      = x.copy()
                T = T_init

                for _ in range(self.num_sweeps):
                    i     = int(self.rng.integers(n_vars))
                    delta = float(delta_fn(x, i))
                    if delta < 0.0 or self.rng.random() < np.exp(-delta / max(T, 1e-12)):
                        x[i] ^= 1
                        energy += delta
                        if energy < local_best_energy:
                            local_best_energy = energy
                            local_best_x      = x.copy()
                    T *= decay

                if local_best_energy < best_energy:
                    best_energy     = local_best_energy
                    best_assignment = local_best_x.tolist()
        else:
            # Standard path: full re-evaluation per flip.
            for _ in range(self.num_reads):
                x = self.rng.integers(0, 2, size=n_vars).tolist()
                energy            = energy_fn(x)
                local_best_energy = energy
                local_best_x      = x.copy()
                T = T_init

                for _ in range(self.num_sweeps):
                    i          = int(self.rng.integers(n_vars))
                    x[i]       = 1 - x[i]
                    new_energy = energy_fn(x)
                    delta      = new_energy - energy
                    if delta < 0 or self.rng.random() < np.exp(-delta / max(T, 1e-12)):
                        energy = new_energy
                        if energy < local_best_energy:
                            local_best_energy = energy
                            local_best_x      = x.copy()
                    else:
                        x[i] = 1 - x[i]  # revert
                    T *= decay

                if local_best_energy < best_energy:
                    best_energy     = local_best_energy
                    best_assignment = local_best_x

        runtime_ms = (time.perf_counter() - start_time) * 1000
        return best_assignment or [0] * n_vars, float(best_energy), float(runtime_ms)


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════════════════
class PSpinBenchmarkRunner:
    # FIX #2 — Use a factory method instead of class-level instantiation to
    #           avoid sharing mutable reducer state across runner instances.
    @staticmethod
    def _build_methods() -> List[Tuple[str, Any]]:
        return [
            ("DeducReduc",              DeducReduc()),
            ("NTR_KZFD",                NTR_KZFD()),
            ("NTR_ABCG",                NTR_ABCG()),
            ("NTR_ABCG2",               NTR_ABCG2()),
            ("NTR_GBP",                 NTR_GBP()),
            ("PTR_Ishikawa",            PTR_Ishikawa()),
            ("PTR_KZ",                  PTR_KZ()),
            ("PTR_GBP",                 PTR_GBP()),
            ("BitFlipping",             BitFlipping()),
            ("ReductionBySubstitution", ReductionBySubstitution()),
            ("FGBZ_Negative",           FGBZ_Negative()),
            ("FGBZ_Positive",           FGBZ_Positive()),
            ("PairwiseCovers",          PairwiseCovers()),
            ("FERQ",                    FERQ(max_degree=15)),
        ]

    def __init__(self, output_dir: str = "pspin_benchmark", num_runs: int = 10):
        self.output_dir = Path(output_dir)
        for sub in ["instances", "results", "assignments", "csv", "plots"]:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

        self.num_runs  = num_runs
        self.methods   = self._build_methods()
        self.optimizer = UnifiedSimulatedAnnealing(num_reads=20, num_sweeps=2000, seed=42)

    def generate_instances(
        self,
        node_counts: List[int],
        degrees: List[int],
        instances_per_config: int = 5,
    ) -> List[PSpinInstance]:
        instances: List[PSpinInstance] = []
        print(f"\n{'#'*80}")
        print("# Generating Random p-Spin Glass Instances")
        print(f"{'#'*80}")

        # FIX #3 — Compute the *actual* number of valid (n, p) configs for the
        #           progress-bar total so it does not report incorrect progress
        #           when some (p > n) configs are skipped.
        valid_configs = [(n, p) for n in node_counts for p in degrees if p <= n]
        total_to_gen  = len(valid_configs) * instances_per_config
        pbar = tqdm(total=total_to_gen, desc="Generating p-Spin Instances", unit="inst")

        for n, p in valid_configs:
            # FIX #4 — Cap the number of requested interactions at the actual
            #           number of distinct p-subsets to prevent an infinite loop
            #           in the while-loop below (e.g. n=10, p=9 → only 10 edges).
            max_possible    = comb(n, p)
            num_interactions = min(max(5, 2 * n), max_possible)

            for i in range(instances_per_config):
                interactions: List[Tuple[Tuple[int, ...], float]] = []
                seen: set = set()

                while len(interactions) < num_interactions:
                    edge = tuple(sorted(random.sample(range(n), p)))
                    if edge not in seen:
                        seen.add(edge)
                        weight = random.gauss(0, 1.0)
                        interactions.append((edge, weight))

                inst_id = f"N{n}_P{p}_Idx{i}"
                inst    = PSpinInstance(n_nodes=n, degree=p, interactions=interactions, source_id=inst_id)
                instances.append(inst)
                pbar.update(1)

                # Persist instance data.
                data = {
                    'n': n, 'p': p,
                    'interactions': [{'indices': list(e), 'weight': w} for e, w in interactions],
                }
                with open(self.output_dir / "instances" / f"{inst_id}.json", 'w') as f:
                    json.dump(data, f, indent=2)

        pbar.close()
        return instances

    def run_instance(self, instance: PSpinInstance, save_files: bool = True) -> InstanceBenchmarkResults:
        # 1. Ground Truth (brute force — only feasible for n_nodes ≤ 20).
        opt_energy, opt_assign = instance.get_optimal_bruteforce()

        results = InstanceBenchmarkResults(
            instance_id=instance.source_id,
            n_nodes=instance.n_nodes,
            degree=instance.degree,
            n_interactions=len(instance.interactions),
            optimal_energy=float(opt_energy),
            optimal_assignment=[int(x) for x in opt_assign],
            graph_data={
                'n': instance.n_nodes,
                'p': instance.degree,
                'interactions': [{'indices': list(e), 'weight': w} for e, w in instance.interactions],
            },
            metadata={'timestamp': datetime.now().isoformat(), 'num_runs': self.num_runs},
        )

        hobo         = instance.to_hobo()
        total_methods = len(self.methods)
        print(
            f"\n>> Processing: {instance.source_id} "
            f"(Nodes={instance.n_nodes}, Deg={instance.degree}, OptE={opt_energy:.4f})"
        )

        for method_idx, (method_name, method) in enumerate(
            tqdm(self.methods, desc="Methods", leave=False, unit="method")
        ):
            print(f"   ➤ [{method_idx + 1}/{total_methods}] Method: {method_name}")

            try:
                for run_id in range(self.num_runs):
                    sa_seed = hash(f"{instance.source_id}_{method_name}_{run_id}") % (2 ** 32)
                    self.optimizer.seed = sa_seed
                    self.optimizer.rng  = np.random.default_rng(sa_seed)

                    t0          = time.perf_counter()
                    quad_result = method(hobo.copy())
                    quad_time   = (time.perf_counter() - t0) * 1000

                    # FIX #5 — Only use the delta / fast-path when the FERQ evaluator
                    #           is actually available.  If it falls back to a plain
                    #           QuadResult, treat it like every other method.
                    #           Previously, if method_name=="FERQ" but ferq_evaluator
                    #           was absent, a dead delta_fn (returning None) was
                    #           silently passed to the optimizer, causing a
                    #           float(None) TypeError at runtime.
                    use_fast_path = (
                        method_name == "FERQ"
                        and hasattr(quad_result, 'ferq_evaluator')
                    )

                    if use_fast_path:
                        evaluator   = quad_result.ferq_evaluator
                        n_vars_opt  = instance.n_nodes + getattr(quad_result, 'n_aux', 0)

                        def energy_fn(x, _ev=evaluator):
                            if not isinstance(x, np.ndarray):
                                x = np.asarray(x, dtype=np.int32)
                            return float(_ev.evaluate_fast(x))

                        def delta_fn(x, bit_idx, _ev=evaluator):
                            return _ev.compute_delta(x, bit_idx)

                        chosen_delta_fn: Optional[Callable] = delta_fn
                    else:
                        qubo       = quad_result.qubo
                        n_vars_opt = instance.n_nodes + quad_result.n_aux

                        def energy_fn(x, _q=qubo):
                            return float(_q.evaluate(dict(enumerate(x))))

                        # No incremental delta available — standard SA path.
                        chosen_delta_fn = None

                    assignment, energy, opt_time = self.optimizer.optimize(
                        energy_fn, n_vars_opt, delta_fn=chosen_delta_fn
                    )

                    orig_assign  = assignment[:instance.n_nodes]
                    found_energy = instance.compute_energy(orig_assign)

                    # Relative error (avoid division by zero when optimal is ~0).
                    if abs(opt_energy) > 1e-9:
                        rel_err = abs(found_energy - opt_energy) / abs(opt_energy)
                    else:
                        rel_err = abs(found_energy - opt_energy)

                    is_optimal = bool(abs(found_energy - opt_energy) < 1e-6)

                    run_result = OptimizationRunResult(
                        run_id=run_id,
                        method_name=method_name,
                        n_vars_original=instance.n_nodes,
                        n_vars_quadratized=n_vars_opt,
                        n_auxiliary=int(quad_result.n_aux),
                        energy=float(found_energy),
                        optimal_energy=float(opt_energy),
                        relative_error=float(rel_err),
                        is_optimal=is_optimal,
                        runtime_ms=float(quad_time + opt_time),
                        assignment=[int(x) for x in orig_assign],
                    )
                    results.runs.append(run_result)

                    if save_files and run_id == 0:
                        self._save_assignment(instance.source_id, method_name, run_result)

            except Exception as e:
                print(f"      ✗ Error in {method_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if save_files:
            self._save_results(results)
        return results

    def _save_assignment(self, instance_id: str, method_name: str, result: OptimizationRunResult):
        data = {
            'instance_id':   instance_id,
            'method':        method_name,
            'run_id':        result.run_id,
            'assignment':    result.assignment,
            'energy':        result.energy,
            'optimal_energy':result.optimal_energy,
            'rel_error':     result.relative_error,
            'is_optimal':    result.is_optimal,
            'timestamp':     datetime.now().isoformat(),
        }
        path = self.output_dir / "assignments" / f"{instance_id}_{method_name}_run{result.run_id}.json"
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

    def _save_results(self, results: InstanceBenchmarkResults):
        path = self.output_dir / "results" / f"{results.instance_id}.json"
        with open(path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, cls=NumpyEncoder)

    def _generate_plots_and_csv(self, all_results: List[InstanceBenchmarkResults]):
        data = []
        for res in all_results:
            for run in res.runs:
                data.append({
                    'n_nodes':    res.n_nodes,
                    'degree':     res.degree,
                    'method':     run.method_name,
                    'rel_error':  run.relative_error,
                    'is_optimal': 1 if run.is_optimal else 0,
                    'runtime_ms': run.runtime_ms,
                    'n_aux':      run.n_auxiliary,
                })

        if not data:
            print("⚠ No run data collected — skipping plots and CSV.")
            return

        methods = sorted(set(d['method'] for d in data))
        configs = sorted(set((d['n_nodes'], d['degree']) for d in data))

        # ── Plot: Relative Error by Method ─────────────────────────────────
        if _MATPLOTLIB_AVAILABLE:
            fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 5), sharey=True)
            if len(configs) == 1:
                axes = [axes]

            for ax, (n, p) in zip(axes, configs):
                subset = [d for d in data if d['n_nodes'] == n and d['degree'] == p]
                if not subset:
                    continue

                means, stds, labels = [], [], []
                for m in methods:
                    vals = [d['rel_error'] for d in subset if d['method'] == m]
                    means.append(np.mean(vals) if vals else 0.0)
                    stds.append(np.std(vals)  if vals else 0.0)
                    labels.append(m)

                x = np.arange(len(labels))
                ax.bar(x, means, yerr=stds, capsize=3, alpha=0.7)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
                ax.set_title(f"N={n}, p={p}")
                ax.set_ylim(0, max(1.0, max(means) + 0.5) if means else 1.0)
                if ax == axes[0]:
                    ax.set_ylabel("Relative Error")

            plt.suptitle("p-Spin Glass: Relative Error vs Optimal", fontsize=14)
            plt.tight_layout()
            plt.savefig(self.output_dir / "plots" / "rel_error.png", dpi=300)
            print("✓ Plot saved: rel_error.png")
            plt.close()
        else:
            print("⚠ Skipping plots (matplotlib unavailable).")

        # ── CSV Summary ─────────────────────────────────────────────────────
        csv_rows = []
        for m in methods:
            for n, p in configs:
                subset = [d for d in data if d['method'] == m and d['n_nodes'] == n and d['degree'] == p]
                if not subset:
                    continue
                csv_rows.append({
                    'Method':               m,
                    'Nodes':                n,
                    'Degree(p)':            p,
                    'Avg_Rel_Error_MEAN':   np.mean([d['rel_error']  for d in subset]),
                    'Avg_Rel_Error_STD':    np.std( [d['rel_error']  for d in subset]),
                    'Success_Rate':         np.mean([d['is_optimal'] for d in subset]),
                    'Avg_Time':             np.mean([d['runtime_ms'] for d in subset]),
                    'Avg_Aux':              np.mean([d['n_aux']      for d in subset]),
                })

        # FIX #6 — Guard against empty csv_rows before indexing csv_rows[0].
        if not csv_rows:
            print("⚠ No data for CSV summary.")
            return

        csv_path = self.output_dir / "csv" / "pspin_summary.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"✓ CSV saved: {csv_path}")

    def run_full_benchmark(
        self,
        node_counts: List[int],
        degrees: List[int],
        instances_per_config: int = 5,
        save_files: bool = True,
    ) -> List[InstanceBenchmarkResults]:
        instances   = self.generate_instances(node_counts, degrees, instances_per_config)
        all_results = []

        print(f"\n{'#'*80}")
        print(f"# Starting Benchmark: {len(instances)} Instances × {self.num_runs} Runs")
        print(f"{'#'*80}")

        overall_pbar = tqdm(total=len(instances), desc="Overall Instances", unit="inst")
        for inst in instances:
            res = self.run_instance(inst, save_files=save_files)
            all_results.append(res)
            overall_pbar.update(1)
        overall_pbar.close()

        self._generate_plots_and_csv(all_results)
        return all_results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # FIX #7 — Node counts and degrees brought into agreement with the module
    #           docstring and the brute-force feasibility limit (≤ 20 nodes).
    #           Original values [10, 20, 30] / [3, 5, 7] caused n=20 to be
    #           borderline-slow (2^20 ≈ 1 M states) and n=30 to be completely
    #           infeasible (2^30 ≈ 1 B states) for the brute-force oracle.
    NODE_COUNTS          = [10]   # All safe for brute-force (≤ 2^14 = 16 384)
    DEGREES              = [3]      # Native degree range matching the docstring
    INSTANCES_PER_CONFIG = 2
    RUNS_PER_INSTANCE    = 2
    OUTPUT_DIR           = './pspin_benchmark_results'

    print("=" * 80)
    print("HIGH-ORDER BENCHMARK: Random p-Spin Glass Model")
    print("=" * 80)
    print(f"Nodes:            {NODE_COUNTS}")
    print(f"Degrees (p):      {DEGREES}")
    print(f"Instances/Config: {INSTANCES_PER_CONFIG}")
    print(f"Runs/Instance:    {RUNS_PER_INSTANCE}")
    print("=" * 80)

    runner = PSpinBenchmarkRunner(output_dir=OUTPUT_DIR, num_runs=RUNS_PER_INSTANCE)

    t0 = time.perf_counter()
    try:
        all_res = runner.run_full_benchmark(
            node_counts=NODE_COUNTS,
            degrees=DEGREES,
            instances_per_config=INSTANCES_PER_CONFIG,
            save_files=True,
        )
        total_time = time.perf_counter() - t0
        print(f"\n{'='*80}")
        print("BENCHMARK COMPLETE")
        print(f"{'='*80}")
        print(f"Total Time: {total_time / 60:.2f} mins")
        print(f"Results:    {OUTPUT_DIR}/")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()