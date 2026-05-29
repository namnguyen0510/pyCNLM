#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
QUANTUM COMPUTING BENCHMARK: k-Local Hamiltonian Reduction to 2-Local
═══════════════════════════════════════════════════════════════════════════
GOAL: Benchmark reducers on compiling k-local Hamiltonians for 2-local hardware.
PROBLEM: Minimize Spectral Error when reducing H_k (degree k) -> H_2 (degree 2).
CONTEXT: Essential for VQE, QAOA, and Quantum Simulation on NISQ devices.
CONFIG: Qubits = [8, 10, 12], Interaction Order (k) = [3, 4, 5], Runs = 10.
OUTPUT: Generated Hamiltonians, JSON results, CSV summaries, Plots.
═══════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import csv
import time
import math      # FIX #2: needed for math.comb
import random
import hashlib   # FIX #1: deterministic seeding
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable
from itertools import product
# FIX #6: removed unused imports: combinations, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Plotting
try:
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    print("⚠ Matplotlib not found. Plots will be skipped.")

# Progress Bar
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

# Ensure reducers module is accessible
try:
    _current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _current_dir = os.getcwd()

sys.path.insert(0, os.path.dirname(_current_dir))

try:
    from pycnlm.core.HOBOReducers import (
        HOBO, QuadResult,
        DeducReduc,
        # FIX #6: removed unused ELCReduction import
        NTR_KZFD, NTR_ABCG, NTR_ABCG2, NTR_GBP,
        PTR_Ishikawa, PTR_KZ, PTR_GBP, BitFlipping,
        ReductionBySubstitution, FGBZ_Negative, FGBZ_Positive, PairwiseCovers,
        FERQ,
    )
    _REDUCERS_AVAILABLE = True
except ImportError as e:
    _REDUCERS_AVAILABLE = False
    print(f" CRITICAL ERROR: Could not import reducers module. Error: {e}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# JSON HELPERS
# ═══════════════════════════════════════════════════════════════════════════
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def convert_numpy_types(obj):
    if isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_numpy_types(v) for v in obj]
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

def _deterministic_seed(label: str) -> int:
    """
    FIX #1: Python's built-in hash() is randomised per process (PYTHONHASHSEED).
    Use SHA-256 for a stable, cross-session reproducible 32-bit seed instead.
    """
    return int(hashlib.sha256(label.encode()).hexdigest(), 16) % (2 ** 32)

# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class KLocalHamiltonian:
    n_qubits: int
    interaction_order: int  # k
    terms: List[Tuple[Tuple[int, ...], float]]  # ((i, j, k...), coupling_strength)
    source_id: str

    def to_hobo(self) -> HOBO:
        """
        Map Z_i Z_j ... Z_k to binary variables x_i.
        In Ising model: Z = 1 - 2x.
        Product Z_i Z_j ... = (1-2x_i)(1-2x_j)...
        This expansion creates a polynomial in x.
        However, for benchmarking REDUCERS, we usually treat the input as
        a Pseudo-Boolean function directly representing the energy landscape.

        Simplified for Benchmark: Treat the term weight directly as a monomial coefficient
        in a minimization problem E(x) = Sum J_S * Prod_{i in S} x_i.
        This captures the complexity of the interaction graph.
        """
        hobo_terms = {}
        for indices, weight in self.terms:
            key = frozenset(indices)
            hobo_terms[key] = hobo_terms.get(key, 0.0) + weight

        hobo_terms = {k: v for k, v in hobo_terms.items() if abs(v) > 1e-9}
        return HOBO(hobo_terms, n_vars=self.n_qubits)

    def compute_energy(self, assignment: List[int]) -> float:
        if len(assignment) < self.n_qubits:
            assignment = assignment + [0] * (self.n_qubits - len(assignment))

        energy = 0.0
        for indices, weight in self.terms:
            prod = 1.0
            for idx in indices:
                prod *= assignment[idx]
            energy += weight * prod
        return energy

    def get_ground_state_bruteforce(self) -> Tuple[float, List[int]]:
        min_val = float('inf')
        best_assign: List[int] = [0] * self.n_qubits  # safe default

        for assign in product([0, 1], repeat=self.n_qubits):
            val = self.compute_energy(list(assign))
            if val < min_val:
                min_val = val
                best_assign = list(assign)

        return min_val, best_assign


@dataclass
class QCOptimizationResult:
    run_id: int
    method_name: str
    n_qubits_original: int
    n_qubits_total: int  # Original + Ancillas
    n_ancillas: int
    energy_found: float
    true_ground_energy: float
    spectral_error: float  # |E_found - E_true|
    is_exact: bool
    runtime_ms: float
    assignment: List[int]

    def to_dict(self) -> Dict:
        return {
            'run_id': int(self.run_id),
            'method_name': str(self.method_name),
            'n_qubits_original': int(self.n_qubits_original),
            'n_qubits_total': int(self.n_qubits_total),
            'n_ancillas': int(self.n_ancillas),
            'energy_found': float(self.energy_found),
            'true_ground_energy': float(self.true_ground_energy),
            'spectral_error': float(self.spectral_error),
            'is_exact': bool(self.is_exact),
            'runtime_ms': float(self.runtime_ms),
            'assignment': [int(x) for x in self.assignment]
        }


@dataclass
class InstanceBenchmarkResults:
    instance_id: str
    n_qubits: int
    interaction_order: int
    n_terms: int
    true_ground_energy: float
    true_assignment: List[int]
    hamiltonian_data: Dict
    runs: List[QCOptimizationResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # FIX #5: removed the orphaned `data: List[int]` field — it was never
    # populated, never used, and absent from to_dict(), making it a dead
    # copy-paste artifact that creates a misleading public API.

    def to_dict(self) -> Dict:
        return {
            'instance_id': self.instance_id,
            'n_qubits': self.n_qubits,
            'interaction_order': self.interaction_order,
            'n_terms': self.n_terms,
            'true_ground_energy': float(self.true_ground_energy),
            'true_assignment': [int(x) for x in self.true_assignment],
            'hamiltonian_data': self.hamiltonian_data,
            'runs': [r.to_dict() for r in self.runs],
            'metadata': convert_numpy_types(self.metadata)
        }


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZER (Unified SA)
# ═══════════════════════════════════════════════════════════════════════════
class UnifiedSimulatedAnnealing:
    def __init__(self, num_reads: int = 20, num_sweeps: int = 2000,
                 beta_range: Tuple[float, float] = (0.1, 10.0), seed: int = None):
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.beta_range = beta_range
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def optimize(self, energy_fn: Callable, n_vars: int,
                 delta_fn: Optional[Callable] = None) -> Tuple[List[int], float, float]:
        start_time = time.perf_counter()
        beta_min, beta_max = self.beta_range
        T_init = 1.0 / beta_min
        T_min  = 1.0 / beta_max
        decay  = (T_min / T_init) ** (1.0 / max(self.num_sweeps - 1, 1))

        best_energy = float('inf')
        best_assignment: Optional[List[int]] = None

        if delta_fn is not None:
            for _ in range(self.num_reads):
                x = self.rng.integers(0, 2, size=n_vars, dtype=np.int32)
                energy = float(energy_fn(x))
                local_best_energy = energy
                local_best_x = x.copy()
                T = T_init

                for _ in range(self.num_sweeps):
                    i = int(self.rng.integers(n_vars))
                    delta = float(delta_fn(x, i))
                    if delta < 0.0 or self.rng.random() < np.exp(-delta / max(T, 1e-12)):
                        x[i] ^= 1
                        energy += delta
                        if energy < local_best_energy:
                            local_best_energy = energy
                            local_best_x = x.copy()
                    T *= decay

                if local_best_energy < best_energy:
                    best_energy = local_best_energy
                    best_assignment = local_best_x.tolist()
        else:
            for _ in range(self.num_reads):
                x = self.rng.integers(0, 2, size=n_vars).tolist()
                energy = energy_fn(x)
                local_best_energy = energy
                local_best_x = x.copy()
                T = T_init

                for _ in range(self.num_sweeps):
                    i = int(self.rng.integers(n_vars))
                    x[i] = 1 - x[i]
                    new_energy = energy_fn(x)
                    delta = new_energy - energy
                    if delta < 0 or self.rng.random() < np.exp(-delta / max(T, 1e-12)):
                        energy = new_energy
                        if energy < local_best_energy:
                            local_best_energy = energy
                            local_best_x = x.copy()
                    else:
                        x[i] = 1 - x[i]
                    T *= decay

                if local_best_energy < best_energy:
                    best_energy = local_best_energy
                    best_assignment = local_best_x

        runtime_ms = (time.perf_counter() - start_time) * 1000
        return best_assignment or [0] * n_vars, float(best_energy), float(runtime_ms)


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════════════════
class QCHamiltonianBenchmarkRunner:
    METHODS = [
        ("DeducReduc",             DeducReduc()),
        ("NTR_KZFD",               NTR_KZFD()),
        ("NTR_ABCG",               NTR_ABCG()),
        ("NTR_ABCG2",              NTR_ABCG2()),
        ("NTR_GBP",                NTR_GBP()),
        ("PTR_Ishikawa",           PTR_Ishikawa()),
        ("PTR_KZ",                 PTR_KZ()),
        ("PTR_GBP",                PTR_GBP()),
        ("BitFlipping",            BitFlipping()),
        ("ReductionBySubstitution",ReductionBySubstitution()),
        ("FGBZ_Negative",          FGBZ_Negative()),
        ("FGBZ_Positive",          FGBZ_Positive()),
        ("PairwiseCovers",         PairwiseCovers()),
        ("FERQ",                   FERQ(max_degree=15)),
    ]

    def __init__(self, output_dir: str = "qc_hamiltonian_benchmark", num_runs: int = 10):
        self.output_dir = Path(output_dir)
        for sub in ["hamiltonians", "results", "assignments", "csv", "plots"]:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

        self.num_runs = num_runs
        self.optimizer = UnifiedSimulatedAnnealing(num_reads=20, num_sweeps=2000, seed=42)

    def generate_instances(self, qubit_counts: List[int], orders: List[int],
                           instances_per_config: int = 5) -> List[KLocalHamiltonian]:
        instances = []
        print(f"\n{'#'*80}")
        print(f"# Generating Random k-Local Hamiltonians for QC Compilation")
        print(f"{'#'*80}")

        total_to_gen = len(qubit_counts) * len(orders) * instances_per_config
        pbar = tqdm(total=total_to_gen, desc="Generating Hamiltonians", unit="inst")

        for n in qubit_counts:
            for k in orders:
                if k > n:
                    continue

                # FIX #2: Cap num_terms at C(n, k) — the total number of distinct
                # k-subsets.  Without this, the while-loop below can never fill
                # `terms` when num_terms > C(n, k) (e.g. n=3, k=3 → only 1 term
                # possible, but num_terms = max(5, 6) = 6), causing an infinite loop.
                max_possible_terms = math.comb(n, k)
                num_terms = min(max(5, int(2 * n)), max_possible_terms)

                for i in range(instances_per_config):
                    terms = []
                    seen: set = set()

                    while len(terms) < num_terms:
                        edge = tuple(sorted(random.sample(range(n), k)))
                        if edge not in seen:
                            seen.add(edge)
                            strength = random.gauss(0, 1.0)
                            terms.append((edge, strength))

                    inst_id = f"N{n}_K{k}_Idx{i}"
                    inst = KLocalHamiltonian(
                        n_qubits=n, interaction_order=k, terms=terms, source_id=inst_id
                    )
                    instances.append(inst)
                    pbar.update(1)

                    # Save Hamiltonian Data
                    data = {
                        'n': n, 'k': k,
                        'terms': [{'indices': list(e), 'strength': w} for e, w in terms]
                    }
                    with open(self.output_dir / "hamiltonians" / f"{inst_id}.json", 'w') as f:
                        json.dump(data, f, indent=2)

        pbar.close()
        return instances

    def run_instance(self, instance: KLocalHamiltonian,
                     save_files: bool = True) -> InstanceBenchmarkResults:
        # 1. Ground Truth
        true_energy, true_assign = instance.get_ground_state_bruteforce()

        results = InstanceBenchmarkResults(
            instance_id=instance.source_id,
            n_qubits=instance.n_qubits,
            interaction_order=instance.interaction_order,
            n_terms=len(instance.terms),
            true_ground_energy=float(true_energy),
            true_assignment=[int(x) for x in true_assign],
            hamiltonian_data={
                'n': instance.n_qubits,
                'k': instance.interaction_order,
                'terms': [{'indices': list(e), 'strength': w} for e, w in instance.terms]
            },
            metadata={'timestamp': datetime.now().isoformat(), 'num_runs': self.num_runs}
        )

        # Convert to HOBO
        hobo = instance.to_hobo()

        total_methods = len(self.METHODS)
        print(f"\n>> Processing: {instance.source_id} "
              f"(Qubits={instance.n_qubits}, k={instance.interaction_order}, "
              f"E0={true_energy:.4f})")

        for method_idx, (method_name, method) in enumerate(
                tqdm(self.METHODS, desc="Methods", leave=False, unit="method")):
            print(f"   ➤ [{method_idx+1}/{total_methods}] Method: {method_name}")

            try:
                for run_id in range(self.num_runs):
                    # FIX #1: Use SHA-256-based seed for cross-session reproducibility.
                    sa_seed = _deterministic_seed(
                        f"{instance.source_id}_{method_name}_{run_id}"
                    )
                    self.optimizer.seed = sa_seed
                    self.optimizer.rng = np.random.default_rng(sa_seed)

                    t0 = time.perf_counter()
                    quad_result = method(hobo.copy())
                    quad_time = (time.perf_counter() - t0) * 1000

                    # Setup Energy/Delta
                    if method_name == "FERQ" and hasattr(quad_result, 'ferq_evaluator'):
                        evaluator = quad_result.ferq_evaluator
                        def energy_fn(x, _ev=evaluator):
                            if not isinstance(x, np.ndarray):
                                x = np.asarray(x, dtype=np.int32)
                            return float(_ev.evaluate_fast(x))
                        def delta_fn(x, bit_idx, _ev=evaluator):
                            return _ev.compute_delta(x, bit_idx)
                        n_vars_opt = instance.n_qubits
                    else:
                        qubo = quad_result.qubo
                        def energy_fn(x, _q=qubo):
                            return float(_q.evaluate(dict(enumerate(x))))
                        def delta_fn(x, bit_idx): return None
                        n_vars_opt = instance.n_qubits + quad_result.n_aux

                    assignment, energy, opt_time = self.optimizer.optimize(
                        energy_fn, n_vars_opt,
                        delta_fn=delta_fn if method_name == "FERQ" else None
                    )

                    # Extract original qubit assignment
                    orig_assign = assignment[:instance.n_qubits]
                    found_energy = instance.compute_energy(orig_assign)

                    spectral_error = abs(found_energy - true_energy)
                    is_exact = bool(spectral_error < 1e-6)

                    run_result = QCOptimizationResult(
                        run_id=run_id,
                        method_name=method_name,
                        n_qubits_original=instance.n_qubits,
                        n_qubits_total=int(n_vars_opt),
                        n_ancillas=int(quad_result.n_aux),
                        energy_found=float(found_energy),
                        true_ground_energy=float(true_energy),
                        spectral_error=float(spectral_error),
                        is_exact=is_exact,
                        runtime_ms=float(quad_time + opt_time),
                        assignment=[int(x) for x in orig_assign]
                    )
                    results.runs.append(run_result)

                    if save_files and run_id == 0:
                        self._save_assignment(instance.source_id, method_name, run_result)

            except Exception as e:
                print(f"      ✗ Error in {method_name}: {str(e)}")
                continue

        if save_files:
            self._save_results(results)
        return results

    def _save_assignment(self, instance_id, method_name, result: QCOptimizationResult):
        data = {
            'instance_id': instance_id, 'method': method_name, 'run_id': result.run_id,
            'assignment': result.assignment, 'energy': result.energy_found,
            'spectral_error': result.spectral_error, 'is_exact': result.is_exact,
            'timestamp': datetime.now().isoformat()
        }
        path = self.output_dir / "assignments" / f"{instance_id}_{method_name}_run{result.run_id}.json"
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

    def _save_results(self, results: InstanceBenchmarkResults):
        path = self.output_dir / "results" / f"{results.instance_id}.json"
        with open(path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, cls=NumpyEncoder)

    def _generate_plots_and_csv(self, all_results: List[InstanceBenchmarkResults]):
        if not _MATPLOTLIB_AVAILABLE:
            print("⚠ Skipping plots.")
            return

        data = []
        for res in all_results:
            for run in res.runs:
                data.append({
                    'n_qubits': res.n_qubits,
                    'order_k': res.interaction_order,
                    'method': run.method_name,
                    'spectral_error': run.spectral_error,
                    'is_exact': 1 if run.is_exact else 0,
                    'runtime_ms': run.runtime_ms,
                    'n_ancillas': run.n_ancillas
                })

        # FIX #3: Guard against empty data before any plotting or CSV writing.
        if not data:
            print("⚠ No data to plot or summarise.")
            return

        methods = sorted(set(d['method'] for d in data))
        configs = sorted(set((d['n_qubits'], d['order_k']) for d in data))

        # Plot: Spectral Error by Method
        fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 5), sharey=True)
        if len(configs) == 1:
            axes = [axes]

        for ax, (n, k) in zip(axes, configs):
            subset = [d for d in data if d['n_qubits'] == n and d['order_k'] == k]
            if not subset:
                continue

            means, stds, labels = [], [], []
            for m in methods:
                vals = [d['spectral_error'] for d in subset if d['method'] == m]
                means.append(np.mean(vals) if vals else 0.0)
                stds.append(np.std(vals)  if vals else 0.0)
                labels.append(m)

            x = np.arange(len(labels))
            ax.bar(x, means, yerr=stds, capsize=3, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.set_title(f"N={n}, k={k}")

            # FIX #4: 'log' scale crashes (or silently drops bars) when any mean
            # is exactly 0 (a method that always finds the exact ground state).
            # 'symlog' handles zero gracefully by using a linear region near 0
            # and log scaling for larger values.
            ax.set_yscale('symlog', linthresh=1e-6)
            ax.set_ylim(bottom=0)

            if ax == axes[0]:
                ax.set_ylabel("Spectral Error (symlog scale)")

        plt.suptitle("QC Hamiltonian Reduction: Spectral Error", fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "spectral_error.png", dpi=300)
        print("✓ Plot saved: spectral_error.png")
        plt.close()

        # CSV Summary
        csv_rows = []
        for m in methods:
            for n, k in configs:
                subset = [d for d in data
                          if d['method'] == m and d['n_qubits'] == n and d['order_k'] == k]
                if not subset:
                    continue
                csv_rows.append({
                    'Method': m, 'Qubits': n, 'Order(k)': k,
                    'Avg_Spectral_Error': np.mean([d['spectral_error'] for d in subset]),
                    'Exact_Success_Rate': np.mean([d['is_exact']       for d in subset]),
                    'Avg_Time':           np.mean([d['runtime_ms']     for d in subset]),
                    'Avg_Ancillas':       np.mean([d['n_ancillas']     for d in subset])
                })

        # FIX #3 (cont.): csv_rows[0].keys() raises IndexError when csv_rows is empty.
        if not csv_rows:
            print("⚠ No CSV rows to write.")
            return

        csv_path = self.output_dir / "csv" / "qc_hamiltonian_summary.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"✓ CSV saved: {csv_path}")

    def run_full_benchmark(self, qubit_counts: List[int], orders: List[int],
                           instances_per_config: int = 5,
                           save_files: bool = True) -> List[InstanceBenchmarkResults]:
        all_results = []
        instances = self.generate_instances(qubit_counts, orders, instances_per_config)

        print(f"\n{'#'*80}")
        print(f"# Starting QC Benchmark: {len(instances)} Instances x {self.num_runs} Runs")
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
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    QUBIT_COUNTS = [8]       # Number of logical qubits
    INTERACTION_ORDERS = [3]  # k-local interactions
    INSTANCES_PER_CONFIG = 2
    RUNS_PER_INSTANCE = 2
    OUTPUT_DIR = './qc_hamiltonian_benchmark_results'

    print("=" * 80)
    print("QUANTUM COMPUTING BENCHMARK: k-Local Hamiltonian Reduction")
    print("=" * 80)
    print(f"Qubits: {QUBIT_COUNTS}")
    print(f"Interaction Orders (k): {INTERACTION_ORDERS}")
    print(f"Instances/Config: {INSTANCES_PER_CONFIG}")
    print(f"Runs/Instance: {RUNS_PER_INSTANCE}")
    print("=" * 80)

    runner = QCHamiltonianBenchmarkRunner(output_dir=OUTPUT_DIR, num_runs=RUNS_PER_INSTANCE)

    t0 = time.perf_counter()
    try:
        all_res = runner.run_full_benchmark(
            qubit_counts=QUBIT_COUNTS,
            orders=INTERACTION_ORDERS,
            instances_per_config=INSTANCES_PER_CONFIG,
            save_files=True
        )
        total_time = time.perf_counter() - t0
        print(f"\n{'='*80}")
        print("BENCHMARK COMPLETE")
        print(f"{'='*80}")
        print(f"Total Time: {total_time/60:.2f} mins")
        print(f"Results: {OUTPUT_DIR}/")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()