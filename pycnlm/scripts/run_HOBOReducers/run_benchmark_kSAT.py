# ═══════════════════════════════════════════════════════════════════════════
# k-SAT QUADRATIZATION BENCHMARK (Jupyter Notebook Version)
# ═══════════════════════════════════════════════════════════════════════════
# FULL REVISED CODE
# - Added explicit print progression by method
# - Robust tqdm handling (Notebook vs Script)
# - Safe __file__ handling
# ═══════════════════════════════════════════════════════════════════════════

import os
import sys
import json
import csv
import time
import random
import numpy as np
import argparse
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable
from itertools import product
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict

# Robust tqdm import (Notebook vs Standard)
try:
    from tqdm.notebook import tqdm
    _TQDM_MODE = "notebook"
except ImportError:
    from tqdm import tqdm
    _TQDM_MODE = "standard"

# Ensure reducers module is accessible
# Safe handling of __file__ for interactive environments
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
    print(f"⚠ Warning: Could not import reducers module. Error: {e}")
    # Define dummy classes to prevent syntax errors if running without module for testing structure
    class HOBO: pass
    class QuadResult: pass
    class DeducReduc: pass
    # ... (In production, ensure reducers.py is in path)

try:
    import neal
    import dimod
    _NEAL_AVAILABLE = True
    print("✓ Using D-Wave Neal optimizer")
except ImportError:
    _NEAL_AVAILABLE = False
    print("⚠ D-Wave Neal not available, using fallback UnifiedSimulatedAnnealing")


# ═══════════════════════════════════════════════════════════════════════════
# JSON SERIALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class KSATInstance:
    n_vars: int
    n_clauses: int
    k: int
    clauses: List[Tuple[Tuple[int, bool], ...]]
    alpha: float
    source_file: str = ""

    def to_cnf(self) -> str:
        lines = [f"c k-SAT instance (k={self.k}, n={self.n_vars}, m={self.n_clauses})"]
        lines.append(f"c alpha = {self.alpha:.3f}")
        lines.append(f"c Generated: {datetime.now().isoformat()}")
        lines.append(f"c Source: {self.source_file}")
        lines.append(f"p cnf {self.n_vars} {self.n_clauses}")
        for clause in self.clauses:
            lit_str = " ".join(
                f"-{var+1}" if negated else f"{var+1}"
                for var, negated in clause
            )
            lines.append(f"{lit_str} 0")
        return "\n".join(lines)

    def save_cnf(self, filepath: str):
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(self.to_cnf())

    def compute_sat_rate(self, assignment: List[int]) -> float:
        if len(assignment) < self.n_vars:
            assignment = assignment + [0] * (self.n_vars - len(assignment))
        satisfied = 0
        for clause in self.clauses:
            for var_idx, is_negated in clause:
                if var_idx < len(assignment):
                    val = assignment[var_idx]
                    if (not is_negated and val == 1) or (is_negated and val == 0):
                        satisfied += 1
                        break
        return satisfied / len(self.clauses) if self.clauses else 0.0

    def is_satisfied(self, assignment: List[int]) -> bool:
        return self.compute_sat_rate(assignment) >= 1.0 - 1e-9


@dataclass
class OptimizationResult:
    method_name: str
    optimizer_name: str
    n_vars_original: int
    n_vars_quadratized: int
    n_auxiliary: int
    is_satisfied: bool
    energy: float
    runtime_ms: float
    assignment: Optional[List[int]] = None
    sat_rate: float = 0.0
    n_satisfied_clauses: int = 0
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            'method_name': str(self.method_name),
            'optimizer_name': str(self.optimizer_name),
            'n_vars_original': int(self.n_vars_original),
            'n_vars_quadratized': int(self.n_vars_quadratized),
            'n_auxiliary': int(self.n_auxiliary),
            'is_satisfied': bool(self.is_satisfied),
            'energy': float(self.energy),
            'runtime_ms': float(self.runtime_ms),
            'assignment': [int(x) for x in self.assignment] if self.assignment else None,
            'sat_rate': float(self.sat_rate),
            'n_satisfied_clauses': int(self.n_satisfied_clauses),
            'notes': str(self.notes)
        }


@dataclass
class BenchmarkResults:
    instance_id: str
    k: int
    n_vars: int
    n_clauses: int
    alpha: float
    is_satisfiable: bool
    ground_truth_energy: float
    ground_truth_assignment: Optional[List[int]] = None
    source_file: str = ""
    methods: Dict[str, List[OptimizationResult]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'instance_id': str(self.instance_id),
            'k': int(self.k),
            'n_vars': int(self.n_vars),
            'n_clauses': int(self.n_clauses),
            'alpha': float(self.alpha),
            'is_satisfiable': bool(self.is_satisfiable),
            'ground_truth_energy': float(self.ground_truth_energy),
            'ground_truth_assignment': (
                [int(x) for x in self.ground_truth_assignment]
                if self.ground_truth_assignment else None
            ),
            'source_file': self.source_file,
            'methods': {k: [r.to_dict() for r in v] for k, v in self.methods.items()},
            'metadata': convert_numpy_types(self.metadata)
        }


# ═══════════════════════════════════════════════════════════════════════════
# CNF PARSER
# ═══════════════════════════════════════════════════════════════════════════

class CNFParser:
    @staticmethod
    def parse_cnf_file(filepath: str) -> KSATInstance:
        n_vars = n_clauses = 0
        clauses = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('c'):
                    continue
                if line.startswith('p'):
                    parts = line.split()
                    if len(parts) >= 4 and parts[1] == 'cnf':
                        n_vars, n_clauses = int(parts[2]), int(parts[3])
                    continue
                if line.startswith('-') or (line and line[0].isdigit()):
                    literals = list(map(int, line.split()))
                    if literals and literals[-1] == 0:
                        literals = literals[:-1]
                    clause = []
                    for lit in literals:
                        if lit > 0:
                            clause.append((lit - 1, False))
                        else:
                            clause.append((abs(lit) - 1, True))
                    if clause:
                        clauses.append(tuple(clause))
        k = max(len(c) for c in clauses) if clauses else 0
        alpha = n_clauses / n_vars if n_vars > 0 else 0.0
        return KSATInstance(
            n_vars=n_vars, n_clauses=len(clauses), k=k,
            clauses=clauses, alpha=alpha,
            source_file=os.path.basename(filepath)
        )

    @staticmethod
    def load_cnf_folder(folder_path: str) -> List[KSATInstance]:
        instances = []
        folder = Path(folder_path)
        if not folder.exists():
            print(f"⚠ Folder {folder_path} does not exist!")
            return instances
        
        cnf_files = sorted(list(folder.rglob("*.cnf")) + list(folder.rglob("*.CNF")))
        
        if not cnf_files:
            print(f"⚠ No .cnf files found in {folder_path}")
            return instances
        
        print(f"✓ Found {len(cnf_files)} CNF files in {folder_path} (recursive)")
        
        # tqdm progress bar for loading files
        for cnf_file in tqdm(cnf_files, desc="Loading CNF files", unit="file"):
            try:
                instance = CNFParser.parse_cnf_file(str(cnf_file))
                instance.source_file = str(cnf_file.relative_to(folder))
                instances.append(instance)
            except Exception as e:
                print(f"  ✗ Failed to load {cnf_file.name}: {str(e)}")
        
        return instances


# ═══════════════════════════════════════════════════════════════════════════
# SAT TO HOBO CONVERTER
# ═══════════════════════════════════════════════════════════════════════════

class SATToHOBO:
    @staticmethod
    def convert(instance: KSATInstance) -> HOBO:
        terms = {}
        for clause in instance.clauses:
            for term, coeff in SATToHOBO._expand_clause(clause).items():
                terms[term] = terms.get(term, 0.0) + coeff
        terms = {k: v for k, v in terms.items() if abs(v) > 1e-12}
        return HOBO(terms, n_vars=instance.n_vars)

    @staticmethod
    def _expand_clause(clause: Tuple[Tuple[int, bool], ...]) -> Dict[frozenset, float]:
        poly = {frozenset(): 1.0}
        for var_idx, is_negated in clause:
            new_poly = {}
            if is_negated:
                for term, coeff in poly.items():
                    new_term = frozenset(set(term) | {var_idx})
                    new_poly[new_term] = new_poly.get(new_term, 0.0) + coeff
            else:
                for term, coeff in poly.items():
                    new_poly[term] = new_poly.get(term, 0.0) + coeff
                    new_term = frozenset(set(term) | {var_idx})
                    new_poly[new_term] = new_poly.get(new_term, 0.0) - coeff
            poly = new_poly
        return poly


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════

class UnifiedSimulatedAnnealing:
    def __init__(self, num_reads: int = 50, num_sweeps: int = 2000,
                 beta_range: Tuple[float, float] = (0.1, 10.0), seed: int = 42):
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.beta_range = beta_range
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def optimize(
        self,
        energy_fn: Callable,
        n_vars: int,
        delta_fn: Optional[Callable] = None,
    ) -> Tuple[List[int], float, float]:
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

class KSATBenchmarkFromCNF:
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

    def __init__(self, output_dir: str = "ksat_benchmark"):
        self.output_dir = Path(output_dir)
        for sub in ["cnf", "hobo", "results", "summaries", "assignments", "csv"]:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

        self.optimizer = UnifiedSimulatedAnnealing(
            num_reads=50, num_sweeps=2000,
            beta_range=(0.1, 10.0), seed=42,
        )

    def run_instance(self, instance: KSATInstance, instance_id: str,
                     save_files: bool = True, show_progress: bool = True) -> BenchmarkResults:
        results = BenchmarkResults(
            instance_id=instance_id, k=instance.k,
            n_vars=instance.n_vars, n_clauses=instance.n_clauses,
            alpha=instance.alpha, is_satisfiable=False,
            ground_truth_energy=0.0, ground_truth_assignment=None,
            source_file=instance.source_file,
            metadata={'timestamp': datetime.now().isoformat()}
        )

        # [1/3] Convert to HOBO
        t0 = time.perf_counter()
        hobo = SATToHOBO.convert(instance)
        hobo_time = (time.perf_counter() - t0) * 1000
        if save_files:
            self._save_hobo(hobo, instance_id)

        # [2/3] Ground truth
        if instance.n_vars <= 20:
            min_energy = float('inf')
            best_assignment = None
            for assignment in product([0, 1], repeat=instance.n_vars):
                e = hobo.evaluate(dict(enumerate(assignment)))
                if e < min_energy:
                    min_energy = e
                    best_assignment = list(assignment)
            results.ground_truth_energy = float(min_energy)
            results.ground_truth_assignment = [int(x) for x in best_assignment] if best_assignment else None
            results.is_satisfiable = bool(abs(min_energy) < 1e-6)
            if save_files:
                self._save_assignment(instance_id, "ground_truth", best_assignment,
                                      min_energy, instance.compute_sat_rate(best_assignment) if best_assignment else 0.0)
        else:
            results.ground_truth_energy = 0.0
            results.is_satisfiable = True

        # [3/3] Quadratization methods with tqdm AND explicit print progression
        total_methods = len(self.METHODS)
        
        if show_progress:
            method_iter = tqdm(self.METHODS, desc=f"Instance {instance_id[:30]}", unit="method", leave=False)
        else:
            method_iter = self.METHODS

        for method_idx, (method_name, method) in enumerate(method_iter, 1):
            # Explicit print progression by method
            if show_progress:
                print(f"    ➤ [{method_idx}/{total_methods}] Running Method: {method_name}")
            
            method_results = []
            try:
                t0 = time.perf_counter()
                quad_result = method(hobo.copy())
                quad_time = (time.perf_counter() - t0) * 1000

                if method_name == "FERQ" and hasattr(quad_result, 'ferq_evaluator'):
                    evaluator = quad_result.ferq_evaluator

                    def energy_fn(x, _ev=evaluator):
                        if not isinstance(x, np.ndarray):
                            x = np.asarray(x, dtype=np.int32)
                        return float(_ev.evaluate_fast(x))

                    def delta_fn(x, bit_idx, _ev=evaluator):
                        return _ev.compute_delta(x, bit_idx)

                    n_vars_opt = instance.n_vars
                    assignment, energy, opt_time = self.optimizer.optimize(
                        energy_fn, n_vars_opt, delta_fn=delta_fn
                    )
                else:
                    qubo = quad_result.qubo

                    def energy_fn(assignment, _q=qubo):
                        return float(_q.evaluate(dict(enumerate(assignment))))

                    n_vars_opt = instance.n_vars + quad_result.n_aux
                    assignment, energy, opt_time = self.optimizer.optimize(
                        energy_fn, n_vars_opt
                    )

                if assignment:
                    sat_rate = instance.compute_sat_rate(assignment[:instance.n_vars])
                    n_satisfied = int(sat_rate * instance.n_clauses)
                    is_satisfied = bool(sat_rate >= 1.0 - 1e-9)
                else:
                    sat_rate = n_satisfied = 0
                    is_satisfied = False

                opt_result = OptimizationResult(
                    method_name=method_name,
                    optimizer_name="unified_sa",
                    n_vars_original=int(instance.n_vars),
                    n_vars_quadratized=int(n_vars_opt),
                    n_auxiliary=int(quad_result.n_aux),
                    is_satisfied=is_satisfied,
                    energy=float(energy),
                    runtime_ms=float(quad_time + opt_time),
                    assignment=[int(v) for v in assignment[:instance.n_vars]] if assignment else None,
                    sat_rate=float(sat_rate),
                    n_satisfied_clauses=n_satisfied,
                )
                method_results.append(opt_result)

                if save_files and assignment:
                    self._save_assignment(instance_id, method_name,
                                          assignment[:instance.n_vars], energy, sat_rate)

                results.methods[method_name] = method_results

            except Exception as e:
                print(f"      ✗ Error in {method_name}: {str(e)}")
                results.methods[method_name] = []

        if save_files:
            self._save_results(results, instance_id)
        return results

    def _save_hobo(self, hobo: HOBO, instance_id: str):
        data = {
            'n_vars': int(hobo.n_vars), 'degree': int(hobo.degree),
            'n_terms': int(len(hobo.terms)),
            'terms': [{'variables': sorted(list(t)), 'coefficient': float(c)}
                      for t, c in hobo.terms.items()]
        }
        with open(self.output_dir / "hobo" / f"{instance_id}.json", 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

    def _save_assignment(self, instance_id, method_name, assignment, energy, sat_rate):
        data = {
            'instance_id': instance_id, 'method': method_name,
            'assignment': [int(x) for x in assignment],
            'energy': float(energy), 'sat_rate': float(sat_rate),
            'timestamp': datetime.now().isoformat()
        }
        with open(self.output_dir / "assignments" / f"{instance_id}_{method_name}.json", 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

    def _save_results(self, results: BenchmarkResults, instance_id: str):
        with open(self.output_dir / "results" / f"{instance_id}.json", 'w') as f:
            json.dump(results.to_dict(), f, indent=2, cls=NumpyEncoder)

    def _save_csv_summary(self, all_results: List[BenchmarkResults]):
        data_by_method_k = defaultdict(lambda: defaultdict(list))
        for results in all_results:
            k = results.k
            for method_name, opt_results in results.methods.items():
                for opt_result in opt_results:
                    data_by_method_k[(method_name, k)]['sat_rate'].append(opt_result.sat_rate)
                    data_by_method_k[(method_name, k)]['runtime_ms'].append(opt_result.runtime_ms)
                    data_by_method_k[(method_name, k)]['n_auxiliary'].append(opt_result.n_auxiliary)
                    data_by_method_k[(method_name, k)]['energy'].append(opt_result.energy)
                    data_by_method_k[(method_name, k)]['is_satisfied'].append(
                        1 if opt_result.is_satisfied else 0)

        csv_data = []
        for (method_name, k), metrics in sorted(data_by_method_k.items()):
            for metric_name, values in metrics.items():
                if values:
                    csv_data.append({
                        'method': method_name, 'k': k, 'metric': metric_name,
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)) if len(values) > 1 else 0.0,
                        'n_samples': len(values)
                    })

        csv_path = (self.output_dir / "csv" /
                    f"summary_by_k_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['method', 'k', 'metric', 'mean', 'std', 'n_samples'])
            writer.writeheader()
            writer.writerows(csv_data)
        return csv_path

    def run_full_benchmark(self, cnf_folder: str, save_files: bool = True, 
                           show_progress: bool = True) -> List[BenchmarkResults]:
        all_results = []
        print(f"\n{'#'*80}")
        print(f"# Loading CNF instances from: {cnf_folder}")
        print(f"{'#'*80}")

        instances = CNFParser.load_cnf_folder(cnf_folder)
        if not instances:
            print("⚠ No instances loaded, exiting...")
            return all_results

        instances_by_k = defaultdict(list)
        for inst in instances:
            instances_by_k[inst.k].append(inst)

        # Main progress bar for all instances
        total_instances = len(instances)
        if show_progress:
            overall_pbar = tqdm(total=total_instances, desc="Overall Progress", unit="instance")
        else:
            overall_pbar = None

        for k in sorted(instances_by_k.keys()):
            print(f"\n{'#'*80}")
            print(f"# k-SAT Benchmark: k={k} ({len(instances_by_k[k])} instances)")
            print(f"{'#'*80}")
            
            for instance in instances_by_k[k]:
                safe_name = instance.source_file.replace('/', '_').replace('\\', '_').replace('.cnf','').replace('.CNF','')
                instance_id = f"k{k}_{safe_name}"
                
                if save_files:
                    instance.save_cnf(self.output_dir / "cnf" / f"{instance_id}.cnf")
                
                print(f"\n>> Processing Instance: {instance_id}")
                results = self.run_instance(instance, instance_id, save_files, show_progress)
                all_results.append(results)
                
                if overall_pbar:
                    overall_pbar.update(1)

        if overall_pbar:
            overall_pbar.close()

        self._generate_summary(all_results)
        if save_files:
            csv_path = self._save_csv_summary(all_results)
            print(f"\n✓ CSV summary saved to: {csv_path}")
        return all_results

    def _generate_summary(self, all_results: List[BenchmarkResults]):
        print(f"\n{'='*80}")
        print("GENERATING SUMMARY (Stratified by k)")
        print(f"{'='*80}")

        results_by_k = defaultdict(list)
        for r in all_results:
            results_by_k[r.k].append(r)

        for k in sorted(results_by_k.keys()):
            k_results = results_by_k[k]
            print(f"\n{'='*80}")
            print(f"RESULTS FOR k={k} ({len(k_results)} instances)")
            print(f"{'='*80}")
            self._print_method_table(k_results)

        print(f"\n{'='*80}")
        print(f"OVERALL SUMMARY (All k combined, {len(all_results)} total instances)")
        print(f"{'='*80}")
        self._print_method_table(all_results, save_json=True)

    def _print_method_table(self, results_list: List[BenchmarkResults],
                            save_json: bool = False):
        method_stats: Dict[str, Dict] = {}
        for results in results_list:
            for method_name, opt_results in results.methods.items():
                if method_name not in method_stats:
                    method_stats[method_name] = {
                        'total_runs': 0, 'successful_runs': 0,
                        'total_time_ms': 0.0, 'avg_aux': 0.0,
                        'avg_sat_rate': 0.0, 'avg_energy': 0.0,
                    }
                for opt_result in opt_results:
                    s = method_stats[method_name]
                    s['total_runs'] += 1
                    if opt_result.is_satisfied:
                        s['successful_runs'] += 1
                    s['total_time_ms'] += opt_result.runtime_ms
                    s['avg_aux'] += opt_result.n_auxiliary
                    s['avg_sat_rate'] += opt_result.sat_rate
                    s['avg_energy'] += opt_result.energy

        for name, s in method_stats.items():
            n = s['total_runs']
            if n > 0:
                s['success_rate'] = s['successful_runs'] / n
                s['avg_time_ms']  = s['total_time_ms'] / n
                s['avg_aux']      = s['avg_aux'] / n
                s['avg_sat_rate'] = s['avg_sat_rate'] / n
                s['avg_energy']   = s['avg_energy'] / n

        print(f"\n{'Method':<25} | {'Success':<10} | {'SAT Rate':<10} | "
              f"{'Avg Time (ms)':<15} | {'Avg Aux':<10}")
        print("-" * 85)
        for name, s in sorted(method_stats.items(),
                               key=lambda x: (-x[1].get('avg_sat_rate', 0),
                                              -x[1].get('success_rate', 0),
                                               x[1].get('avg_aux', 0))):
            print(f"{name:<25} | {s.get('success_rate',0)*100:>6.1f}%     | "
                  f"{s.get('avg_sat_rate',0):>8.3f}   | "
                  f"{s.get('avg_time_ms',0):>12.2f}   | {s.get('avg_aux',0):>8.2f}")

        if save_json:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_instances': len(results_list),
                'by_method': convert_numpy_types(method_stats)
            }
            summary_path = (self.output_dir / "summaries" /
                            f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, cls=NumpyEncoder)
            print(f"\n✓ JSON summary saved to: {summary_path}")


# ═══════════════════════════════════════════════════════════════════════════
# RUN BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════

# Configuration
CNF_DIR = './test_CNF'           # Your folder with subfolders containing .cnf files
OUTPUT_DIR = './test_HOBO'
SAVE_FILES = True
SHOW_PROGRESS = True         # Enable tqdm progress bars AND print progression
SELECTED_METHODS = None      # None = all methods, or list like ['FERQ', 'NTR_KZFD']

print("=" * 80)
print("k-SAT QUADRATIZATION BENCHMARK SUITE (Jupyter Notebook Version)")
print("=" * 80)
print(f"CNF directory:  {CNF_DIR} (Recursive)")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Optimizer: Unified SA — same algorithm for ALL methods")
print(f"Progress bars: {'Enabled' if SHOW_PROGRESS else 'Disabled'}")
print("=" * 80)

# Initialize and run
benchmark = KSATBenchmarkFromCNF(output_dir=OUTPUT_DIR)

if SELECTED_METHODS:
    benchmark.METHODS = [(n, m) for n, m in benchmark.METHODS if n in SELECTED_METHODS]
    print(f"Selected methods: {SELECTED_METHODS}")

t0 = time.perf_counter()
all_results = benchmark.run_full_benchmark(
    cnf_folder=CNF_DIR, 
    save_files=SAVE_FILES,
    show_progress=SHOW_PROGRESS
)
total_time = time.perf_counter() - t0

print(f"\n{'='*80}")
print(f"BENCHMARK COMPLETE")
print(f"{'='*80}")
print(f"Total instances: {len(all_results)}")
print(f"Total time: {total_time/60:.2f} minutes")
print(f"Results saved to: {OUTPUT_DIR}/")
print(f"{'='*80}")