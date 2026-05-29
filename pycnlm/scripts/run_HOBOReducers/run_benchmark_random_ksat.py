"""
benchmark_ksat.py
Comprehensive k-SAT Benchmark for All Quadratization Methods.

Generates k-SAT instances at phase transition, converts to HOBO,
benchmarks all quadratization methods with SAME optimizer (fair comparison).
Saves results to CSV with mean ± std stratified by k.

Usage:
    python benchmark_ksat.py --k 3 5 7 10 --n-vars 20 --n-instances 10
    python benchmark_ksat.py --output-dir ksat_benchmark --save-all
"""
import os
import sys
import json
import csv
import time
import random
import numpy as np
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable
from itertools import product
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pycnlm.core.HOBOReducers import (
    HOBO, QuadResult,
    # Zero Aux
    DeducReduc, ELCReduction,
    # NTR
    NTR_KZFD, NTR_ABCG, NTR_ABCG2, NTR_GBP,
    # PTR
    PTR_Ishikawa, PTR_KZ, PTR_GBP, BitFlipping,
    # Arbitrary
    ReductionBySubstitution, FGBZ_Negative, FGBZ_Positive, PairwiseCovers,
    # FERQ
    FERQ,
)


# Try to import package optimizers
try:
    import neal
    import dimod
    _NEAL_AVAILABLE = True
    print("✓ Using D-Wave Neal optimizer")
except ImportError:
    _NEAL_AVAILABLE = False
    print("⚠ D-Wave Neal not available, using fallback")


# ═══════════════════════════════════════════════════════════════════════════
# JSON SERIALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
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
    """Recursively convert numpy types to native Python types for JSON."""
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
    else:
        return obj


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class KSATInstance:
    """Represents a k-SAT instance."""
    n_vars: int
    n_clauses: int
    k: int
    clauses: List[Tuple[Tuple[int, bool], ...]]
    alpha: float
    
    def to_cnf(self) -> str:
        """Convert to DIMACS CNF format."""
        lines = [f"c k-SAT instance (k={self.k}, n={self.n_vars}, m={self.n_clauses})"]
        lines.append(f"c alpha = {self.alpha:.3f}")
        lines.append(f"c Generated: {datetime.now().isoformat()}")
        lines.append(f"p cnf {self.n_vars} {self.n_clauses}")
        
        for clause in self.clauses:
            lit_str = " ".join(
                f"-{var+1}" if negated else f"{var+1}" 
                for var, negated in clause
            )
            lines.append(f"{lit_str} 0")
        
        return "\n".join(lines)
    
    def save_cnf(self, filepath: str):
        """Save to DIMACS CNF file."""
        with open(filepath, 'w') as f:
            f.write(self.to_cnf())
    
    def compute_sat_rate(self, assignment: List[int]) -> float:
        """Compute the SAT rate (fraction of satisfied clauses)."""
        if len(assignment) < self.n_vars:
            assignment = assignment + [0] * (self.n_vars - len(assignment))
        
        satisfied = 0
        for clause in self.clauses:
            clause_satisfied = False
            for var_idx, is_negated in clause:
                if var_idx < len(assignment):
                    var_val = assignment[var_idx]
                    # Clause is satisfied if any literal is true
                    if (not is_negated and var_val == 1) or (is_negated and var_val == 0):
                        clause_satisfied = True
                        break
            if clause_satisfied:
                satisfied += 1
        
        return satisfied / len(self.clauses) if len(self.clauses) > 0 else 0.0
    
    def is_satisfied(self, assignment: List[int]) -> bool:
        """Check if assignment satisfies all clauses."""
        return self.compute_sat_rate(assignment) >= 1.0 - 1e-9


@dataclass
class OptimizationResult:
    """Results from optimizing a quadratized problem."""
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
        """Convert to dict with native Python types."""
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
    """Complete benchmark results for one k-SAT instance."""
    instance_id: str
    k: int
    n_vars: int
    n_clauses: int
    alpha: float
    is_satisfiable: bool
    ground_truth_energy: float
    ground_truth_assignment: Optional[List[int]] = None
    methods: Dict[str, List[OptimizationResult]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'instance_id': str(self.instance_id),
            'k': int(self.k),
            'n_vars': int(self.n_vars),
            'n_clauses': int(self.n_clauses),
            'alpha': float(self.alpha),
            'is_satisfiable': bool(self.is_satisfiable),
            'ground_truth_energy': float(self.ground_truth_energy),
            'ground_truth_assignment': [int(x) for x in self.ground_truth_assignment] if self.ground_truth_assignment else None,
            'methods': {
                k: [r.to_dict() for r in v] 
                for k, v in self.methods.items()
            },
            'metadata': convert_numpy_types(self.metadata)
        }


# ═══════════════════════════════════════════════════════════════════════════
# K-SAT INSTANCE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

class KSATGenerator:
    """Generate random k-SAT instances at phase transition."""
    
    ALPHA_CRITICAL = {
        3: 4.27,
        5: 21.1,
        7: 87.3,
        10: 708.7,
    }
    
    @classmethod
    def generate(cls, n_vars: int, k: int, alpha: Optional[float] = None, 
                 seed: int = 42) -> KSATInstance:
        """Generate a random k-SAT instance."""
        random.seed(seed)
        np.random.seed(seed)
        
        if alpha is None:
            alpha = cls.ALPHA_CRITICAL.get(k, 2 ** k * np.log(2))
        
        n_clauses = int(alpha * n_vars)
        
        clauses = []
        used_clauses = set()
        
        max_attempts = n_clauses * 10
        attempts = 0
        
        while len(clauses) < n_clauses and attempts < max_attempts:
            vars_subset = random.sample(range(n_vars), k)
            negations = tuple(random.choice([True, False]) for _ in range(k))
            clause_sig = tuple(sorted(zip(vars_subset, negations)))
            
            if clause_sig not in used_clauses:
                used_clauses.add(clause_sig)
                clause = tuple(zip(vars_subset, negations))
                clauses.append(clause)
            
            attempts += 1
        
        return KSATInstance(
            n_vars=n_vars,
            n_clauses=len(clauses),
            k=k,
            clauses=clauses,
            alpha=len(clauses) / n_vars if n_vars > 0 else 0
        )
    
    @classmethod
    def generate_hard_instances(cls, n_vars: int, k: int, n_instances: int = 10,
                               seed_start: int = 0) -> List[KSATInstance]:
        """Generate multiple hard instances near phase transition."""
        instances = []
        for i in range(n_instances):
            instance = cls.generate(n_vars, k, seed=seed_start + i)
            instances.append(instance)
        return instances


# ═══════════════════════════════════════════════════════════════════════════
# SAT TO HOBO CONVERTER
# ═══════════════════════════════════════════════════════════════════════════

class SATToHOBO:
    """Convert k-SAT instance to HOBO."""
    
    @staticmethod
    def convert(instance: KSATInstance) -> HOBO:
        """Convert k-SAT instance to HOBO."""
        terms = {}
        
        for clause in instance.clauses:
            clause_terms = SATToHOBO._expand_clause(clause)
            for term, coeff in clause_terms.items():
                terms[term] = terms.get(term, 0.0) + coeff
        
        terms = {k: v for k, v in terms.items() if abs(v) > 1e-12}
        return HOBO(terms, n_vars=instance.n_vars)
    
    @staticmethod
    def _expand_clause(clause: Tuple[Tuple[int, bool], ...]) -> Dict[frozenset, float]:
        """Expand a single clause to polynomial terms."""
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
# UNIFIED OPTIMIZER (SAME FOR ALL METHODS - FAIR COMPARISON)
# ═══════════════════════════════════════════════════════════════════════════

class UnifiedSimulatedAnnealing:
    """
    UNIFIED Simulated Annealing optimizer.
    SAME optimizer for ALL quadratization methods - FAIR COMPARISON!
    
    Works with:
    - QUBO (degree ≤ 2): Standard quadratic form
    - HOBO (degree > 2): Higher-order via direct evaluation
    - FERQ: Fermat-QUBO via custom evaluator
    """
    
    def __init__(self, num_reads: int = 50, num_sweeps: int = 2000,
                 beta_range: Tuple[float, float] = (0.1, 10.0), seed: int = 42):
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.beta_range = beta_range
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    def optimize(self, energy_fn: Callable, n_vars: int) -> Tuple[List[int], float, float]:
        """
        Optimize using simulated annealing.
        
        Parameters
        ----------
        energy_fn : Callable
            Function that takes assignment (list of 0/1) and returns energy
        n_vars : int
            Number of variables
        
        Returns
        -------
        assignment : List[int]
            Best assignment found
        energy : float
            Best energy
        runtime_ms : float
            Optimization time in milliseconds
        """
        start_time = time.perf_counter()
        
        beta_min, beta_max = self.beta_range
        T_init = 1.0 / beta_min
        T_min = 1.0 / beta_max
        decay = (T_min / T_init) ** (1.0 / max(self.num_sweeps - 1, 1))
        
        best_energy = float('inf')
        best_assignment = None
        
        for _ in range(self.num_reads):
            # Random initial state
            x = self.rng.integers(0, 2, size=n_vars).tolist()
            energy = energy_fn(x)
            
            local_best_energy = energy
            local_best_x = x.copy()
            T = T_init
            
            for _ in range(self.num_sweeps):
                # Pick random bit to flip
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
                    x[i] = 1 - x[i]  # Revert
                
                T *= decay
            
            if local_best_energy < best_energy:
                best_energy = local_best_energy
                best_assignment = local_best_x
        
        runtime_ms = (time.perf_counter() - start_time) * 1000
        
        return best_assignment or [0] * n_vars, float(best_energy), float(runtime_ms)


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER (WITH CSV EXPORT & STRATIFIED OUTPUT)
# ═══════════════════════════════════════════════════════════════════════════

class KSATBenchmark:
    """
    Complete benchmark suite for k-SAT quadratization.
    FAIR COMPARISON: All methods use the SAME optimizer!
    """
    
    METHODS = [
        ("DeducReduc", DeducReduc()),
        ("ELCReduction", ELCReduction()),
        ("NTR_KZFD", NTR_KZFD()),
        ("NTR_ABCG", NTR_ABCG()),
        ("NTR_ABCG2", NTR_ABCG2()),
        ("NTR_GBP", NTR_GBP()),
        ("PTR_Ishikawa", PTR_Ishikawa()),
        ("PTR_KZ", PTR_KZ()),
        ("PTR_GBP", PTR_GBP()),
        ("BitFlipping", BitFlipping()),
        ("ReductionBySubstitution", ReductionBySubstitution()),
        ("FGBZ_Negative", FGBZ_Negative()),
        ("FGBZ_Positive", FGBZ_Positive()),
        ("PairwiseCovers", PairwiseCovers()),
        ("FERQ", FERQ(max_degree=15)),
    ]
    
    def __init__(self, output_dir: str = "ksat_benchmark"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "cnf").mkdir(exist_ok=True)
        (self.output_dir / "hobo").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "summaries").mkdir(exist_ok=True)
        (self.output_dir / "assignments").mkdir(exist_ok=True)
        (self.output_dir / "csv").mkdir(exist_ok=True)
        
        # SINGLE OPTIMIZER FOR ALL METHODS - FAIR COMPARISON!
        self.optimizer = UnifiedSimulatedAnnealing(
            num_reads=50,
            num_sweeps=2000,
            beta_range=(0.1, 10.0),
            seed=42
        )
    
    def run_instance(self, instance: KSATInstance, instance_id: str,
                    save_files: bool = True) -> BenchmarkResults:
        """Run complete benchmark on one k-SAT instance."""
        print(f"\n{'='*80}")
        print(f"Instance: {instance_id} (k={instance.k}, n={instance.n_vars}, m={instance.n_clauses}, α={instance.alpha:.3f})")
        print(f"{'='*80}")
        
        results = BenchmarkResults(
            instance_id=instance_id,
            k=instance.k,
            n_vars=instance.n_vars,
            n_clauses=instance.n_clauses,
            alpha=instance.alpha,
            is_satisfiable=False,
            ground_truth_energy=0.0,
            ground_truth_assignment=None,
            metadata={'timestamp': datetime.now().isoformat()}
        )
        
        # Convert to HOBO
        print("\n[1/3] Converting SAT to HOBO...")
        start_time = time.perf_counter()
        hobo = SATToHOBO.convert(instance)
        hobo_time = (time.perf_counter() - start_time) * 1000
        print(f"  ✓ HOBO: {hobo.n_vars} vars, degree={hobo.degree}, {len(hobo.terms)} terms ({hobo_time:.2f}ms)")
        
        if save_files:
            self._save_hobo(hobo, instance_id)
        
        # Get ground truth
        print("\n[2/3] Computing ground truth...")
        if instance.n_vars <= 20:
            min_energy = float('inf')
            best_assignment = None
            
            for assignment in product([0, 1], repeat=instance.n_vars):
                assign_dict = dict(enumerate(assignment))
                energy = hobo.evaluate(assign_dict)
                if energy < min_energy:
                    min_energy = energy
                    best_assignment = list(assignment)
            
            results.ground_truth_energy = float(min_energy)
            results.ground_truth_assignment = [int(x) for x in best_assignment] if best_assignment else None
            results.is_satisfiable = bool(abs(min_energy) < 1e-6)
            gt_sat_rate = instance.compute_sat_rate(best_assignment) if best_assignment else 0.0
            print(f"  ✓ Ground truth: {min_energy:.6f} (SAT={results.is_satisfiable}, Rate={gt_sat_rate:.3f})")
            
            if save_files:
                self._save_assignment(instance_id, "ground_truth", best_assignment, 
                                     min_energy, gt_sat_rate)
        else:
            results.ground_truth_energy = 0.0
            results.ground_truth_assignment = None
            results.is_satisfiable = True
            print(f"  ⚠ Skipped (n={instance.n_vars} > 20), assuming SAT")
        
        # Run quadratization methods - ALL USE SAME OPTIMIZER!
        print("\n[3/3] Running quadratization methods (ALL use SAME optimizer)...")
        for method_idx, (method_name, method) in enumerate(self.METHODS, 1):
            print(f"\n  [{method_idx}/{len(self.METHODS)}] Method: {method_name}...")
            method_results = []
            
            try:
                # Apply quadratization
                start_time = time.perf_counter()
                quad_result = method(hobo.copy())
                quad_time = (time.perf_counter() - start_time) * 1000
                
                print(f"      Quadratization: {quad_time:.2f}ms, {quad_result.n_aux} aux vars")
                
                # Create energy function for optimizer - SAME FOR ALL METHODS!
                if method_name == "FERQ" and hasattr(quad_result, 'ferq_evaluator'):
                    # FERQ: Use FERQ evaluator directly
                    evaluator = quad_result.ferq_evaluator
                    def energy_fn(assignment):
                        x = np.array(assignment, dtype=np.int32)
                        return float(evaluator.evaluate_fast(x))
                    n_vars_opt = instance.n_vars  # FERQ is ancilla-free
                else:
                    # Other methods: Use quadratized QUBO/HOBO
                    qubo = quad_result.qubo
                    def energy_fn(assignment):
                        assign_dict = dict(enumerate(assignment))
                        return float(qubo.evaluate(assign_dict))
                    n_vars_opt = instance.n_vars + quad_result.n_aux
                
                # OPTIMIZE WITH SAME OPTIMIZER FOR ALL METHODS!
                assignment, energy, opt_time = self.optimizer.optimize(energy_fn, n_vars_opt)
                
                # Compute SAT rate from ORIGINAL CNF (FAIR METRIC!)
                if assignment:
                    sat_rate = instance.compute_sat_rate(assignment[:instance.n_vars])
                    n_satisfied = int(sat_rate * instance.n_clauses)
                    is_satisfied = bool(sat_rate >= 1.0 - 1e-9)
                else:
                    sat_rate = 0.0
                    n_satisfied = 0
                    is_satisfied = False
                
                opt_result = OptimizationResult(
                    method_name=method_name,
                    optimizer_name="unified_sa",  # SAME optimizer for all!
                    n_vars_original=int(instance.n_vars),
                    n_vars_quadratized=int(n_vars_opt),
                    n_auxiliary=int(quad_result.n_aux),
                    is_satisfied=is_satisfied,
                    energy=float(energy),
                    runtime_ms=float(quad_time + opt_time),
                    assignment=[int(x) for x in assignment[:instance.n_vars]] if assignment else None,
                    sat_rate=float(sat_rate),
                    n_satisfied_clauses=n_satisfied
                )
                
                method_results.append(opt_result)
                
                status = "✓" if is_satisfied else "✗"
                print(f"      unified_sa: SAT={sat_rate:.3f}, E={energy:.6f} ({opt_result.runtime_ms:.2f}ms) {status}")
                
                # Save assignment
                if save_files and assignment:
                    self._save_assignment(instance_id, method_name, assignment[:instance.n_vars],
                                        energy, sat_rate)
                
                results.methods[method_name] = method_results
                
            except Exception as e:
                print(f"    ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
                results.methods[method_name] = []
        
        if save_files:
            self._save_results(results, instance_id)
        
        return results
    
    def _save_hobo(self, hobo: HOBO, instance_id: str):
        """Save HOBO to JSON."""
        data = {
            'n_vars': int(hobo.n_vars),
            'degree': int(hobo.degree),
            'n_terms': int(len(hobo.terms)),
            'terms': [
                {'variables': sorted(list(term)), 'coefficient': float(coeff)}
                for term, coeff in hobo.terms.items()
            ]
        }
        filepath = self.output_dir / "hobo" / f"{instance_id}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
    
    def _save_assignment(self, instance_id: str, method_name: str, 
                        assignment: List[int], energy: float, sat_rate: float):
        """Save assignment to JSON file."""
        data = {
            'instance_id': instance_id,
            'method': method_name,
            'assignment': [int(x) for x in assignment],
            'energy': float(energy),
            'sat_rate': float(sat_rate),
            'timestamp': datetime.now().isoformat()
        }
        filepath = self.output_dir / "assignments" / f"{instance_id}_{method_name}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
    
    def _save_results(self, results: BenchmarkResults, instance_id: str):
        """Save benchmark results to JSON."""
        filepath = self.output_dir / "results" / f"{instance_id}.json"
        with open(filepath, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, cls=NumpyEncoder)
    
    def _save_csv_summary(self, all_results: List[BenchmarkResults]):
        """
        Save summary statistics as CSV, stratified by k.
        Format: method, k, metric, mean, std
        """
        # Collect all data by (method, k)
        data_by_method_k = defaultdict(lambda: defaultdict(list))
        
        for results in all_results:
            k = results.k
            for method_name, opt_results in results.methods.items():
                for opt_result in opt_results:
                    data_by_method_k[(method_name, k)]['sat_rate'].append(opt_result.sat_rate)
                    data_by_method_k[(method_name, k)]['runtime_ms'].append(opt_result.runtime_ms)
                    data_by_method_k[(method_name, k)]['n_auxiliary'].append(opt_result.n_auxiliary)
                    data_by_method_k[(method_name, k)]['energy'].append(opt_result.energy)
                    data_by_method_k[(method_name, k)]['is_satisfied'].append(1 if opt_result.is_satisfied else 0)
        
        # Compute mean and std for each (method, k, metric)
        csv_data = []
        for (method_name, k), metrics in sorted(data_by_method_k.items()):
            for metric_name, values in metrics.items():
                if len(values) > 0:
                    mean_val = np.mean(values)
                    std_val = np.std(values) if len(values) > 1 else 0.0
                    csv_data.append({
                        'method': method_name,
                        'k': k,
                        'metric': metric_name,
                        'mean': float(mean_val),
                        'std': float(std_val),
                        'n_samples': len(values)
                    })
        
        # Save to CSV
        csv_path = self.output_dir / "csv" / f"summary_by_k_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ['method', 'k', 'metric', 'mean', 'std', 'n_samples']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        return csv_path
    
    def run_full_benchmark(self, k_values: List[int], n_vars: int,
                          n_instances: int = 10, save_files: bool = True) -> List[BenchmarkResults]:
        """Run complete benchmark across multiple k values and instances."""
        all_results = []
        
        for k in k_values:
            print(f"\n{'#'*80}")
            print(f"# k-SAT Benchmark: k={k}")
            print(f"#"*80)
            
            instances = KSATGenerator.generate_hard_instances(
                n_vars=n_vars, k=k, n_instances=n_instances,
                seed_start=k * 1000
            )
            
            for i, instance in enumerate(instances):
                instance_id = f"k{k}_n{n_vars}_inst{i:03d}"
                
                if save_files:
                    cnf_path = self.output_dir / "cnf" / f"{instance_id}.cnf"
                    instance.save_cnf(cnf_path)
                
                results = self.run_instance(instance, instance_id, save_files)
                all_results.append(results)
        
        self._generate_summary(all_results)
        
        # Save CSV summary
        if save_files:
            csv_path = self._save_csv_summary(all_results)
            print(f"\n✓ CSV summary saved to: {csv_path}")
        
        return all_results
    
    def _generate_summary(self, all_results: List[BenchmarkResults]):
        """Generate summary statistics stratified by k."""
        print(f"\n{'='*80}")
        print("GENERATING SUMMARY (Stratified by k)")
        print(f"{'='*80}")
        
        # Group by k
        results_by_k = defaultdict(list)
        for results in all_results:
            results_by_k[results.k].append(results)
        
        # Print summary for each k
        for k in sorted(results_by_k.keys()):
            k_results = results_by_k[k]
            print(f"\n{'='*80}")
            print(f"RESULTS FOR k={k} ({len(k_results)} instances)")
            print(f"{'='*80}")
            
            method_stats = {}
            for results in k_results:
                for method_name, opt_results in results.methods.items():
                    if method_name not in method_stats:
                        method_stats[method_name] = {
                            'total_runs': 0,
                            'successful_runs': 0,
                            'total_time_ms': 0.0,
                            'avg_aux': 0.0,
                            'avg_sat_rate': 0.0,
                            'avg_energy': 0.0,
                        }
                    
                    for opt_result in opt_results:
                        method_stats[method_name]['total_runs'] += 1
                        if opt_result.is_satisfied:
                            method_stats[method_name]['successful_runs'] += 1
                        method_stats[method_name]['total_time_ms'] += opt_result.runtime_ms
                        method_stats[method_name]['avg_aux'] += opt_result.n_auxiliary
                        method_stats[method_name]['avg_sat_rate'] += opt_result.sat_rate
                        method_stats[method_name]['avg_energy'] += opt_result.energy
            
            # Compute averages
            for method_name, stats in method_stats.items():
                if stats['total_runs'] > 0:
                    stats['success_rate'] = stats['successful_runs'] / stats['total_runs']
                    stats['avg_time_ms'] = stats['total_time_ms'] / stats['total_runs']
                    stats['avg_aux'] = stats['avg_aux'] / stats['total_runs']
                    stats['avg_sat_rate'] = stats['avg_sat_rate'] / stats['total_runs']
                    stats['avg_energy'] = stats['avg_energy'] / stats['total_runs']
            
            # Print summary table for this k
            print(f"\n{'Method':<25} | {'Success':<10} | {'SAT Rate':<10} | {'Avg Time (ms)':<15} | {'Avg Aux':<10}")
            print("-"*85)
            for method_name, stats in sorted(method_stats.items(), 
                                             key=lambda x: (-x[1].get('avg_sat_rate', 0), 
                                                           -x[1].get('success_rate', 0),
                                                           x[1].get('avg_aux', 0))):
                print(f"{method_name:<25} | {stats.get('success_rate', 0)*100:>6.1f}%     | "
                      f"{stats.get('avg_sat_rate', 0):>8.3f}   | "
                      f"{stats.get('avg_time_ms', 0):>12.2f}   | {stats.get('avg_aux', 0):>8.2f}")
        
        # Overall summary (all k combined)
        print(f"\n{'='*80}")
        print(f"OVERALL SUMMARY (All k combined, {len(all_results)} total instances)")
        print(f"{'='*80}")
        
        method_stats = {}
        for results in all_results:
            for method_name, opt_results in results.methods.items():
                if method_name not in method_stats:
                    method_stats[method_name] = {
                        'total_runs': 0,
                        'successful_runs': 0,
                        'total_time_ms': 0.0,
                        'avg_aux': 0.0,
                        'avg_sat_rate': 0.0,
                        'avg_energy': 0.0,
                    }
                
                for opt_result in opt_results:
                    method_stats[method_name]['total_runs'] += 1
                    if opt_result.is_satisfied:
                        method_stats[method_name]['successful_runs'] += 1
                    method_stats[method_name]['total_time_ms'] += opt_result.runtime_ms
                    method_stats[method_name]['avg_aux'] += opt_result.n_auxiliary
                    method_stats[method_name]['avg_sat_rate'] += opt_result.sat_rate
                    method_stats[method_name]['avg_energy'] += opt_result.energy
        
        # Compute averages
        for method_name, stats in method_stats.items():
            if stats['total_runs'] > 0:
                stats['success_rate'] = stats['successful_runs'] / stats['total_runs']
                stats['avg_time_ms'] = stats['total_time_ms'] / stats['total_runs']
                stats['avg_aux'] = stats['avg_aux'] / stats['total_runs']
                stats['avg_sat_rate'] = stats['avg_sat_rate'] / stats['total_runs']
                stats['avg_energy'] = stats['avg_energy'] / stats['total_runs']
        
        # Save JSON summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_instances': len(all_results),
            'by_method': convert_numpy_types(method_stats)
        }
        
        summary_path = self.output_dir / "summaries" / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        
        # Print overall summary table
        print(f"\n{'Method':<25} | {'Success':<10} | {'SAT Rate':<10} | {'Avg Time (ms)':<15} | {'Avg Aux':<10}")
        print("-"*85)
        for method_name, stats in sorted(method_stats.items(), 
                                         key=lambda x: (-x[1].get('avg_sat_rate', 0), 
                                                       -x[1].get('success_rate', 0),
                                                       x[1].get('avg_aux', 0))):
            print(f"{method_name:<25} | {stats.get('success_rate', 0)*100:>6.1f}%     | "
                  f"{stats.get('avg_sat_rate', 0):>8.3f}   | "
                  f"{stats.get('avg_time_ms', 0):>12.2f}   | {stats.get('avg_aux', 0):>8.2f}")
        
        print(f"\n✓ JSON summary saved to: {summary_path}")
        print(f"✓ Assignments saved to: {self.output_dir / 'assignments'}/")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="k-SAT Benchmark for Quadratization Methods")
    parser.add_argument('--k', type=int, nargs='+', default=[3, 5, 7, 10], help='Clause sizes')
    parser.add_argument('--n-vars', type=int, default=20, help='Number of variables')
    parser.add_argument('--n-instances', type=int, default=5, help='Instances per k')
    parser.add_argument('--output-dir', type=str, default='ksat_benchmark', help='Output directory')
    parser.add_argument('--no-save', action='store_true', help='Do not save files')
    parser.add_argument('--methods', type=str, nargs='+', default=None, help='Specific methods')
    
    args = parser.parse_args()
    
    print("="*80)
    print("k-SAT QUADRATIZATION BENCHMARK SUITE")
    print("="*80)
    print(f"k values: {args.k}")
    print(f"Variables: {args.n_vars}")
    print(f"Instances per k: {args.n_instances}")
    print(f"Output directory: {args.output_dir}")
    print(f"Optimizer: Unified Simulated Annealing (SAME for ALL methods)")
    print("="*80)
    
    benchmark = KSATBenchmark(output_dir=args.output_dir)
    
    if args.methods:
        benchmark.METHODS = [(name, m) for name, m in benchmark.METHODS if name in args.methods]
    
    start_time = time.perf_counter()
    all_results = benchmark.run_full_benchmark(
        k_values=args.k,
        n_vars=args.n_vars,
        n_instances=args.n_instances,
        save_files=not args.no_save
    )
    total_time = time.perf_counter() - start_time
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"Total instances: {len(all_results)}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Results saved to: {args.output_dir}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()