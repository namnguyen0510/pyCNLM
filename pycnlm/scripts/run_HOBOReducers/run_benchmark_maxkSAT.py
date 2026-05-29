# ═══════════════════════════════════════════════════════════════════════════
# Max-k-SAT QUADRATIZATION BENCHMARK (Jupyter Notebook Version)
# ═══════════════════════════════════════════════════════════════════════════
# REVISED FOR Max-k-SAT:
#   - Loads .wcnf files from MaxkSAT_benchmark folder (recursive)
#   - Weighted HOBO objective: minimize sum_c w_c * I[clause_c unsatisfied]
#   - Proper Max-k-SAT metrics: weight_satisfied, weighted_sat_rate,
#     approx_ratio, n_clauses_satisfied, is_optimal
#   - All reduction methods preserved unchanged
#   - Ground truth via brute-force (n_vars <= 20): optimal_weight known
#   - Approximation ratio = weight_satisfied / optimal_weight
# ═══════════════════════════════════════════════════════════════════════════

import os
import sys
import json
import csv
import time
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
    class HOBO: pass
    class QuadResult: pass
    class DeducReduc: pass

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
class MaxKSATInstance:
    """
    Represents a weighted Max-k-SAT instance parsed from a .wcnf file.

    Objective: find an assignment that MAXIMIZES the total weight of
    satisfied clauses (equivalently, minimizes weight of unsatisfied clauses).

    Attributes
    ----------
    n_vars        : number of boolean variables
    n_clauses     : number of clauses (length of `clauses` list)
    k             : maximum clause length across all clauses
    clauses       : list of clauses; each clause is a tuple of (var_idx, is_negated) pairs
    weights       : per-clause weights  (len == n_clauses)
    total_weight  : sum of all clause weights
    top_weight    : hard-clause threshold from 'p wcnf' header (0.0 if absent / pure Max-SAT)
    alpha         : clause-to-variable ratio  n_clauses / n_vars
    source_file   : path relative to the benchmark root folder
    """
    n_vars: int
    n_clauses: int
    k: int
    clauses: List[Tuple[Tuple[int, bool], ...]]
    weights: List[float]
    total_weight: float
    top_weight: float
    alpha: float
    source_file: str = ""

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def compute_weight_satisfied(self, assignment: List[int]) -> float:
        """
        Returns the total weight of clauses satisfied by `assignment`.

        Parameters
        ----------
        assignment : list of 0/1 values, length >= n_vars (extras ignored)

        Returns
        -------
        float  – sum of weights of all satisfied clauses
        """
        if len(assignment) < self.n_vars:
            assignment = list(assignment) + [0] * (self.n_vars - len(assignment))
        weight_sat = 0.0
        for clause, w in zip(self.clauses, self.weights):
            for var_idx, is_negated in clause:
                if var_idx < len(assignment):
                    val = assignment[var_idx]
                    # literal is True ↔ clause is satisfied by this literal
                    if (not is_negated and val == 1) or (is_negated and val == 0):
                        weight_sat += w
                        break
        return weight_sat

    def compute_n_clauses_satisfied(self, assignment: List[int]) -> int:
        """
        Returns the number of clauses satisfied by `assignment` (unweighted count).
        """
        if len(assignment) < self.n_vars:
            assignment = list(assignment) + [0] * (self.n_vars - len(assignment))
        n_sat = 0
        for clause in self.clauses:
            for var_idx, is_negated in clause:
                if var_idx < len(assignment):
                    val = assignment[var_idx]
                    if (not is_negated and val == 1) or (is_negated and val == 0):
                        n_sat += 1
                        break
        return n_sat

    def weighted_sat_rate(self, assignment: List[int]) -> float:
        """
        Returns weight_satisfied / total_weight  ∈ [0, 1].
        0.0 when total_weight == 0.
        """
        if self.total_weight == 0.0:
            return 0.0
        return self.compute_weight_satisfied(assignment) / self.total_weight

    def to_wcnf(self) -> str:
        """Serialises the instance back to WCNF text format."""
        lines = [
            f"c Max-k-SAT instance (k={self.k}, n={self.n_vars}, m={self.n_clauses})",
            f"c alpha = {self.alpha:.3f}",
            f"c total_weight = {self.total_weight:.3f}",
            f"c Generated: {datetime.now().isoformat()}",
            f"c Source: {self.source_file}",
            f"p wcnf {self.n_vars} {self.n_clauses} {int(self.top_weight) if self.top_weight else 'hard'}",
        ]
        for clause, w in zip(self.clauses, self.weights):
            lit_str = " ".join(
                f"-{var+1}" if negated else f"{var+1}"
                for var, negated in clause
            )
            lines.append(f"{int(w) if w == int(w) else w} {lit_str} 0")
        return "\n".join(lines)

    def save_wcnf(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(self.to_wcnf())


@dataclass
class OptimizationResult:
    """
    Holds the result of one method applied to one Max-k-SAT instance.

    Key Max-k-SAT metrics
    ---------------------
    weight_satisfied  : total weight of clauses satisfied by the found assignment
    total_weight      : total weight of all clauses in the instance
    weighted_sat_rate : weight_satisfied / total_weight  ∈ [0, 1]
    n_clauses_satisfied: number of clauses satisfied (unweighted count)
    approx_ratio      : weight_satisfied / optimal_weight
                        (1.0 = found optimal; < 1.0 = suboptimal)
                        set to -1.0 when optimal_weight is not known
    is_optimal        : True iff approx_ratio >= 1.0 - 1e-6
    energy            : raw HOBO/QUBO energy =
                        sum_c w_c * I[clause_c unsatisfied]
                        = total_weight - weight_satisfied
    """
    method_name: str
    optimizer_name: str
    n_vars_original: int
    n_vars_quadratized: int
    n_auxiliary: int
    weight_satisfied: float
    total_weight: float
    weighted_sat_rate: float
    n_clauses_satisfied: int
    approx_ratio: float          # -1.0 if optimal not known
    is_optimal: bool             # approx_ratio >= 1.0 - 1e-6
    energy: float                # raw HOBO energy (= total_weight - weight_satisfied)
    runtime_ms: float
    assignment: Optional[List[int]] = None
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            'method_name':         str(self.method_name),
            'optimizer_name':      str(self.optimizer_name),
            'n_vars_original':     int(self.n_vars_original),
            'n_vars_quadratized':  int(self.n_vars_quadratized),
            'n_auxiliary':         int(self.n_auxiliary),
            'weight_satisfied':    float(self.weight_satisfied),
            'total_weight':        float(self.total_weight),
            'weighted_sat_rate':   float(self.weighted_sat_rate),
            'n_clauses_satisfied': int(self.n_clauses_satisfied),
            'approx_ratio':        float(self.approx_ratio),
            'is_optimal':          bool(self.is_optimal),
            'energy':              float(self.energy),
            'runtime_ms':          float(self.runtime_ms),
            'assignment':          [int(x) for x in self.assignment] if self.assignment else None,
            'notes':               str(self.notes),
        }


@dataclass
class BenchmarkResults:
    """
    Aggregates all method results for one Max-k-SAT instance.

    Ground truth fields
    -------------------
    optimal_weight       : maximum achievable weight_satisfied  (from brute-force)
    optimal_weight_known : True when brute-force was feasible (n_vars <= 20)
    optimal_assignment   : the brute-force best assignment (None if not computed)
    """
    instance_id: str
    k: int
    n_vars: int
    n_clauses: int
    alpha: float
    total_weight: float
    optimal_weight: float
    optimal_weight_known: bool
    optimal_assignment: Optional[List[int]] = None
    source_file: str = ""
    methods: Dict[str, List[OptimizationResult]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'instance_id':          str(self.instance_id),
            'k':                    int(self.k),
            'n_vars':               int(self.n_vars),
            'n_clauses':            int(self.n_clauses),
            'alpha':                float(self.alpha),
            'total_weight':         float(self.total_weight),
            'optimal_weight':       float(self.optimal_weight),
            'optimal_weight_known': bool(self.optimal_weight_known),
            'optimal_assignment':   (
                [int(x) for x in self.optimal_assignment]
                if self.optimal_assignment else None
            ),
            'source_file':  self.source_file,
            'methods':      {k: [r.to_dict() for r in v] for k, v in self.methods.items()},
            'metadata':     convert_numpy_types(self.metadata),
        }


# ═══════════════════════════════════════════════════════════════════════════
# WCNF PARSER
# ═══════════════════════════════════════════════════════════════════════════

class WCNFParser:
    """
    Parses Weighted CNF (.wcnf) files conforming to the Max-SAT Evaluation format.

    File format (per line):
      c  <comment>
      p wcnf <n_vars> <n_clauses> [<top_weight>]
      <weight> <lit1> <lit2> ... <litk> 0

    Notes
    -----
    - Lines beginning with 'c' are comments (skipped).
    - The 'p' header specifies n_vars and n_clauses.
      The optional <top_weight> field marks the hard-clause threshold in
      Partial Max-SAT instances (clauses with weight >= top_weight are hard).
      For pure Max-SAT benchmarks the field may be absent or very large.
    - Each data line begins with the clause weight (integer or float),
      followed by DIMACS-style signed literals, terminated by 0.
    - Positive literal  l > 0  corresponds to variable (l - 1) not negated.
    - Negative literal  l < 0  corresponds to variable (|l| - 1) negated.
    """

    @staticmethod
    def parse_wcnf_file(filepath: str) -> MaxKSATInstance:
        n_vars: int = 0
        n_clauses_header: int = 0
        top_weight: float = 0.0
        clauses: List[Tuple[Tuple[int, bool], ...]] = []
        weights: List[float] = []

        with open(filepath, 'r') as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith('c'):
                    continue

                # ── header ──────────────────────────────────────────────
                if line.startswith('p'):
                    parts = line.split()
                    if len(parts) >= 4 and parts[1] in ('wcnf', 'WCNF'):
                        n_vars          = int(parts[2])
                        n_clauses_header = int(parts[3])
                        if len(parts) >= 5:
                            top_weight = float(parts[4])
                    continue

                # ── data line: weight lit1 lit2 ... litk 0 ──────────────
                parts = line.split()
                if not parts:
                    continue
                try:
                    w_raw = parts[0]
                    # Support integer or float weights
                    w = float(w_raw)
                    literals = list(map(int, parts[1:]))
                    if literals and literals[-1] == 0:
                        literals = literals[:-1]
                    clause: List[Tuple[int, bool]] = []
                    for lit in literals:
                        if lit == 0:
                            continue
                        if lit > 0:
                            clause.append((lit - 1, False))   # positive literal
                        else:
                            clause.append((abs(lit) - 1, True))  # negated literal
                    if clause:
                        clauses.append(tuple(clause))
                        weights.append(w)
                except (ValueError, IndexError):
                    # Skip malformed lines silently
                    continue

        k = max(len(c) for c in clauses) if clauses else 0
        total_weight = float(sum(weights))
        alpha = len(clauses) / n_vars if n_vars > 0 else 0.0

        return MaxKSATInstance(
            n_vars=n_vars,
            n_clauses=len(clauses),
            k=k,
            clauses=clauses,
            weights=weights,
            total_weight=total_weight,
            top_weight=top_weight,
            alpha=alpha,
            source_file=os.path.basename(filepath),
        )

    @staticmethod
    def load_wcnf_folder(folder_path: str) -> List[MaxKSATInstance]:
        """
        Recursively loads all .wcnf / .WCNF files under `folder_path`.

        Returns a list of MaxKSATInstance objects sorted by filename.
        """
        instances: List[MaxKSATInstance] = []
        folder = Path(folder_path)

        if not folder.exists():
            print(f"⚠ Folder {folder_path} does not exist!")
            return instances

        wcnf_files = sorted(
            list(folder.rglob("*.wcnf")) + list(folder.rglob("*.WCNF"))
        )

        if not wcnf_files:
            print(f"⚠ No .wcnf files found in {folder_path}")
            return instances

        print(f"✓ Found {len(wcnf_files)} WCNF files in {folder_path} (recursive)")

        for wcnf_file in tqdm(wcnf_files, desc="Loading WCNF files", unit="file"):
            try:
                instance = WCNFParser.parse_wcnf_file(str(wcnf_file))
                instance.source_file = str(wcnf_file.relative_to(folder))
                instances.append(instance)
            except Exception as e:
                print(f"  ✗ Failed to load {wcnf_file.name}: {str(e)}")

        return instances


# ═══════════════════════════════════════════════════════════════════════════
# Max-k-SAT → HOBO CONVERTER
# ═══════════════════════════════════════════════════════════════════════════

class MaxKSATToHOBO:
    """
    Converts a Max-k-SAT instance to a HOBO minimization problem.

    Objective (to MINIMIZE):
        E(x) = sum_{c} w_c * I[clause_c is UNSATISFIED by x]

    At the optimum, E(x*) = total_weight - optimal_weight, where
    optimal_weight is the maximum total weight of satisfiable clauses.

    Relationship to Max-k-SAT solution:
        weight_satisfied(x) = total_weight - E(x)
        => maximising weight_satisfied ≡ minimising E(x).

    Per-clause unsatisfied indicator:
        A clause (l_1 ∨ l_2 ∨ … ∨ l_k) is unsatisfied iff ALL its
        literals are False.  In 0/1 variables:
          - positive literal  x_i  is False when x_i = 0
            → false indicator = (1 − x_i)
          - negated literal   ¬x_i is False when x_i = 1
            → false indicator = x_i
        So I[clause unsatisfied] = ∏_i false_indicator(l_i),
        which is a multilinear polynomial of degree k in {x_i}.
    """

    @staticmethod
    def convert(instance: MaxKSATInstance) -> HOBO:
        """
        Builds the HOBO whose minimum encodes the Max-k-SAT optimum.

        Returns
        -------
        HOBO  – weighted sum of per-clause unsatisfied indicators
        """
        terms: Dict[frozenset, float] = {}
        for clause, weight in zip(instance.clauses, instance.weights):
            for term, coeff in MaxKSATToHOBO._expand_clause(clause).items():
                terms[term] = terms.get(term, 0.0) + weight * coeff
        # Drop numerical zero coefficients
        terms = {t: c for t, c in terms.items() if abs(c) > 1e-12}
        return HOBO(terms, n_vars=instance.n_vars)

    @staticmethod
    def _expand_clause(
        clause: Tuple[Tuple[int, bool], ...]
    ) -> Dict[frozenset, float]:
        """
        Expands ∏_{l in clause} false_indicator(l) into a multilinear polynomial.

        Returns a dict mapping frozenset(variable_indices) → coefficient,
        representing the polynomial in the HOBO term basis.

        Starting polynomial: 1 (constant)
        At each literal (var_idx, is_negated):
          - is_negated=True  (literal = ¬x_i): multiply current poly by x_i
          - is_negated=False (literal =  x_i): multiply current poly by (1 − x_i)
        """
        poly: Dict[frozenset, float] = {frozenset(): 1.0}

        for var_idx, is_negated in clause:
            new_poly: Dict[frozenset, float] = {}

            if is_negated:
                # Multiply by x_{var_idx}
                for term, coeff in poly.items():
                    new_term = frozenset(set(term) | {var_idx})
                    new_poly[new_term] = new_poly.get(new_term, 0.0) + coeff

            else:
                # Multiply by (1 − x_{var_idx})
                for term, coeff in poly.items():
                    # Constant part: coeff * 1
                    new_poly[term] = new_poly.get(term, 0.0) + coeff
                    # Linear part:  coeff * (−x_{var_idx})
                    new_term = frozenset(set(term) | {var_idx})
                    new_poly[new_term] = new_poly.get(new_term, 0.0) - coeff

            poly = new_poly

        return poly


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED OPTIMIZER  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════

class UnifiedSimulatedAnnealing:
    def __init__(self, num_reads: int = 50, num_sweeps: int = 2000,
                 beta_range: Tuple[float, float] = (0.1, 10.0), seed: int = 42):
        self.num_reads   = num_reads
        self.num_sweeps  = num_sweeps
        self.beta_range  = beta_range
        self.seed        = seed
        self.rng         = np.random.default_rng(seed)

    def optimize(
        self,
        energy_fn:  Callable,
        n_vars:     int,
        delta_fn:   Optional[Callable] = None,
    ) -> Tuple[List[int], float, float]:
        start_time = time.perf_counter()
        beta_min, beta_max = self.beta_range
        T_init = 1.0 / beta_min
        T_min  = 1.0 / beta_max
        decay  = (T_min / T_init) ** (1.0 / max(self.num_sweeps - 1, 1))

        best_energy:    float         = float('inf')
        best_assignment: Optional[List[int]] = None

        if delta_fn is not None:
            # Fast path: incremental delta evaluation
            for _ in range(self.num_reads):
                x      = self.rng.integers(0, 2, size=n_vars, dtype=np.int32)
                energy = float(energy_fn(x))
                local_best_energy = energy
                local_best_x      = x.copy()
                T = T_init

                for _ in range(self.num_sweeps):
                    i     = int(self.rng.integers(n_vars))
                    delta = float(delta_fn(x, i))

                    if delta < 0.0 or self.rng.random() < np.exp(-delta / max(T, 1e-12)):
                        x[i]   ^= 1
                        energy += delta
                        if energy < local_best_energy:
                            local_best_energy = energy
                            local_best_x      = x.copy()

                    T *= decay

                if local_best_energy < best_energy:
                    best_energy      = local_best_energy
                    best_assignment  = local_best_x.tolist()

        else:
            # Standard path: full energy recomputation
            for _ in range(self.num_reads):
                x      = self.rng.integers(0, 2, size=n_vars).tolist()
                energy = energy_fn(x)
                local_best_energy = energy
                local_best_x      = x.copy()
                T = T_init

                for _ in range(self.num_sweeps):
                    i    = int(self.rng.integers(n_vars))
                    x[i] = 1 - x[i]

                    new_energy = energy_fn(x)
                    delta      = new_energy - energy

                    if delta < 0 or self.rng.random() < np.exp(-delta / max(T, 1e-12)):
                        energy = new_energy
                        if energy < local_best_energy:
                            local_best_energy = energy
                            local_best_x      = x.copy()
                    else:
                        x[i] = 1 - x[i]

                    T *= decay

                if local_best_energy < best_energy:
                    best_energy     = local_best_energy
                    best_assignment = local_best_x

        runtime_ms = (time.perf_counter() - start_time) * 1000
        return best_assignment or [0] * n_vars, float(best_energy), float(runtime_ms)


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════════════════

class MaxKSATBenchmarkFromWCNF:
    """
    Full Max-k-SAT benchmark runner.

    Workflow (per instance)
    -----------------------
    1. Convert MaxKSATInstance → HOBO  (weighted unsatisfied-clause polynomial)
    2. Brute-force ground truth when n_vars <= 20:
         optimal_weight = max_{x} weight_satisfied(x)
    3. For each reduction method:
         a. Apply method to HOBO → QUBO / reduced HOBO
         b. Optimise with UnifiedSimulatedAnnealing
         c. Evaluate assignment on the ORIGINAL (unquadratized) instance:
              weight_satisfied, weighted_sat_rate, n_clauses_satisfied
         d. Compute approx_ratio = weight_satisfied / optimal_weight
            (only when optimal_weight is known)
    """

    METHODS = [
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

    def __init__(self, output_dir: str = "maxksat_benchmark"):
        self.output_dir = Path(output_dir)
        for sub in ["wcnf", "hobo", "results", "summaries", "assignments", "csv"]:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

        self.optimizer = UnifiedSimulatedAnnealing(
            num_reads=50, num_sweeps=2000,
            beta_range=(0.1, 10.0), seed=42,
        )

    # ------------------------------------------------------------------
    # Instance-level runner
    # ------------------------------------------------------------------

    def run_instance(
        self,
        instance:    MaxKSATInstance,
        instance_id: str,
        save_files:  bool = True,
        show_progress: bool = True,
    ) -> BenchmarkResults:

        results = BenchmarkResults(
            instance_id=instance_id,
            k=instance.k,
            n_vars=instance.n_vars,
            n_clauses=instance.n_clauses,
            alpha=instance.alpha,
            total_weight=instance.total_weight,
            optimal_weight=instance.total_weight,   # upper bound; refined below
            optimal_weight_known=False,
            optimal_assignment=None,
            source_file=instance.source_file,
            metadata={'timestamp': datetime.now().isoformat()},
        )

        # ── [1/3] Convert to HOBO ────────────────────────────────────────
        t0   = time.perf_counter()
        hobo = MaxKSATToHOBO.convert(instance)
        hobo_time = (time.perf_counter() - t0) * 1000
        if save_files:
            self._save_hobo(hobo, instance_id)

        # ── [2/3] Ground truth via brute-force (feasible ≤ 20 vars) ─────
        #
        # The HOBO energy E(x) = total_weight - weight_satisfied(x)
        # so the minimum HOBO energy corresponds to maximum weight_satisfied.
        # We compute optimal_weight = total_weight - min_energy_hobo.
        if instance.n_vars <= 20:
            min_energy      = float('inf')
            best_assignment: Optional[List[int]] = None

            for bits in product([0, 1], repeat=instance.n_vars):
                e = hobo.evaluate(dict(enumerate(bits)))
                if e < min_energy:
                    min_energy      = e
                    best_assignment = list(bits)

            optimal_weight = instance.total_weight - float(min_energy)
            results.optimal_weight       = float(optimal_weight)
            results.optimal_weight_known = True
            results.optimal_assignment   = [int(x) for x in best_assignment] if best_assignment else None

            if save_files and best_assignment is not None:
                w_sat = instance.compute_weight_satisfied(best_assignment)
                self._save_assignment(
                    instance_id, "ground_truth",
                    best_assignment, float(min_energy),
                    w_sat, instance.total_weight,
                    instance.compute_n_clauses_satisfied(best_assignment),
                )
        else:
            # Optimal unknown: set optimal_weight = total_weight as fallback
            results.optimal_weight       = instance.total_weight
            results.optimal_weight_known = False

        optimal_weight = results.optimal_weight   # alias for use below

        # ── [3/3] Run all quadratization methods ─────────────────────────
        total_methods = len(self.METHODS)

        if show_progress:
            method_iter = tqdm(
                self.METHODS,
                desc=f"Instance {instance_id[:30]}",
                unit="method", leave=False,
            )
        else:
            method_iter = self.METHODS

        for method_idx, (method_name, method) in enumerate(method_iter, 1):
            if show_progress:
                print(f"    ➤ [{method_idx}/{total_methods}] Running Method: {method_name}")

            method_results: List[OptimizationResult] = []

            try:
                t0          = time.perf_counter()
                quad_result = method(hobo.copy())
                quad_time   = (time.perf_counter() - t0) * 1000

                # ── Set up energy / delta functions ──────────────────
                if method_name == "FERQ" and hasattr(quad_result, 'ferq_evaluator'):
                    evaluator = quad_result.ferq_evaluator

                    def energy_fn(x, _ev=evaluator):
                        if not isinstance(x, np.ndarray):
                            x = np.asarray(x, dtype=np.int32)
                        return float(_ev.evaluate_fast(x))

                    def delta_fn(x, bit_idx, _ev=evaluator):
                        return _ev.compute_delta(x, bit_idx)

                    n_vars_opt = instance.n_vars
                    assignment, raw_energy, opt_time = self.optimizer.optimize(
                        energy_fn, n_vars_opt, delta_fn=delta_fn
                    )
                else:
                    qubo = quad_result.qubo

                    def energy_fn(asgn, _q=qubo):
                        return float(_q.evaluate(dict(enumerate(asgn))))

                    n_vars_opt = instance.n_vars + quad_result.n_aux
                    assignment, raw_energy, opt_time = self.optimizer.optimize(
                        energy_fn, n_vars_opt
                    )

                # ── Evaluate on the original (unquadratized) instance ─
                orig_assignment = (
                    assignment[:instance.n_vars] if assignment else [0] * instance.n_vars
                )

                weight_sat   = instance.compute_weight_satisfied(orig_assignment)
                n_clauses_sat = instance.compute_n_clauses_satisfied(orig_assignment)
                wsat_rate    = (
                    weight_sat / instance.total_weight
                    if instance.total_weight > 0.0 else 0.0
                )

                # Approximation ratio: weight_sat / optimal_weight
                # − When optimal_weight_known=True  : exact ratio in [0, 1]
                # − When optimal_weight_known=False  : ratio relative to total_weight
                #   (upper bound on optimal) — may slightly underestimate true ratio;
                #   flagged by optimal_weight_known=False in BenchmarkResults.
                if optimal_weight > 1e-12:
                    approx_ratio = float(weight_sat / optimal_weight)
                else:
                    approx_ratio = 1.0 if weight_sat < 1e-12 else -1.0

                is_optimal = bool(approx_ratio >= 1.0 - 1e-6)

                opt_result = OptimizationResult(
                    method_name=method_name,
                    optimizer_name="unified_sa",
                    n_vars_original=int(instance.n_vars),
                    n_vars_quadratized=int(n_vars_opt),
                    n_auxiliary=int(quad_result.n_aux),
                    weight_satisfied=float(weight_sat),
                    total_weight=float(instance.total_weight),
                    weighted_sat_rate=float(wsat_rate),
                    n_clauses_satisfied=int(n_clauses_sat),
                    approx_ratio=float(approx_ratio),
                    is_optimal=is_optimal,
                    energy=float(raw_energy),
                    runtime_ms=float(quad_time + opt_time),
                    assignment=[int(v) for v in orig_assignment] if orig_assignment else None,
                )
                method_results.append(opt_result)

                if save_files and orig_assignment:
                    self._save_assignment(
                        instance_id, method_name,
                        orig_assignment, float(raw_energy),
                        float(weight_sat), float(instance.total_weight),
                        int(n_clauses_sat),
                    )

                results.methods[method_name] = method_results

            except Exception as e:
                print(f"      ✗ Error in {method_name}: {str(e)}")
                results.methods[method_name] = []

        if save_files:
            self._save_results(results, instance_id)

        return results

    # ------------------------------------------------------------------
    # File I/O helpers
    # ------------------------------------------------------------------

    def _save_hobo(self, hobo: HOBO, instance_id: str):
        data = {
            'n_vars':  int(hobo.n_vars),
            'degree':  int(hobo.degree),
            'n_terms': int(len(hobo.terms)),
            'terms': [
                {'variables': sorted(list(t)), 'coefficient': float(c)}
                for t, c in hobo.terms.items()
            ],
        }
        with open(self.output_dir / "hobo" / f"{instance_id}.json", 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

    def _save_assignment(
        self,
        instance_id: str,
        method_name: str,
        assignment:  List[int],
        energy:      float,
        weight_satisfied: float,
        total_weight:     float,
        n_clauses_satisfied: int,
    ):
        data = {
            'instance_id':        instance_id,
            'method':             method_name,
            'assignment':         [int(x) for x in assignment],
            'energy':             float(energy),
            'weight_satisfied':   float(weight_satisfied),
            'total_weight':       float(total_weight),
            'weighted_sat_rate':  float(weight_satisfied / total_weight) if total_weight > 0 else 0.0,
            'n_clauses_satisfied': int(n_clauses_satisfied),
            'timestamp':          datetime.now().isoformat(),
        }
        path = self.output_dir / "assignments" / f"{instance_id}_{method_name}.json"
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

    def _save_results(self, results: BenchmarkResults, instance_id: str):
        with open(self.output_dir / "results" / f"{instance_id}.json", 'w') as f:
            json.dump(results.to_dict(), f, indent=2, cls=NumpyEncoder)

    # ------------------------------------------------------------------
    # CSV summary
    # ------------------------------------------------------------------

    def _save_csv_summary(self, all_results: List[BenchmarkResults]):
        """
        Per-(method, k) aggregate statistics over all instances.

        Metrics tracked per row:
          approx_ratio, weighted_sat_rate, n_clauses_satisfied,
          weight_satisfied, runtime_ms, n_auxiliary, is_optimal
        """
        data_by_method_k: Dict = defaultdict(lambda: defaultdict(list))

        for results in all_results:
            k = results.k
            for method_name, opt_results in results.methods.items():
                for opt_result in opt_results:
                    bucket = data_by_method_k[(method_name, k)]
                    bucket['approx_ratio'].append(opt_result.approx_ratio)
                    bucket['weighted_sat_rate'].append(opt_result.weighted_sat_rate)
                    bucket['n_clauses_satisfied'].append(opt_result.n_clauses_satisfied)
                    bucket['weight_satisfied'].append(opt_result.weight_satisfied)
                    bucket['runtime_ms'].append(opt_result.runtime_ms)
                    bucket['n_auxiliary'].append(opt_result.n_auxiliary)
                    bucket['is_optimal'].append(1 if opt_result.is_optimal else 0)

        csv_rows = []
        for (method_name, k), metrics in sorted(data_by_method_k.items()):
            for metric_name, values in metrics.items():
                if values:
                    csv_rows.append({
                        'method':    method_name,
                        'k':         k,
                        'metric':    metric_name,
                        'mean':      float(np.mean(values)),
                        'std':       float(np.std(values)) if len(values) > 1 else 0.0,
                        'n_samples': len(values),
                    })

        csv_path = (
            self.output_dir / "csv" /
            f"summary_by_k_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['method', 'k', 'metric', 'mean', 'std', 'n_samples']
            )
            writer.writeheader()
            writer.writerows(csv_rows)
        return csv_path

    # ------------------------------------------------------------------
    # Full benchmark entry point
    # ------------------------------------------------------------------

    def run_full_benchmark(
        self,
        wcnf_folder:   str,
        save_files:    bool = True,
        show_progress: bool = True,
    ) -> List[BenchmarkResults]:

        all_results: List[BenchmarkResults] = []
        print(f"\n{'#'*80}")
        print(f"# Loading WCNF instances from: {wcnf_folder}")
        print(f"{'#'*80}")

        instances = WCNFParser.load_wcnf_folder(wcnf_folder)
        if not instances:
            print("⚠ No instances loaded, exiting...")
            return all_results

        instances_by_k: Dict[int, List[MaxKSATInstance]] = defaultdict(list)
        for inst in instances:
            instances_by_k[inst.k].append(inst)

        total_instances = len(instances)
        overall_pbar = (
            tqdm(total=total_instances, desc="Overall Progress", unit="instance")
            if show_progress else None
        )

        for k in sorted(instances_by_k.keys()):
            print(f"\n{'#'*80}")
            print(f"# Max-k-SAT Benchmark: k={k} ({len(instances_by_k[k])} instances)")
            print(f"{'#'*80}")

            for instance in instances_by_k[k]:
                safe_name = (
                    instance.source_file
                    .replace('/', '_').replace('\\', '_')
                    .replace('.wcnf', '').replace('.WCNF', '')
                )
                instance_id = f"k{k}_{safe_name}"

                if save_files:
                    instance.save_wcnf(
                        self.output_dir / "wcnf" / f"{instance_id}.wcnf"
                    )

                print(f"\n>> Processing Instance: {instance_id}")
                results = self.run_instance(
                    instance, instance_id, save_files, show_progress
                )
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

    # ------------------------------------------------------------------
    # Summary / reporting
    # ------------------------------------------------------------------

    def _generate_summary(self, all_results: List[BenchmarkResults]):
        print(f"\n{'='*80}")
        print("GENERATING SUMMARY (Stratified by k)")
        print(f"{'='*80}")

        results_by_k: Dict[int, List[BenchmarkResults]] = defaultdict(list)
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

    def _print_method_table(
        self,
        results_list: List[BenchmarkResults],
        save_json:    bool = False,
    ):
        """
        Prints a ranked table of methods by Max-k-SAT performance metrics.

        Columns
        -------
        Approx Ratio   : mean approximation ratio = weight_satisfied / optimal_weight
                         (shows how close the optimizer got to the optimal solution)
        Wt-SAT Rate    : mean weighted satisfaction rate = weight_satisfied / total_weight
        Clauses Sat    : mean number of clauses satisfied (unweighted count)
        Optimal%       : percentage of runs that found an optimal solution
        Avg Time (ms)  : mean total runtime per run (quadratization + optimization)
        Avg Aux        : mean number of auxiliary variables introduced

        Rows are sorted by (approx_ratio DESC, weighted_sat_rate DESC, n_auxiliary ASC).
        """
        method_stats: Dict[str, Dict] = {}

        for results in results_list:
            for method_name, opt_results in results.methods.items():
                if method_name not in method_stats:
                    method_stats[method_name] = {
                        'total_runs':        0,
                        'optimal_runs':      0,
                        'total_time_ms':     0.0,
                        'sum_aux':           0.0,
                        'sum_approx_ratio':  0.0,
                        'sum_wsat_rate':     0.0,
                        'sum_clauses_sat':   0.0,
                        'sum_weight_sat':    0.0,
                    }
                for opt_result in opt_results:
                    s = method_stats[method_name]
                    s['total_runs']       += 1
                    if opt_result.is_optimal:
                        s['optimal_runs'] += 1
                    s['total_time_ms']    += opt_result.runtime_ms
                    s['sum_aux']          += opt_result.n_auxiliary
                    # approx_ratio may be -1 when optimal unknown; exclude from mean
                    if opt_result.approx_ratio >= 0.0:
                        s['sum_approx_ratio'] += opt_result.approx_ratio
                    s['sum_wsat_rate']    += opt_result.weighted_sat_rate
                    s['sum_clauses_sat']  += opt_result.n_clauses_satisfied
                    s['sum_weight_sat']   += opt_result.weight_satisfied

        # Compute averages
        for name, s in method_stats.items():
            n = s['total_runs']
            if n > 0:
                s['optimal_rate']     = s['optimal_runs']   / n
                s['avg_time_ms']      = s['total_time_ms']  / n
                s['avg_aux']          = s['sum_aux']         / n
                s['avg_approx_ratio'] = s['sum_approx_ratio'] / n
                s['avg_wsat_rate']    = s['sum_wsat_rate']   / n
                s['avg_clauses_sat']  = s['sum_clauses_sat'] / n
                s['avg_weight_sat']   = s['sum_weight_sat']  / n

        # Print table header
        header = (
            f"\n{'Method':<25} | {'Approx Ratio':<14} | {'Wt-SAT Rate':<13} | "
            f"{'Clauses Sat':<13} | {'Optimal%':<10} | "
            f"{'Avg Time (ms)':<15} | {'Avg Aux':<8}"
        )
        print(header)
        print("-" * 110)

        # Sort: approx_ratio DESC, wsat_rate DESC, n_aux ASC
        for name, s in sorted(
            method_stats.items(),
            key=lambda x: (
                -x[1].get('avg_approx_ratio', 0.0),
                -x[1].get('avg_wsat_rate', 0.0),
                 x[1].get('avg_aux', 0.0),
            ),
        ):
            approx_str = (
                f"{s.get('avg_approx_ratio', 0.0):>10.4f}"
                if s.get('avg_approx_ratio', 0.0) >= 0.0
                else "      N/A "
            )
            print(
                f"{name:<25} | {approx_str}     | "
                f"{s.get('avg_wsat_rate', 0.0):>10.4f}    | "
                f"{s.get('avg_clauses_sat', 0.0):>10.2f}    | "
                f"{s.get('optimal_rate', 0.0)*100:>7.1f}%    | "
                f"{s.get('avg_time_ms', 0.0):>12.2f}   | "
                f"{s.get('avg_aux', 0.0):>7.2f}"
            )

        if save_json:
            summary = {
                'timestamp':        datetime.now().isoformat(),
                'total_instances':  len(results_list),
                'by_method':        convert_numpy_types(method_stats),
            }
            summary_path = (
                self.output_dir / "summaries" /
                f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, cls=NumpyEncoder)
            print(f"\n✓ JSON summary saved to: {summary_path}")


# ═══════════════════════════════════════════════════════════════════════════
# RUN BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════

# ── Configuration ──────────────────────────────────────────────────────────
WCNF_DIR    = 'MaxkSAT-HAMLIB'   # folder containing .wcnf files
OUTPUT_DIR  = 'result_maxksat_benchmark_HAMLIB'
SAVE_FILES  = True
SHOW_PROGRESS = True        # Enable tqdm progress bars AND explicit print progression
SELECTED_METHODS = None     # None = all methods; or e.g. ['FERQ', 'NTR_KZFD']

print("=" * 80)
print("Max-k-SAT QUADRATIZATION BENCHMARK SUITE (Jupyter Notebook Version)")
print("=" * 80)
print(f"WCNF directory:   {WCNF_DIR} (Recursive)")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Optimizer:        Unified SA — same algorithm for ALL methods")
print(f"Progress bars:    {'Enabled' if SHOW_PROGRESS else 'Disabled'}")
print("=" * 80)
print()
print("Metrics (Max-k-SAT):")
print("  approx_ratio      = weight_satisfied / optimal_weight")
print("                      (1.0 = optimal found; <1.0 = suboptimal)")
print("  weighted_sat_rate = weight_satisfied / total_weight")
print("  n_clauses_sat     = number of clauses satisfied (unweighted)")
print("  optimal%          = % runs where approx_ratio >= 1 - 1e-6")
print("  HOBO energy       = total_weight - weight_satisfied")
print("                      (minimised by the optimizer)")
print("=" * 80)

# ── Initialise and run ─────────────────────────────────────────────────────
benchmark = MaxKSATBenchmarkFromWCNF(output_dir=OUTPUT_DIR)

if SELECTED_METHODS:
    benchmark.METHODS = [
        (n, m) for n, m in benchmark.METHODS if n in SELECTED_METHODS
    ]
    print(f"Selected methods: {SELECTED_METHODS}")

t0 = time.perf_counter()
all_results = benchmark.run_full_benchmark(
    wcnf_folder=WCNF_DIR,
    save_files=SAVE_FILES,
    show_progress=SHOW_PROGRESS,
)
total_time = time.perf_counter() - t0

print(f"\n{'='*80}")
print(f"BENCHMARK COMPLETE")
print(f"{'='*80}")
print(f"Total instances: {len(all_results)}")
print(f"Total time:      {total_time/60:.2f} minutes")
print(f"Results saved to: {OUTPUT_DIR}/")
print(f"{'='*80}")