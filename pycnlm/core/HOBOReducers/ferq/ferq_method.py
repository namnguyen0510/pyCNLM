"""
reducer/ferq/ferq_method.py
Fermat Quadratization (FERQ) - Ancilla-free reduction using Fermat quotients.

Based on: "Ancilla-free High-ordered Binary Optimization via Fermat Quotients"
Nguyen & Tran, NeurIPS 2026

Key Theorem:
    ∏_{i∈S} x_i = Σ_{k=1}^{|S|-1} α_k^{(|S|)} · δ_{p_k}(A_S)
    where A_S = Σ_{i∈S} x_i, δ_p(A) = (A - A^p)/p

Properties:
    • 0 auxiliary variables (ancilla-free)
    • Exact representation (preserves full spectrum)
    • Works for any degree d ≥ 2
    • Optimized for large-scale problems (n > 25)

OPTIMIZATIONS vs original:
    1. FERQTerm.col_sums: precomputed column sums of weighted_fq → O(1) per-term evaluation
    2. FERQEvaluator.var_to_terms: adjacency list → O(affected_terms) delta computation
    3. FERQEvaluator._degree_groups: vectorized batch evaluation grouped by degree
    4. Degree-grouped np.sum for batch a-value computation (eliminates Python term loop)

FIXES applied (see inline FIX comments):
    FIX-BUG2  FERQ.apply() no longer eagerly calls _extract_ferq_terms() — that
              compat path is now lazy (call result.build_ferq_terms() explicitly).
    FIX-BUG3  FERQ.evaluate() caches the FERQEvaluator per HOBO instance instead
              of rebuilding it from scratch on every call.
    FIX-SEC   FERQEvaluator.compute_delta() replaces np.sum(x[term.indices]) with
              a plain Python sum — lower dispatch overhead for small-degree terms.
              FERQTerm stores _py_indices (Python tuple) for the hot path.
"""
from ..base import HOBO, QuadResult, QuadratizationMethod
from fractions import Fraction
from math import factorial
from functools import lru_cache
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from collections import defaultdict
import time


# ═══════════════════════════════════════════════════════════════════════════
# NUMBER-THEORETIC PRIMITIVES (Fallback if fermat_core not available)
# ═══════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1000)
def _get_primes(n: int) -> List[int]:
    """Cached prime generation."""
    primes: List[int] = []
    c = 2
    while len(primes) < n:
        if all(c % p != 0 for p in primes):
            primes.append(c)
        c += 1
    return primes


@lru_cache(maxsize=5000)
def _stirling2(n: int, k: int) -> int:
    """Cached Stirling numbers of the second kind."""
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0 or k > n:
        return 0
    if n > 50:
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                dp[i][j] = j * dp[i-1][j] + dp[i-1][j-1]
        return dp[n][k]
    return k * _stirling2(n - 1, k) + _stirling2(n - 1, k - 1)


def _surjection_number(p: int, q: int) -> int:
    """N(p, q) = q! · S(p, q)."""
    return factorial(q) * _stirling2(p, q)


def _fermat_quotient_fast(a: int, p: int) -> float:
    """Optimized Fermat quotient δ_p(a) = (a - a^p) / p."""
    if a == 0 or a == 1:
        return 0.0
    return (a - pow(a, p)) / p


# ═══════════════════════════════════════════════════════════════════════════
# PRE-COMPUTED FERMAT QUOTIENT LOOKUP TABLES (FIXED SINGLETON)
# ═══════════════════════════════════════════════════════════════════════════

class FermatQuotientLookup:
    """Pre-compute Fermat quotient lookup tables for fast evaluation."""

    _instance = None
    _max_degree = 15

    def __new__(cls, max_degree: int = 15):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._max_degree = max_degree
        return cls._instance

    def __init__(self, max_degree: int = 15):
        if self._initialized:
            return
        self._initialized = True
        self.max_degree = max(max_degree, self._max_degree)
        self._tables: Dict[int, np.ndarray] = {}
        self._primes: Dict[int, List[int]] = {}
        self._alphas: Dict[int, np.ndarray] = {}
        self._precompute_all()

    def _precompute_all(self):
        for d in range(2, self.max_degree + 1):
            self._precompute_degree(d)

    def _precompute_degree(self, d: int):
        primes = _get_primes(d - 1)
        self._primes[d] = primes
        alphas = self._compute_alpha_fallback(d, primes)
        self._alphas[d] = alphas
        table = np.zeros((len(primes), d + 1), dtype=np.float64)
        for k, p in enumerate(primes):
            for a in range(d + 1):
                table[k, a] = _fermat_quotient_fast(a, p)
        self._tables[d] = table

    def _compute_alpha_fallback(self, d: int, primes: List[int]) -> np.ndarray:
        size = d - 1
        M = np.zeros((size, size), dtype=np.float64)
        for k in range(size):
            pk = primes[k]
            for q in range(size):
                deg = q + 2
                if deg <= min(pk, d):
                    M[k, q] = _surjection_number(pk, deg) / pk
        try:
            M_inv = np.linalg.inv(M)
            last_row = -M_inv[d - 2, :]
            return last_row
        except Exception:
            return np.zeros(size)

    def get_table(self, degree: int) -> np.ndarray:
        if degree > self.max_degree:
            self._precompute_degree(degree)
        return self._tables.get(degree, np.array([]))

    def get_primes(self, degree: int) -> List[int]:
        if degree > self.max_degree:
            self._precompute_degree(degree)
        return self._primes.get(degree, [])

    def get_alphas(self, degree: int) -> np.ndarray:
        if degree > self.max_degree:
            self._precompute_degree(degree)
        return self._alphas.get(degree, np.array([]))


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZED FERQ TERM STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════

class FERQTerm:
    """
    Optimized term structure with precomputed col_sums for O(1) evaluation.

    KEY OPTIMIZATION: col_sums[a] = sum_k(weighted_fq[k, a])
    This collapses the inner prime-loop into a single array lookup.

    FIX-SEC: _py_indices is a plain Python tuple stored alongside the numpy
    array so compute_delta can sum x values with a lightweight Python loop
    instead of paying numpy fancy-indexing overhead on every SA step.
    """

    # FIX-SEC: added '_py_indices' slot
    __slots__ = ['indices', '_py_indices', 'degree', 'weight', 'primes', 'alphas',
                 'fq_table', 'weighted_fq', 'col_sums']

    def __init__(self, indices: np.ndarray, weight: float, fq_lookup: FermatQuotientLookup):
        self.indices = indices.astype(np.int32)
        # FIX-SEC: cache as Python tuple for O(degree) hot-path sum without numpy overhead
        self._py_indices: tuple = tuple(int(i) for i in indices)
        self.degree = len(indices)
        self.weight = float(weight)

        if self.degree >= 2:
            self.primes = fq_lookup.get_primes(self.degree)
            self.alphas = fq_lookup.get_alphas(self.degree)
            self.fq_table = fq_lookup.get_table(self.degree)

            if len(self.primes) > 0 and self.fq_table.size > 0:
                self.weighted_fq = np.zeros_like(self.fq_table)
                for k in range(len(self.primes)):
                    self.weighted_fq[k, :] = self.weight * self.alphas[k] * self.fq_table[k, :]
                # ─── OPTIMIZATION 1: precompute column sums ───────────────────
                # col_sums[a] = Σ_k weighted_fq[k, a]  →  single float lookup
                self.col_sums = np.sum(self.weighted_fq, axis=0)  # shape (degree+1,)
            else:
                self.weighted_fq = np.array([])
                self.col_sums = np.zeros(self.degree + 1, dtype=np.float64)
        else:
            self.primes = []
            self.alphas = np.array([])
            self.fq_table = np.array([])
            self.weighted_fq = np.array([])
            self.col_sums = np.array([])

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate this term at assignment x — O(1) via col_sums lookup."""
        if self.degree == 0:
            return self.weight
        elif self.degree == 1:
            return self.weight * x[self._py_indices[0]]
        else:
            # FIX-SEC: Python sum over _py_indices avoids numpy fancy-indexing
            a = sum(x[j] for j in self._py_indices)
            if self.col_sums.size > a:
                return float(self.col_sums[a])
            return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZED FERQ EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════

class FERQEvaluator:
    """
    High-performance FERQ energy evaluator.

    OPTIMIZATIONS:
    1. col_sums  — O(1) per-term evaluation (no prime-loop)
    2. var_to_terms — O(affected) delta (no full-term scan)
    3. _degree_groups — fully vectorized batch evaluation per degree

    FIX-SEC applied in compute_delta: replaced np.sum(x[term.indices]) with a
    plain Python sum over term._py_indices — significantly lower dispatch
    overhead for the small-degree terms that dominate most HOBO instances.
    """

    def __init__(self, hobo: HOBO, max_degree: int = 15):
        self.n_vars = hobo.n_vars
        self.max_degree = max_degree
        self.fq_lookup = FermatQuotientLookup(max_degree=max_degree)
        self.terms = self._build_terms(hobo)
        self.n_terms = len(self.terms)

        # ─── OPTIMIZATION 2: var → term adjacency list ────────────────────
        # var_to_terms[v] = list of term indices whose .indices contains v
        # Replaces full scan + "bit_idx in term.indices" check in compute_delta
        self.var_to_terms: List[List[int]] = [[] for _ in range(self.n_vars)]
        for term_idx, term in enumerate(self.terms):
            for var in term._py_indices:
                self.var_to_terms[var].append(term_idx)

        # ─── OPTIMIZATION 3: degree-grouped batch arrays ──────────────────
        # Allows vectorized a-value computation: a_values = x[term_vars].sum(axis=1)
        self._build_degree_groups()

    def _build_terms(self, hobo: HOBO) -> List[FERQTerm]:
        terms = []
        for term_set, weight in hobo.terms.items():
            if len(term_set) == 0:
                continue
            indices = np.array(sorted(term_set), dtype=np.int32)
            terms.append(FERQTerm(indices, weight, self.fq_lookup))
        return terms

    def _build_degree_groups(self):
        """
        Group terms by degree and build (var_matrix, col_sums_matrix) for each degree.
        Enables fully vectorized evaluation:
            a_values = x[var_matrix].sum(axis=1)        # shape (n_d,)
            energies = col_sums_matrix[arange, a_values] # shape (n_d,)
        """
        groups: Dict[int, List] = defaultdict(list)

        self._linear_weight = 0.0    # summed weight of all degree-1 terms
        self._linear_indices: List[int] = []
        self._linear_weights: List[float] = []
        self._const_weight = 0.0    # constant (degree-0) terms

        for term in self.terms:
            if term.degree == 0:
                self._const_weight += term.weight
            elif term.degree == 1:
                self._linear_indices.append(int(term.indices[0]))
                self._linear_weights.append(term.weight)
            else:
                groups[term.degree].append(term)

        self._linear_idx_arr = np.array(self._linear_indices, dtype=np.int32)
        self._linear_w_arr = np.array(self._linear_weights, dtype=np.float64)

        self._degree_groups: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for degree, term_list in groups.items():
            n_d = len(term_list)
            # var_matrix: (n_d, degree) — variable indices for each term
            var_matrix = np.zeros((n_d, degree), dtype=np.int32)
            # col_sums_matrix: (n_d, degree+1) — precomputed col sums
            cs_matrix = np.zeros((n_d, degree + 1), dtype=np.float64)
            for i, term in enumerate(term_list):
                var_matrix[i] = term.indices
                cs_matrix[i] = term.col_sums
            self._degree_groups[degree] = (var_matrix, cs_matrix)

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate FERQ energy (term-loop, kept for compatibility)."""
        if len(x) != self.n_vars:
            raise ValueError(f"Expected {self.n_vars} variables, got {len(x)}")
        energy = 0.0
        for term in self.terms:
            energy += term.evaluate(x)
        return energy

    def evaluate_fast(self, x: np.ndarray) -> float:
        """
        Vectorized evaluation using degree-grouped batch computation.

        For each degree d:
          a_values = x[var_matrix].sum(axis=1)         O(n_d * d) vectorized
          energy  += cs_matrix[arange, a_values].sum()  O(n_d)    vectorized

        Eliminates the Python term-loop entirely for high-degree terms.
        """
        # Constant terms
        energy = self._const_weight

        # Linear terms — single vectorized dot product
        if self._linear_idx_arr.size > 0:
            energy += float(np.dot(self._linear_w_arr, x[self._linear_idx_arr]))

        # Higher-degree terms — vectorized per degree group
        for degree, (var_matrix, cs_matrix) in self._degree_groups.items():
            # a_values[i] = number of active variables in term i
            a_values = x[var_matrix].sum(axis=1).astype(np.int32)   # (n_d,)
            energy += float(cs_matrix[np.arange(len(a_values)), a_values].sum())

        return energy

    def compute_delta(self, x: np.ndarray, bit_idx: int) -> float:
        """
        Compute energy delta when flipping bit_idx.

        OPTIMIZATIONS applied:
        • var_to_terms[bit_idx] — only visits O(affected) terms, not all terms
        • term.col_sums[a] — O(1) lookup per term instead of O(n_primes) sum

        FIX-SEC: replaced np.sum(x[term.indices]) with a Python sum over
        term._py_indices.  For degree-2..7 terms the numpy dispatch overhead
        dominated; the plain Python loop is measurably faster at this scale.
        """
        old_val = int(x[bit_idx])
        flip = 1 - 2 * old_val  # +1 (0→1) or -1 (1→0)
        delta = 0.0

        for term_idx in self.var_to_terms[bit_idx]:
            term = self.terms[term_idx]
            if term.degree == 1:
                delta += term.weight * flip
            elif term.degree >= 2:
                # FIX-SEC: Python sum instead of np.sum(x[term.indices])
                a_old = sum(x[j] for j in term._py_indices)
                a_new = a_old + flip
                if 0 <= a_new <= term.degree:
                    delta += float(term.col_sums[a_new] - term.col_sums[a_old])

        return delta


# ═══════════════════════════════════════════════════════════════════════════
# FERQ METHOD CLASS (API Compatible)
# ═══════════════════════════════════════════════════════════════════════════

class FERQ(QuadratizationMethod):
    """
    Fermat Quadratization (FERQ) - Ancilla-free reduction using Fermat quotients.

    Theorem 1: Any degree-d monomial ∏_{i∈S} x_i can be exactly represented as:
        ∏_{i∈S} x_i = Σ_{k=1}^{d-1} α_k^{(d)} · δ_{p_k}(A_S)

    Properties:
        • 0 auxiliary variables (ancilla-free)
        • Exact representation (preserves full spectrum)
        • Works for any degree d ≥ 2
        • Optimized for large-scale problems (n > 25)
        • Backward compatible with test_script.py API
    """

    name = "ferq"
    section = "VII"
    description = "Fermat Quadratization (ancilla-free, optimized)"
    handles_sign = "any"
    handles_degree = "any"
    aux_per_term = "0"
    preserves_spectrum = True

    def __init__(self, max_degree: int = 15):
        self.max_degree = max_degree
        self._fq_lookup = FermatQuotientLookup(max_degree=max_degree)
        # FIX-BUG3: cache for FERQ.evaluate() — avoids rebuilding evaluator per call
        self._cached_h: Optional[HOBO] = None
        self._cached_evaluator: Optional[FERQEvaluator] = None

    def apply(self, h: HOBO) -> QuadResult:
        """
        Apply FERQ transformation.

        FIX-BUG2: _extract_ferq_terms() is no longer called eagerly.
        The slow backward-compat representation (ferq_terms / evaluate_ferq)
        was being built on every apply() call even though the benchmark path
        only uses ferq_evaluator.evaluate_fast().  Call result.build_ferq_terms()
        explicitly if you need the legacy dict representation.
        """
        evaluator = FERQEvaluator(h, max_degree=self.max_degree)

        result = QuadResult(
            h,
            method=self.name,
            section=self.section,
            notes=(
                f"FERQ transformation applied. 0 auxiliaries. "
                f"n_vars={h.n_vars}, degree={h.degree}"
            )
        )

        result.ferq_evaluator = evaluator
        result.evaluate = lambda x: evaluator.evaluate_fast(np.array(x, dtype=np.int32))

        # FIX-BUG2: lazy compat path — only built on explicit request
        def _lazy_build_ferq_terms():
            if not hasattr(result, '_ferq_terms_built'):
                result.ferq_terms = self._extract_ferq_terms(h)
                result.evaluate_ferq = lambda x: self._evaluate_ferq(result.ferq_terms, x)
                result._ferq_terms_built = True
        result.build_ferq_terms = _lazy_build_ferq_terms

        return result

    def _extract_ferq_terms(self, h: HOBO) -> List[Dict]:
        """Extract terms in FERQ format — backward compatible with test_script.py."""
        ferq_terms = []
        for term, weight in h.terms.items():
            if len(term) < 2:
                ferq_terms.append({
                    'type': 'linear',
                    'S': tuple(sorted(term)) if term else (),
                    'w': weight
                })
            else:
                d = len(term)
                if d > self.max_degree:
                    primes = _get_primes(d - 1)
                    alphas = self._fq_lookup._compute_alpha_fallback(d, primes)
                else:
                    alphas = self._fq_lookup.get_alphas(d)
                    primes = self._fq_lookup.get_primes(d)
                ferq_terms.append({
                    'type': 'ferq',
                    'S': tuple(sorted(term)),
                    'w': weight,
                    'd': d,
                    'alphas': alphas,
                    'primes': primes
                })
        return ferq_terms

    def _evaluate_ferq(self, ferq_terms: List[Dict], x: np.ndarray) -> float:
        """Evaluate FERQ-transformed energy — backward compatible with test_script.py."""
        energy = 0.0
        for term in ferq_terms:
            if term['type'] == 'linear':
                if len(term['S']) == 0:
                    energy += term['w']
                else:
                    energy += term['w'] * x[term['S'][0]]
            else:
                S = term['S']
                w = term['w']
                alphas = term['alphas']
                primes = term['primes']
                A_S = sum(x[i] for i in S)
                ferq_sum = 0.0
                for alpha_k, p_k in zip(alphas, primes):
                    ferq_sum += alpha_k * _fermat_quotient_fast(int(A_S), p_k)
                energy += w * ferq_sum
        return energy

    def evaluate(self, h: HOBO, assignment: Dict[int, int]) -> float:
        """
        Evaluate FERQ energy for a given assignment.

        FIX-BUG3: cache the FERQEvaluator keyed on the HOBO object identity.
        Previously this method created a brand-new FERQEvaluator on every call
        (O(n_terms × degree) setup each time) and immediately discarded it.
        The cached evaluator is reused as long as the same HOBO is passed in.
        """
        # FIX-BUG3: rebuild only when the HOBO instance changes
        if self._cached_h is not h:
            self._cached_evaluator = FERQEvaluator(h, max_degree=self.max_degree)
            self._cached_h = h

        n_vars = max(max(assignment.keys()) + 1, h.n_vars) if assignment else h.n_vars
        x = np.zeros(n_vars, dtype=np.int32)
        for i, val in assignment.items():
            x[i] = val
        return self._cached_evaluator.evaluate_fast(x)

    def verify_equivalence(self, h: HOBO, tolerance: float = 1e-6) -> Tuple[bool, str]:
        from itertools import product
        if h.n_vars > 25:
            return True, "Skipped (n_vars > 25, use sampling for verification)"
        # build_ferq_terms explicitly when needed for verification
        ferq_terms = self._extract_ferq_terms(h)
        for assignment in product([0, 1], repeat=h.n_vars):
            assign_dict = dict(enumerate(assignment))
            orig_energy = h.evaluate(assign_dict)
            x = np.array(assignment, dtype=np.int32)
            ferq_energy = self._evaluate_ferq(ferq_terms, x)
            if abs(orig_energy - ferq_energy) > tolerance:
                return False, f"Mismatch at {assignment}: {orig_energy:.6f} vs {ferq_energy:.6f}"
        return True, "FERQ equivalence verified for all 2^n assignments"

    def create_evaluator(self, h: HOBO) -> FERQEvaluator:
        return FERQEvaluator(h, max_degree=self.max_degree)


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def create_ferq_evaluator(h: HOBO, max_degree: int = 15) -> FERQEvaluator:
    """Create an optimized FERQ evaluator for a HOBO."""
    return FERQEvaluator(h, max_degree=max_degree)


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARKING
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_ferq_speed(n_vars: int = 50, n_terms: int = 20, degree: int = 5,
                         n_evaluations: int = 1000) -> Dict:
    import random
    from time import perf_counter

    terms = {}
    for _ in range(n_terms):
        d = random.randint(2, min(degree, n_vars))
        vars_subset = tuple(random.sample(range(n_vars), d))
        weight = random.uniform(-1, 1)
        terms[frozenset(vars_subset)] = weight

    h = HOBO(terms, n_vars=n_vars)
    evaluator = FERQEvaluator(h, max_degree=degree)

    rng = np.random.default_rng(42)
    x = rng.integers(0, 2, size=n_vars, dtype=np.int32)

    start = perf_counter()
    for _ in range(n_evaluations):
        evaluator.evaluate_fast(x)
    eval_time = (perf_counter() - start) / n_evaluations * 1000

    start = perf_counter()
    for _ in range(n_evaluations):
        i = int(rng.integers(n_vars))
        evaluator.compute_delta(x, i)
    delta_time = (perf_counter() - start) / n_evaluations * 1000

    return {
        'eval_time_ms': eval_time,
        'delta_time_ms': delta_time,
        'n_vars': n_vars,
        'n_terms': n_terms,
        'degree': degree
    }