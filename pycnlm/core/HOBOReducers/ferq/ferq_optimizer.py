"""
reducer/ferq/ferq_optimizer.py
Optimized FERQ solver with precomputed tables, delta computation, and optional Numba JIT.

OPTIMIZATIONS vs original:
    1. FERQTerm.col_sums: column-summed weighted_fq → O(1) per-term eval/delta
    2. FERQOptimizer.var_to_terms: adjacency list → O(affected) delta, not O(all_terms)
    3. _build_flat_arrays(): flat contiguous arrays for optional Numba JIT SA kernel
    4. Numba JIT SA (when numba installed): entire inner loop runs as native code

FIXES applied (see inline FIX comments):
    FIX-BUG4  Removed redundant `bit_idx not in self.indices` check inside
              FERQTerm.compute_delta().  The var_to_terms adjacency list already
              guarantees that compute_delta is only called for terms that contain
              bit_idx, so the O(degree) tuple scan always returned False — pure
              overhead in the SA inner loop, repeated millions of times.
    FIX-SEC   Replaced np.sum(x[self.term_vars]) with a plain Python sum over
              self._py_indices for small-degree hot-path evaluation.
    FIX-SEC   Emit a RuntimeWarning at import time when numba is absent so the
              performance degradation is visible rather than silent.
"""
import warnings
import numpy as np
from typing import List, Tuple, Dict, Optional
import time

# ── Optional Numba JIT ────────────────────────────────────────────────────
try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    # FIX-SEC: surface the degradation so users can act on it
    warnings.warn(
        "numba is not installed — FERQOptimizer will use the pure-Python SA "
        "fallback, which is 10-50× slower than the Numba JIT kernel.  "
        "Install numba (`pip install numba`) for full performance.",
        RuntimeWarning,
        stacklevel=1,
    )

    # No-op decorator so the decorated function runs as plain Python
    def njit(*args, **kwargs):          # type: ignore[misc]
        def _wrap(fn):
            return fn
        return _wrap

# ── Optional fermat_core ──────────────────────────────────────────────────
try:
    from fermat_core import FermatAnnealingSolver, FermatQuboReduction
    _FERMAT_CORE_AVAILABLE = True
except ImportError:
    _FERMAT_CORE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# NUMBA-JIT SA KERNEL
# Runs the full (reads × sweeps) SA loop as native code when Numba available.
# Falls back to _pure_python_sa when Numba is absent.
# ═══════════════════════════════════════════════════════════════════════════

@njit(cache=True, fastmath=True)
def _sa_numba_kernel(
    n_vars: int,
    num_reads: int,
    num_sweeps: int,
    T_init: float,
    decay: float,
    # Flattened term data
    term_offsets: np.ndarray,     # int32[n_terms+1]  — CSR row pointers
    term_var_flat: np.ndarray,    # int32[Σ degree]   — variable indices
    term_col_sums: np.ndarray,    # float64[n_terms, max_degree+1]
    # Flattened adjacency
    var_term_offsets: np.ndarray, # int32[n_vars+1]
    var_term_flat: np.ndarray,    # int32[Σ adj]
    seed: int,
) -> Tuple[np.ndarray, float]:
    """
    Simulated annealing with delta computation — fully native via Numba.

    Uses a simple 64-bit LCG (Knuth multiplicative) for speed.
    Quality is sufficient for SA; no external RNG dependencies.
    """
    n_terms = term_offsets.shape[0] - 1
    max_a = term_col_sums.shape[1]

    best_x = np.zeros(n_vars, dtype=np.int32)
    best_energy = 1.0e18

    # LCG constants (Knuth)
    A = np.int64(6364136223846793005)
    C = np.int64(1442695040888963407)
    rng = np.int64(seed | 1)  # ensure non-zero

    for _read in range(num_reads):
        # Random initial assignment
        x = np.zeros(n_vars, dtype=np.int32)
        for i in range(n_vars):
            rng = rng * A + C
            x[i] = np.int32((rng >> np.int64(63)) & np.int64(1))

        # Initial energy
        energy = 0.0
        for t in range(n_terms):
            a = 0
            for k in range(term_offsets[t], term_offsets[t + 1]):
                a += x[term_var_flat[k]]
            if a < max_a:
                energy += term_col_sums[t, a]

        local_best_energy = energy
        local_best_x = x.copy()
        T = T_init

        for _sweep in range(num_sweeps):
            # Pick random bit
            rng = rng * A + C
            bit = int(rng >> np.int64(32)) % n_vars
            if bit < 0:
                bit = -bit

            flip = 1 - 2 * x[bit]  # +1 or -1

            # Delta: sum contributions from terms that contain this bit
            delta = 0.0
            for vi in range(var_term_offsets[bit], var_term_offsets[bit + 1]):
                t = var_term_flat[vi]
                a_old = 0
                for k in range(term_offsets[t], term_offsets[t + 1]):
                    a_old += x[term_var_flat[k]]
                a_new = a_old + flip
                if 0 <= a_old < max_a and 0 <= a_new < max_a:
                    delta += term_col_sums[t, a_new] - term_col_sums[t, a_old]

            # Metropolis criterion
            accept = delta < 0.0
            if not accept and T > 1e-12:
                rng = rng * A + C
                u = float(rng >> np.int64(11)) * (1.0 / 9007199254740992.0)
                accept = u < np.exp(-delta / T)

            if accept:
                x[bit] = 1 - x[bit]
                energy += delta
                if energy < local_best_energy:
                    local_best_energy = energy
                    local_best_x = x.copy()

            T *= decay

        if local_best_energy < best_energy:
            best_energy = local_best_energy
            best_x = local_best_x.copy()

    return best_x, best_energy


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZED FERQ TERM
# ═══════════════════════════════════════════════════════════════════════════

class FERQTerm:
    """
    FERQ term with precomputed col_sums for O(1) evaluation and delta.

    col_sums[a] = Σ_k (weight * alpha_k * δ_{p_k}(a))
    This collapses the per-prime sum into a single array lookup.

    FIX-SEC: _py_indices (Python tuple) stored alongside term_vars (numpy array)
    for fast iteration in compute_delta without numpy dispatch overhead.
    """

    # FIX-SEC: added '_py_indices' slot
    __slots__ = ['indices', '_py_indices', 'degree', 'weight', 'n_primes',
                 'weighted_fq', 'col_sums', 'term_vars']

    def __init__(self, indices: Tuple[int, ...], weight: float, degree: int,
                 primes: List[int], alphas: np.ndarray, fq_table: np.ndarray):
        self.indices = indices
        # FIX-SEC: Python tuple for O(degree) hot-path sum without numpy overhead
        self._py_indices: tuple = indices  # already a tuple from set_terms()
        self.degree = degree
        self.term_vars = np.array(indices, dtype=np.int32)
        self.weight = float(weight)
        self.n_primes = len(primes)

        self.weighted_fq = np.zeros((self.n_primes, degree + 1), dtype=np.float64)
        for k in range(self.n_primes):
            self.weighted_fq[k, :] = self.weight * alphas[k] * fq_table[k, :]

        # ─── OPTIMIZATION 1: precompute column sums ───────────────────────
        # col_sums[a] = Σ_k weighted_fq[k, a]  →  O(1) lookup per term
        self.col_sums = np.sum(self.weighted_fq, axis=0)  # shape (degree+1,)

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate at x — O(1) via col_sums."""
        if self.degree == 1:
            return self.weight * x[self._py_indices[0]]
        elif self.degree >= 2:
            # FIX-SEC: Python sum instead of np.sum(x[self.term_vars])
            a = sum(x[j] for j in self._py_indices)
            return float(self.col_sums[a])
        return 0.0

    def compute_delta(self, x: np.ndarray, bit_idx: int, flip: int) -> float:
        """
        Energy delta when flipping bit_idx by flip (+1 or -1).
        Uses col_sums for O(1) lookup instead of summing over primes.

        FIX-BUG4: removed `if bit_idx not in self.indices: return 0.0`.
        That check was an O(degree) tuple scan executed on every SA step for
        every term in var_to_terms[bit_idx].  Because var_to_terms is built
        from the adjacency list, bit_idx is ALWAYS in self.indices at this
        point — the early return was dead code that cost more than it saved.

        FIX-SEC: replaced np.sum(x[self.term_vars]) with a Python sum over
        self._py_indices — avoids numpy fancy-indexing overhead for small degrees.
        """
        if self.degree == 1:
            # var_to_terms guarantees bit_idx == self.indices[0] for degree-1 terms
            return self.weight * flip
        elif self.degree >= 2:
            # FIX-BUG4: removed redundant membership check (always True here)
            # FIX-SEC: Python sum — faster than np.sum for degree ≤ ~8
            a_old = sum(x[j] for j in self._py_indices)
            a_new = a_old + flip
            if 0 <= a_new <= self.degree:
                return float(self.col_sums[a_new] - self.col_sums[a_old])
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZED FERQ OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════

class FERQOptimizer:
    """
    Optimized FERQ solver.

    KEY OPTIMIZATIONS:
    1. col_sums   — O(1) per-term evaluation (no prime-loop at runtime)
    2. var_to_terms — O(affected) delta, not O(all_terms)
    3. Flat CSR arrays — ready for Numba JIT SA kernel
    4. Numba JIT   — entire SA loop as native code (10-50× over pure Python)
    """

    _alpha_cache: Dict[int, Dict] = {}

    @classmethod
    def _get_alpha_data(cls, degree: int) -> Dict:
        if degree in cls._alpha_cache:
            return cls._alpha_cache[degree]

        try:
            from fermat_core import compute_alpha
            alphas, primes = compute_alpha(degree)
        except Exception:
            from math import factorial
            from functools import lru_cache

            @lru_cache(maxsize=None)
            def stirling2(n, k):
                if n == 0 and k == 0:
                    return 1
                if n == 0 or k == 0 or k > n:
                    return 0
                return k * stirling2(n - 1, k) + stirling2(n - 1, k - 1)

            def get_primes(n):
                p, c = [], 2
                while len(p) < n:
                    if all(c % x != 0 for x in p):
                        p.append(c)
                    c += 1
                return p

            primes = get_primes(degree - 1)
            size = degree - 1
            M = np.zeros((size, size), dtype=np.float64)
            for k in range(size):
                pk = primes[k]
                for q in range(size):
                    deg = q + 2
                    if deg <= min(pk, degree):
                        M[k, q] = factorial(deg) * stirling2(pk, deg) / pk
            try:
                last_row = -np.linalg.inv(M)[degree - 2, :]
                alphas = last_row
            except Exception:
                alphas = np.zeros(size)

        fq_table = np.zeros((len(primes), degree + 1), dtype=np.float64)
        for k, p in enumerate(primes):
            for a in range(degree + 1):
                fq_table[k, a] = (a - pow(a, p)) / p if a > 1 else 0.0

        cls._alpha_cache[degree] = {
            'alphas': alphas,
            'primes': primes,
            'fq_table': fq_table,
        }
        return cls._alpha_cache[degree]

    def __init__(self, n_vars: int, num_reads: int = 50, num_sweeps: int = 2000,
                 beta_range: Tuple[float, float] = (0.1, 10.0), seed: int = 42):
        self.n_vars = n_vars
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.beta_range = beta_range
        self.seed = seed
        self.terms: List[FERQTerm] = []

        # Built by set_terms()
        self.var_to_terms: List[List[int]] = [[] for _ in range(n_vars)]
        self._flat_arrays: Optional[Dict] = None

    def set_terms(self, hobo_terms: Dict, max_degree: int = 15):
        """Convert HOBO terms to FERQTerm objects and build auxiliary structures."""
        self.terms = []
        self.var_to_terms = [[] for _ in range(self.n_vars)]

        for term_set, weight in hobo_terms.items():
            if len(term_set) == 0:
                continue
            indices = tuple(sorted(term_set))
            degree = len(indices)
            if degree > max_degree:
                continue

            alpha_data = self._get_alpha_data(degree)
            term = FERQTerm(
                indices=indices,
                weight=weight,
                degree=degree,
                primes=alpha_data['primes'],
                alphas=alpha_data['alphas'],
                fq_table=alpha_data['fq_table'],
            )
            term_idx = len(self.terms)
            self.terms.append(term)

            # ─── OPTIMIZATION 2: adjacency list ───────────────────────────
            for var in indices:
                self.var_to_terms[var].append(term_idx)

        # Build flat CSR arrays for Numba kernel
        self._flat_arrays = self._build_flat_arrays()

    def _build_flat_arrays(self) -> Dict:
        """
        Build flat CSR-style arrays for Numba JIT kernel.

        term_offsets[t]    = start index in term_var_flat for term t
        term_var_flat      = concatenated variable indices for all terms
        term_col_sums[t,a] = precomputed energy contribution at sum=a
        var_term_offsets[v] = start index in var_term_flat for var v
        var_term_flat       = concatenated term indices for each var
        """
        n_terms = len(self.terms)
        if n_terms == 0:
            return {}

        max_degree = max(t.degree for t in self.terms)

        # Term CSR
        term_offsets = np.zeros(n_terms + 1, dtype=np.int32)
        for i, term in enumerate(self.terms):
            term_offsets[i + 1] = term_offsets[i] + term.degree

        term_var_flat = np.zeros(int(term_offsets[-1]), dtype=np.int32)
        for i, term in enumerate(self.terms):
            s, e = term_offsets[i], term_offsets[i + 1]
            term_var_flat[s:e] = term.term_vars

        # Padded col_sums matrix (n_terms × max_degree+1)
        term_col_sums = np.zeros((n_terms, max_degree + 1), dtype=np.float64)
        for i, term in enumerate(self.terms):
            term_col_sums[i, :term.degree + 1] = term.col_sums

        # Variable adjacency CSR
        var_term_offsets = np.zeros(self.n_vars + 1, dtype=np.int32)
        for v, tlist in enumerate(self.var_to_terms):
            var_term_offsets[v + 1] = var_term_offsets[v] + len(tlist)

        var_term_flat = np.zeros(int(var_term_offsets[-1]), dtype=np.int32)
        for v, tlist in enumerate(self.var_to_terms):
            s = var_term_offsets[v]
            for j, t in enumerate(tlist):
                var_term_flat[s + j] = t

        return {
            'term_offsets': term_offsets,
            'term_var_flat': term_var_flat,
            'term_col_sums': term_col_sums,
            'var_term_offsets': var_term_offsets,
            'var_term_flat': var_term_flat,
        }

    # ── energy / delta helpers (used by fermat_core path) ─────────────────

    def _evaluate(self, x: np.ndarray) -> float:
        return sum(term.evaluate(x) for term in self.terms)

    def _compute_delta(self, x: np.ndarray, bit_idx: int, flip: int) -> float:
        """O(affected_terms) delta using adjacency list + col_sums."""
        delta = 0.0
        for term_idx in self.var_to_terms[bit_idx]:
            delta += self.terms[term_idx].compute_delta(x, bit_idx, flip)
        return delta

    # ── SA implementations ────────────────────────────────────────────────

    def _numba_sa(self) -> Tuple[np.ndarray, float]:
        """Numba JIT SA using flat CSR arrays — fastest path."""
        fa = self._flat_arrays
        beta_min, beta_max = self.beta_range
        T_init = 1.0 / beta_min
        T_min  = 1.0 / beta_max
        decay  = (T_min / T_init) ** (1.0 / max(self.num_sweeps - 1, 1))

        x_opt, energy = _sa_numba_kernel(
            self.n_vars, self.num_reads, self.num_sweeps, T_init, decay,
            fa['term_offsets'], fa['term_var_flat'], fa['term_col_sums'],
            fa['var_term_offsets'], fa['var_term_flat'],
            self.seed,
        )
        return x_opt, energy

    def _custom_sa(self) -> Tuple[np.ndarray, float]:
        """Pure-Python / numpy SA with adjacency-based delta — fallback."""
        rng = np.random.default_rng(self.seed)
        beta_min, beta_max = self.beta_range
        T_init = 1.0 / beta_min
        T_min  = 1.0 / beta_max
        decay  = (T_min / T_init) ** (1.0 / max(self.num_sweeps - 1, 1))

        best_x = None
        best_energy = float('inf')

        for _ in range(self.num_reads):
            x = rng.integers(0, 2, size=self.n_vars, dtype=np.int32)
            energy = self._evaluate(x)
            local_best_x = x.copy()
            local_best_energy = energy
            T = T_init

            for _ in range(self.num_sweeps):
                i = int(rng.integers(self.n_vars))
                flip = 1 if x[i] == 0 else -1

                # ─── OPTIMIZATION 2: adjacency delta ──────────────────────
                delta = self._compute_delta(x, i, flip)

                if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-12)):
                    x[i] = 1 - x[i]
                    energy += delta
                    if energy < local_best_energy:
                        local_best_energy = energy
                        local_best_x = x.copy()

                T *= decay

            if local_best_energy < best_energy:
                best_energy = local_best_energy
                best_x = local_best_x

        return best_x, best_energy

    def solve(self) -> Tuple[np.ndarray, float, float]:
        """
        Run optimization and return (x_opt, energy, runtime_ms).

        Priority:
          1. fermat_core (FermatAnnealingSolver) if available
          2. Numba JIT kernel if numba installed
          3. Pure-Python/numpy SA with adjacency delta
        """
        start_time = time.perf_counter()

        if _FERMAT_CORE_AVAILABLE:
            solver = FermatAnnealingSolver(
                n_vars=self.n_vars,
                objective_fn=self._evaluate,
                num_reads=self.num_reads,
                num_sweeps=self.num_sweeps,
                beta_range=self.beta_range,
                seed=self.seed,
            )
            x_opt, energy = solver.solve()
        elif _NUMBA_AVAILABLE and self._flat_arrays:
            x_opt, energy = self._numba_sa()
        else:
            x_opt, energy = self._custom_sa()

        runtime_ms = (time.perf_counter() - start_time) * 1000
        return x_opt.astype(np.int32), float(energy), float(runtime_ms)