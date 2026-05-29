"""
Problem instances and the literal incidence matrix used by the
CNLM-Langevin solver.

Encoding
--------
For a CNF clause C_j = (ℓ_1 ∨ ... ∨ ℓ_{k_j}) where ℓ ∈ {x_i, ¬x_i}, we build
a signed literal incidence matrix L ∈ {-1, 0, +1}^{m × n}:

    L[j, i] = +1   if  x_i  appears positively in clause j
    L[j, i] = -1   if  ¬x_i appears in clause j
    L[j, i] =  0   otherwise.

The continuous clause score is then

    s~_j(z) = Σ_i L[j,i] * σ(z_i) + (|N_j| - 1 + ε)
            = (L σ(z))_j + b_j

where N_j is the set of negatively-occurring variables in clause j
and 0 < ε < 1 is the SDNF margin parameter.  This dichotomy gives:
    s~_j > 0  ⇔  clause j is satisfied (when z is at a Boolean corner).

For MaxSAT we additionally store per-clause weights w_j ≥ 0; for hard
clauses we use a large effective weight (or a separate bookkeeping path
that the solver respects).

The implementation prefers SciPy CSR sparse matrices for the gradient
computation because clause widths are typically tiny (≤ 10) and a dense
m×n matrix is wasteful for industrial instances.  When SciPy is missing
we fall back to a dense ndarray transparently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np

try:
    import scipy.sparse as sp
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    sp = None  # type: ignore
    _HAS_SCIPY = False

from .parser import ParsedCNF, ParsedWCNF


# --------------------------------------------------------------------- core builder
def build_literal_matrix(
    clauses: Sequence[Sequence[int]],
    n_vars: int,
    *,
    use_sparse: Optional[bool] = None,
):
    """
    Construct the signed literal-incidence matrix L (m × n) and the offset
    vector b ∈ R^m so that s~(z) = L σ(z) + b for ε supplied separately.

    Returns
    -------
    L : (m, n) signed matrix (CSR sparse if available, dense otherwise)
    n_neg : (m,) int array, number of negative literals per clause
    width : (m,) int array, clause widths (number of literals)
    """
    m = len(clauses)
    if use_sparse is None:
        use_sparse = _HAS_SCIPY and (n_vars * m > 10_000)
    use_sparse = use_sparse and _HAS_SCIPY

    n_neg = np.zeros(m, dtype=np.int32)
    width = np.zeros(m, dtype=np.int32)

    if use_sparse:
        # build COO triples
        rows: List[int] = []
        cols: List[int] = []
        vals: List[int] = []
        for j, cl in enumerate(clauses):
            width[j] = len(cl)
            for lit in cl:
                v = abs(lit) - 1
                if v < 0 or v >= n_vars:
                    raise ValueError(f"clause {j}: literal {lit} out of range for n_vars={n_vars}")
                rows.append(j)
                cols.append(v)
                if lit > 0:
                    vals.append(1)
                else:
                    vals.append(-1)
                    n_neg[j] += 1
        L = sp.csr_matrix(
            (np.asarray(vals, dtype=np.float32),
             (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
            shape=(m, n_vars),
        )
    else:
        L = np.zeros((m, n_vars), dtype=np.float32)
        for j, cl in enumerate(clauses):
            width[j] = len(cl)
            for lit in cl:
                v = abs(lit) - 1
                if v < 0 or v >= n_vars:
                    raise ValueError(f"clause {j}: literal {lit} out of range for n_vars={n_vars}")
                if lit > 0:
                    L[j, v] = 1.0
                else:
                    L[j, v] = -1.0
                    n_neg[j] += 1

    return L, n_neg, width


def matvec_L(L, x):
    """L @ x for either dense or sparse L (returns ndarray)."""
    if _HAS_SCIPY and sp.issparse(L):
        return L @ x
    return L @ x


def matvec_LT(L, y):
    """L.T @ y for either dense or sparse L (returns ndarray)."""
    if _HAS_SCIPY and sp.issparse(L):
        return L.T @ y
    return L.T @ y


# --------------------------------------------------------------------- evaluation helpers
def evaluate_clauses_bool(clauses: Sequence[Sequence[int]], x: np.ndarray) -> np.ndarray:
    """
    Boolean evaluation of each CNF clause under assignment x ∈ {0,1}^n.
    Returns a (m,) bool ndarray.
    """
    n = x.shape[0]
    sat = np.zeros(len(clauses), dtype=bool)
    for j, cl in enumerate(clauses):
        ok = False
        for lit in cl:
            v = abs(lit) - 1
            if v >= n:
                continue
            xv = bool(x[v])
            if (lit > 0 and xv) or (lit < 0 and not xv):
                ok = True
                break
        sat[j] = ok
    return sat


def evaluate_clauses_bool_vectorized(L, n_neg, x_bool: np.ndarray) -> np.ndarray:
    """
    Vectorised Boolean evaluation.  Uses the identity:
        clause j satisfied ⇔ Σ_{i∈P_j} x_i + Σ_{i∈N_j} (1-x_i) ≥ 1
                          ⇔ (L x)_j + |N_j| ≥ 1
    Works on x ∈ {0,1}.
    """
    xs = x_bool.astype(np.float32, copy=False)
    s_vals = matvec_L(L, xs) + n_neg
    return s_vals >= 1.0 - 1e-9


# ====================================================================== SAT
@dataclass
class SATInstance:
    """A SAT instance encoded as a CNF formula."""
    n_vars: int
    clauses: List[List[int]]
    name: str = ""
    L: object = field(default=None, repr=False)
    n_neg: np.ndarray = field(default=None, repr=False)
    width: np.ndarray = field(default=None, repr=False)

    @classmethod
    def from_parsed(cls, parsed: ParsedCNF, name: str = "") -> "SATInstance":
        inst = cls(n_vars=parsed.n_vars, clauses=parsed.clauses, name=name)
        inst._build()
        return inst

    @classmethod
    def from_clauses(cls, n_vars: int, clauses: List[List[int]], name: str = "") -> "SATInstance":
        inst = cls(n_vars=n_vars, clauses=clauses, name=name)
        inst._build()
        return inst

    def _build(self) -> None:
        self.L, self.n_neg, self.width = build_literal_matrix(self.clauses, self.n_vars)

    @property
    def n_clauses(self) -> int:
        return len(self.clauses)

    def evaluate(self, x: np.ndarray) -> Tuple[bool, int, np.ndarray]:
        """Return (is_SAT, n_satisfied, sat_mask)."""
        sat = evaluate_clauses_bool_vectorized(self.L, self.n_neg, x)
        n_sat = int(sat.sum())
        return n_sat == self.n_clauses, n_sat, sat


# ====================================================================== MaxSAT
@dataclass
class MaxSATInstance:
    """A weighted CNF instance with explicit hard/soft separation."""
    n_vars: int
    clauses: List[List[int]]
    weights: np.ndarray            # (m,)
    is_hard: np.ndarray            # (m,) bool
    top: float
    name: str = ""
    L: object = field(default=None, repr=False)
    n_neg: np.ndarray = field(default=None, repr=False)
    width: np.ndarray = field(default=None, repr=False)
    new_format: bool = False

    @classmethod
    def from_parsed(cls, parsed: ParsedWCNF, name: str = "") -> "MaxSATInstance":
        inst = cls(
            n_vars=parsed.n_vars,
            clauses=parsed.clauses,
            weights=np.asarray(parsed.weights, dtype=np.float64),
            is_hard=np.asarray(parsed.is_hard, dtype=bool),
            top=float(parsed.top),
            name=name,
            new_format=parsed.new_format,
        )
        inst._build()
        return inst

    def _build(self) -> None:
        self.L, self.n_neg, self.width = build_literal_matrix(self.clauses, self.n_vars)

    @property
    def n_clauses(self) -> int:
        return len(self.clauses)

    @property
    def n_hard(self) -> int:
        return int(self.is_hard.sum())

    @property
    def n_soft(self) -> int:
        return int((~self.is_hard).sum())

    @property
    def total_soft_weight(self) -> float:
        if self.n_soft == 0:
            return 0.0
        w = self.weights[~self.is_hard]
        finite = w[np.isfinite(w)]
        return float(finite.sum())

    def effective_weights(self, hard_scale: float = 1e3) -> np.ndarray:
        """
        Convert (possibly-infinite) hard weights into finite gradient
        weights, scaled relative to the largest soft weight.
        """
        w = self.weights.astype(np.float64).copy()
        soft_mask = ~self.is_hard
        soft_max = float(w[soft_mask].max()) if soft_mask.any() else 1.0
        if soft_max <= 0:
            soft_max = 1.0
        hard_w = max(hard_scale * soft_max, hard_scale)
        w[self.is_hard] = hard_w
        # safety: replace any residual non-finite values
        w[~np.isfinite(w)] = hard_w
        return w

    def evaluate(self, x: np.ndarray):
        """
        Returns dict with: hard_sat (bool), n_hard_sat, n_soft_sat,
        n_unsat_soft, soft_weight_sat, soft_weight_unsat (cost), total.
        """
        sat_mask = evaluate_clauses_bool_vectorized(self.L, self.n_neg, x)
        hard_mask = self.is_hard
        soft_mask = ~hard_mask
        n_hard_sat = int(sat_mask[hard_mask].sum())
        n_soft_sat = int(sat_mask[soft_mask].sum())
        n_unsat_soft = int((~sat_mask & soft_mask).sum())

        if soft_mask.any():
            w_soft = self.weights[soft_mask].astype(np.float64)
            soft_w_sat = float(w_soft[sat_mask[soft_mask]].sum()) if n_soft_sat else 0.0
            soft_w_unsat = float(w_soft[~sat_mask[soft_mask]].sum()) if n_unsat_soft else 0.0
        else:
            soft_w_sat = soft_w_unsat = 0.0

        return {
            "is_SAT_hard": bool(sat_mask[hard_mask].all()) if hard_mask.any() else True,
            "n_hard_total": int(hard_mask.sum()),
            "n_hard_sat": n_hard_sat,
            "n_hard_unsat": int(hard_mask.sum()) - n_hard_sat,
            "n_soft_total": int(soft_mask.sum()),
            "n_soft_sat": n_soft_sat,
            "n_soft_unsat": n_unsat_soft,
            "soft_weight_satisfied": soft_w_sat,
            "cost": soft_w_unsat,                             # MaxSAT cost (sum of unsatisfied soft weights)
            "n_total_sat": int(sat_mask.sum()),
            "n_total": int(sat_mask.size),
            "sat_mask": sat_mask,
        }
