"""
reducer/base.py
Core data structures shared by all quadratization methods.
"""
from __future__ import annotations
import abc
from typing import Dict, FrozenSet, List, Optional, Tuple, Set

# ── type aliases ──────────────────────────────────────────────────────────────
Term = FrozenSet[int]       # monomial: frozenset of variable indices
Poly = Dict[Term, float]    # sparse polynomial: term → coefficient
# ═════════════════════════════════════════════════════════════════════════════

class HOBO:
    """
    Sparse pseudo-Boolean polynomial (Higher-Order Binary Optimization):
        E(x) = Σ_{S ⊆ [n], 1 ≤ |S| ≤ d}  w_S · Π_{i∈S} x_i

    Stored as  { frozenset(variable_indices) : float_coefficient }.
    frozenset()  (empty set) represents the constant term.

    Original variables: indices  0 … n_vars-1.
    Auxiliary variables: indices n_vars … _aux_counter-1  (allocated on demand).
    """

    def __init__(self, terms: Poly, n_vars: Optional[int] = None):
        self.terms: Poly = {k: float(v) for k, v in terms.items() if v != 0}
        all_idx = {i for t in self.terms for i in t}
        self.n_vars: int = (
            n_vars if n_vars is not None
            else (max(all_idx) + 1 if all_idx else 0)
        )
        self._aux_counter: int = self.n_vars  # next free auxiliary index

    # ── properties ─────────────────────────────────────────────────────────

    @property
    def degree(self) -> int:
        """Maximum monomial degree present."""
        return max((len(t) for t in self.terms), default=0)

    @property
    def is_quadratic(self) -> bool:
        return self.degree <= 2

    @property
    def n_aux(self) -> int:
        """Number of auxiliary variables introduced so far."""
        return self._aux_counter - self.n_vars

    @property
    def variables(self) -> frozenset:
        return frozenset(i for t in self.terms for i in t)

    @property
    def original_vars(self) -> frozenset:
        """Original variable indices (0 to n_vars-1)."""
        return frozenset(range(self.n_vars))

    @property
    def auxiliary_vars(self) -> frozenset:
        """Auxiliary variable indices."""
        return frozenset(range(self.n_vars, self._aux_counter))

    # ── evaluation ──────────────────────────────────────────────────────────

    def evaluate(self, assignment: Dict[int, int]) -> float:
        """
        Evaluate E(x) for a binary assignment {var_index: 0_or_1}.
        Absent variables are treated as 0.
        """
        total = 0.0
        for term, coeff in self.terms.items():
            val = coeff
            for i in term:
                val *= assignment.get(i, 0)
            total += val
        return total

    def evaluate_all(self) -> Dict[Tuple[int, ...], float]:
        """
        Evaluate E(x) for ALL 2^n original variable assignments.
        Returns dict mapping (x0, x1, ..., xn-1) → energy.
        """
        from itertools import product
        results = {}
        for assignment in product([0, 1], repeat=self.n_vars):
            assign_dict = dict(enumerate(assignment))
            energy = self.evaluate(assign_dict)
            results[assignment] = energy
        return results

    # ── mutation helpers (used internally by reducers) ──────────────────────

    def _new_aux(self) -> int:
        """Allocate and return the index of a fresh auxiliary variable."""
        idx = self._aux_counter
        self._aux_counter += 1
        return idx

    def _new_aux_batch(self, count: int) -> List[int]:
        """Allocate multiple auxiliary variables at once."""
        indices = list(range(self._aux_counter, self._aux_counter + count))
        self._aux_counter += count
        return indices

    def _add_term(self, key: Term, coeff: float) -> None:
        """
        Add `coeff` to the coefficient of monomial `key`.
        Automatically removes terms whose coefficient becomes zero.
        """
        prev = self.terms.get(key, 0.0)
        new_val = prev + coeff
        if abs(new_val) < 1e-12:
            self.terms.pop(key, None)
        else:
            self.terms[key] = new_val

    def _remove_term(self, key: Term) -> float:
        """Remove term and return its coefficient (0 if absent)."""
        return self.terms.pop(key, 0.0)

    def copy(self) -> HOBO:
        """Deep copy (terms dict + counter)."""
        h = HOBO({k: v for k, v in self.terms.items()}, n_vars=self.n_vars)
        h._aux_counter = self._aux_counter
        return h

    def get_terms_by_degree(self, degree: int) -> List[Tuple[Term, float]]:
        """Get all terms of a specific degree."""
        return [(t, c) for t, c in self.terms.items() if len(t) == degree]

    def get_high_order_terms(self, min_degree: int = 3) -> List[Tuple[Term, float]]:
        """Get all terms with degree >= min_degree."""
        return [(t, c) for t, c in self.terms.items() if len(t) >= min_degree]

    # ── display ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        lines = [
            f"HOBO(n_orig={self.n_vars}, degree={self.degree}, "
            f"n_aux={self.n_aux}, n_terms={len(self.terms)})"
        ]
        for t in sorted(self.terms, key=lambda s: (len(s), sorted(s))):
            c = self.terms[t]
            label = "·".join(f"x{i}" for i in sorted(t)) if t else "1"
            lines.append(f"  {c:+g} · {label}")
        return "\n".join(lines)

    def to_string(self) -> str:
        """Human-readable string representation."""
        parts = []
        for t in sorted(self.terms, key=lambda s: (len(s), sorted(s))):
            c = self.terms[t]
            if t:
                var_part = "·".join(f"b{i}" for i in sorted(t))
            else:
                var_part = "1"
            if c >= 0:
                parts.append(f"+ {c:g}{var_part}")
            else:
                parts.append(f"- {abs(c):g}{var_part}")
        return " ".join(parts).lstrip("+ ")

# ═════════════════════════════════════════════════════════════════════════════

class QuadResult:
    """
    Output of a quadratization method.
    """

    def __init__(self, qubo: HOBO, method: str,
                 section: str = "", notes: str = ""):
        self.qubo    = qubo
        self.method  = method
        self.section = section
        self.notes   = notes

    @property
    def n_aux(self) -> int:
        return self.qubo.n_aux

    @property
    def aux_vars(self) -> List[int]:
        return list(range(self.qubo.n_vars, self.qubo._aux_counter))

    def verify_spectrum(self, original: HOBO, tolerance: float = 1e-6) -> Tuple[bool, str]:
        """
        Verify that the quadratized function preserves the energy spectrum
        for all original variable assignments (minimizing over auxiliaries).
        
        Returns: (success, message)
        """
        from itertools import product
        
        orig_vars = list(range(original.n_vars))
        aux_vars = self.aux_vars
        
        for orig_assignment in product([0, 1], repeat=len(orig_vars)):
            orig_dict = dict(zip(orig_vars, orig_assignment))
            orig_energy = original.evaluate(orig_dict)
            
            # Minimize over auxiliaries
            min_quad_energy = float('inf')
            if not aux_vars:
                min_quad_energy = self.qubo.evaluate(orig_dict)
            else:
                for aux_assignment in product([0, 1], repeat=len(aux_vars)):
                    full_assign = orig_dict.copy()
                    full_assign.update(dict(zip(aux_vars, aux_assignment)))
                    val = self.qubo.evaluate(full_assign)
                    if val < min_quad_energy:
                        min_quad_energy = val
            
            if abs(orig_energy - min_quad_energy) > tolerance:
                return False, f"Mismatch at {orig_assignment}: {orig_energy} vs {min_quad_energy}"
        
        return True, "Spectrum verified successfully"

    def __repr__(self) -> str:
        return (
            f"QuadResult(method={self.method!r}, section={self.section!r}, "
            f"n_aux={self.n_aux}, degree={self.qubo.degree})\n"
            + repr(self.qubo)
        )

# ═════════════════════════════════════════════════════════════════════════════

class QuadratizationMethod(abc.ABC):
    """
    Abstract base class for all quadratization methods.
    """

    name           : str = ""
    section        : str = ""
    description    : str = ""
    handles_sign   : str = "any"     # 'negative' | 'positive' | 'any'
    handles_degree : str = "any"     # 'cubic' | 'any'
    aux_per_term   : str = ""
    requires_symmetric : bool = False
    preserves_spectrum : bool = True 

    def __call__(self, h: HOBO) -> QuadResult:
        return self.apply(h.copy())

    @abc.abstractmethod
    def apply(self, h: HOBO) -> QuadResult:
        """
        Quadratize h in-place (caller passes a copy).
        Must reduce all applicable higher-order terms so that
        h.is_quadratic == True on return (for terms within scope).
        """
        ...

    def can_handle(self, h: HOBO) -> bool:
        """Check if this method can handle the given HOBO."""
        if self.requires_symmetric and not self._is_symmetric(h):
            return False
        
        for term, coeff in h.terms.items():
            if len(term) < 3:
                continue
            if self.handles_sign == "negative" and coeff >= 0:
                continue
            if self.handles_sign == "positive" and coeff <= 0:
                continue
            if self.handles_degree == "cubic" and len(term) != 3:
                continue
            return True
        return False

    def _is_symmetric(self, h: HOBO) -> bool:
        """Check if function is symmetric (invariant under variable permutation)."""
        # Simplified check - full implementation would test all permutations
        return False  # Override in specific methods

    def __repr__(self) -> str:
        return (f"<{self.__class__.__name__} "
                f"section={self.section!r} aux={self.aux_per_term!r}>")