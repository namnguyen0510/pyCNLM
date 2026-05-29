"""
reducer/zero_aux/split_reduction.py
I-E  Split Reduction (Okada, Tanburn, Dattani, 2015)
Summary: Condition on most connected variables. Each split creates sub-problems.
Cost: 0 auxiliary variables, but multiple optimization runs needed.

NOTE: Ground state = min over ALL sub-problems. Test must check all sub-problems.
"""
from ..base import HOBO, QuadResult, QuadratizationMethod
from typing import List, Tuple

class SplitReduction(QuadratizationMethod):
    name           = "split_reduction"
    section        = "I-E"
    description    = "Split reduction by conditioning on variables"
    handles_sign   = "any"
    handles_degree = "any"
    aux_per_term   = "0"
    preserves_spectrum = False

    def __init__(self, max_splits: int = 1):
        self.max_splits = max_splits
        self.sub_problems: List[HOBO] = []

    def apply(self, h: HOBO) -> QuadResult:
        self.sub_problems = []
        var_connectivity = self._compute_connectivity(h)
        
        if not var_connectivity or self.max_splits <= 0:
            self.sub_problems.append(h)
            return QuadResult(
                h,
                method=self.name,
                section=self.section,
                notes="No splits. Ground state = min over all sub-problems."
            )
        
        split_var = max(var_connectivity, key=var_connectivity.get)
        self._split_on_variable(h, split_var)
        
        return QuadResult(
            self.sub_problems[0] if self.sub_problems else h,
            method=self.name,
            section=self.section,
            notes=f"Split on {split_var}. {len(self.sub_problems)} sub-problems. Ground state = min over ALL."
        )

    def _compute_connectivity(self, h: HOBO) -> dict:
        connectivity = {}
        for term in h.terms:
            for var in term:
                connectivity[var] = connectivity.get(var, 0) + len(term)
        return connectivity

    def _split_on_variable(self, h: HOBO, var: int):
        # Case 1: var = 0 (remove all terms containing var)
        h0 = h.copy()
        for term in list(h0.terms.keys()):
            if var in term:
                h0._remove_term(term)
        self.sub_problems.append(h0)
        
        # Case 2: var = 1 (remove var from terms, keep rest)
        h1 = h.copy()
        for term, coeff in list(h1.terms.items()):
            if var in term:
                h1._remove_term(term)
                new_term = frozenset(v for v in term if v != var)
                h1._add_term(new_term, coeff)
        self.sub_problems.append(h1)

    def get_ground_state(self) -> Tuple[dict, float]:
        """Get ground state by minimizing over ALL sub-problems."""
        from itertools import product
        
        min_energy = float('inf')
        best_assign = None
        
        for sub_hobo in self.sub_problems:
            for assignment in product([0, 1], repeat=sub_hobo.n_vars):
                assign_dict = dict(enumerate(assignment))
                energy = sub_hobo.evaluate(assign_dict)
                if energy < min_energy:
                    min_energy = energy
                    best_assign = assign_dict
        
        return best_assign, min_energy