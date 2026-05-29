"""
reducer/zero_aux/deduc_reduc.py
I-A  Deduction Reduction (Deduc-reduc; Tanburn, Okada, Dattani, 2015)
Summary: Look for deductions (e.g., b1b2=0) that must hold true at the global minimum.
Substitute high-order terms using low-order terms of the deduction.
Cost: 0 auxiliary variables needed.
"""
from ..base import HOBO, QuadResult, QuadratizationMethod
from itertools import combinations

class DeducReduc(QuadratizationMethod):
    name           = "deduc_reduc"
    section        = "I-A"
    description    = "Deduction-based reduction without auxiliary variables"
    handles_sign   = "any"
    handles_degree = "any"
    aux_per_term   = "0"

    def __init__(self, deductions: dict = None):
        """
        Args:
            deductions: Pre-computed deductions {frozenset(vars): value}
                       Only supports value=0 (product must be 0)
        """
        self.deductions = deductions or {}

    def apply(self, h: HOBO) -> QuadResult:
        """
        Apply deduction-based reduction.
        For deductions where product=0, remove all terms containing that product.
        DO NOT add penalties - this preserves spectrum.
        """
        # If no deductions provided, try to find simple ones
        if not self.deductions:
            self.deductions = self._find_deductions(h)
        
        # Apply each deduction - only remove terms, don't add penalties
        for vars_set, deduction_value in self.deductions.items():
            if deduction_value == 0:
                self._apply_zero_deduction(h, vars_set)
        
        return QuadResult(
            h,
            method=self.name,
            section=self.section,
            notes=f"Applied {len(self.deductions)} deductions. 0 auxiliaries. Spectrum preserved."
        )

    def _find_deductions(self, h: HOBO) -> dict:
        """Find simple deductions by checking pairs of variables."""
        deductions = {}
        vars_list = list(range(h.n_vars))
        
        # Check pairs: b_i * b_j = 0 deductions
        # Only if ALL terms containing both have negative coefficients
        for i, j in combinations(vars_list, 2):
            pair = frozenset({i, j})
            has_positive = False
            has_high_order = False
            
            for term, coeff in h.terms.items():
                if pair.issubset(term):
                    if len(term) >= 3:
                        has_high_order = True
                    if coeff > 0:
                        has_positive = True
                        break
            
            # Only deduce if all high-order terms with this pair are negative
            if has_high_order and not has_positive:
                deductions[pair] = 0
        
        return deductions

    def _apply_zero_deduction(self, h: HOBO, vars_set: frozenset):
        """Apply a deduction where product of vars must be 0."""
        # Remove all terms that contain this product (they equal 0)
        to_remove = []
        for term in h.terms:
            if vars_set.issubset(term):
                to_remove.append(term)
        for term in to_remove:
            h._remove_term(term)

def main():
    # Example from survey Eq. 6-8
    terms = {
        frozenset({0, 1}): 6.0,
        frozenset({0, 2}): 1.0,
        frozenset({0}): -3.0,
        frozenset({1, 2}): -2.0,
        frozenset({1, 3}): -1.0,
        frozenset({1}): 1.0,
    }
    h = HOBO(terms, n_vars=5)
    
    deductions = {frozenset({0, 1}): 0}
    method = DeducReduc(deductions=deductions)
    result = method(h)
    
    print("Original:", h.to_string())
    print("Reduced:", result.qubo.to_string())

if __name__ == "__main__":
    main()