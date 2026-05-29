r"""
reducer/arbitrary/fgbz_positive.py
V-C  FGBZ Reduction for Positive Terms (Fix-Gruber-Boros-Zabih, 2011)
Eq. 154: ∑_H α_H ∏_{j∈H} b_j → ∑_H α_H b_a ∏_{j∈C} b_j + ∑_H α_H(1 - b_a) ∏_{j∈H\C} b_j

NOTE: This is a 'perfect' transformation - minimizes over b_a recovers original.
"""
from ..base import HOBO, QuadResult, QuadratizationMethod
from collections import defaultdict
from itertools import combinations

class FGBZ_Positive(QuadratizationMethod):
    name           = "fgbz_positive"
    section        = "V-C"
    description    = "FGBZ reduction for positive terms"
    handles_sign   = "positive"
    handles_degree = "any"
    aux_per_term   = "1 per application"

    def apply(self, h: HOBO) -> QuadResult:
        # Find positive high-order terms
        positive_terms = [(t, c) for t, c in h.terms.items() if c > 0 and len(t) >= 3]
        
        if not positive_terms:
            return QuadResult(
                h,
                method=self.name,
                section=self.section,
                notes="No positive high-order terms found."
            )
        
        # Find most common variable subset (size >= 2)
        var_pair_counts = defaultdict(int)
        for term, _ in positive_terms:
            for pair in combinations(sorted(term), 2):
                var_pair_counts[pair] += 1
        
        if not var_pair_counts:
            return QuadResult(h, method=self.name, section=self.section)
        
        # Get most common pair
        common_pair = max(var_pair_counts, key=var_pair_counts.get)
        C = frozenset(common_pair)
        
        ba = h._new_aux()
        
        # Apply Eq. 154 to each positive term containing C
        for term, coeff in positive_terms:
            if C.issubset(term):
                # Remove original term
                h._add_term(term, -coeff)
                
                # H \ C (remaining variables)
                H_minus_C = frozenset(v for v in term if v not in C)
                
                # Eq. 154: α_H·b_a·∏_C b_j + α_H·(1 - b_a)·∏_{H\C} b_j
                # = α_H·b_a·∏_C b_j + α_H·∏_{H\C} b_j - α_H·b_a·∏_{H\C} b_j
                
                # α_H·b_a·∏_C b_j
                if len(C) == 1:
                    h._add_term(frozenset({list(C)[0], ba}), coeff)
                else:
                    # Higher order (will be reduced in future iterations)
                    h._add_term(frozenset(C | {ba}), coeff)
                
                # α_H·∏_{H\C} b_j (term without aux)
                if len(H_minus_C) > 0:
                    h._add_term(H_minus_C, coeff)
                
                # -α_H·b_a·∏_{H\C} b_j (negative term, can use NTR methods)
                if len(H_minus_C) == 0:
                    h._add_term(frozenset({ba}), -coeff)
                elif len(H_minus_C) == 1:
                    h._add_term(frozenset({list(H_minus_C)[0], ba}), -coeff)
                else:
                    # Higher order negative term
                    h._add_term(frozenset(H_minus_C | {ba}), -coeff)
        
        return QuadResult(
            h,
            method=self.name,
            section=self.section,
            notes=f"Split on common pair {common_pair}. Creates negative terms for NTR reduction."
        )