"""
reducer/arbitrary/pairwise_covers.py
V-D  Pairwise Covers (Anthony-Boros-Crama-Gruber, 2017)
Eq. 159-160: Handle sets of monomials with common components

NOTE: Does NOT fully quadratize in one step. Creates terms needing further reduction.
"""
from ..base import HOBO, QuadResult, QuadratizationMethod
from collections import defaultdict
from itertools import combinations

class PairwiseCovers(QuadratizationMethod):
    name           = "pairwise_covers"
    section        = "V-D"
    description    = "Pairwise covers for multiple monomials"
    handles_sign   = "any"
    handles_degree = "any"
    aux_per_term   = "1 per application"
    preserves_spectrum = False  # Ground state only

    def apply(self, h: HOBO) -> QuadResult:
        # Group terms by variable pairs
        pair_terms = defaultdict(list)
        for term, coeff in h.terms.items():
            if len(term) >= 3:
                vars_list = sorted(list(term))
                for i in range(len(vars_list)):
                    for j in range(i + 1, len(vars_list)):
                        pair = (vars_list[i], vars_list[j])
                        pair_terms[pair].append((term, coeff))
        
        if not pair_terms:
            return QuadResult(h, method=self.name, section=self.section)
        
        best_pair = max(pair_terms, key=lambda p: len(pair_terms[p]))
        terms_to_reduce = pair_terms[best_pair]
        
        v1, v2 = best_pair
        C = frozenset({v1, v2})
        ba = h._new_aux()
        
        positive_terms = [(t, c) for t, c in terms_to_reduce if c > 0]
        negative_terms = [(t, c) for t, c in terms_to_reduce if c < 0]
        
        # Eq. 159 for positive terms
        for term, coeff in positive_terms:
            h._add_term(term, -coeff)
            H_minus_C = frozenset(v for v in term if v not in C)
            
            # (∑ α_H)·b_a·∏_C b_j
            h._add_term(frozenset({v1, v2, ba}), coeff)
            
            # α_H·∏_{H\C} b_j
            if len(H_minus_C) > 0:
                h._add_term(H_minus_C, coeff)
            
            # -α_H·b_a·∏_{H\C} b_j (negative)
            if len(H_minus_C) > 0:
                h._add_term(frozenset(H_minus_C | {ba}), -coeff)
            else:
                h._add_term(frozenset({ba}), -coeff)
        
        # Eq. 160 for negative terms
        for term, coeff in negative_terms:
            h._add_term(term, -coeff)
            H_minus_C = frozenset(v for v in term if v not in C)
            
            # α_H·b_a
            h._add_term(frozenset({ba}), coeff)
            
            # -α_H·∏_C b_j·b_a
            h._add_term(frozenset({v1, v2, ba}), -coeff)
            
            # -α_H·∏_{H\C} b_j·b_a
            if len(H_minus_C) > 0:
                h._add_term(frozenset(H_minus_C | {ba}), -coeff)
        
        return QuadResult(
            h,
            method=self.name,
            section=self.section,
            notes=f"Covered pair {best_pair}. Creates terms needing further reduction. Ground state preserved."
        )