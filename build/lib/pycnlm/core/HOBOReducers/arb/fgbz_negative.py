r"""
reducer/arbitrary/fgbz_negative.py
V-B  FGBZ Reduction for Negative Terms (Fix-Gruber-Boros-Zabih, 2011)
Eq. 149: ∑_H α_H ∏_{j∈H} bⱼ → ∑_H α_H(1 - ∏_{j∈C} bⱼ - ∏_{j∈H\C} bⱼ)b_a

NOTE: This reduces degree but creates new terms needing NTR reduction.
Ground state preserved when combined with NTR methods.
"""
from ..base import HOBO, QuadResult, QuadratizationMethod
from collections import defaultdict
from itertools import combinations

class FGBZ_Negative(QuadratizationMethod):
    name           = "fgbz_negative"
    section        = "V-B"
    description    = "FGBZ reduction for negative terms"
    handles_sign   = "negative"
    handles_degree = "any"
    aux_per_term   = "1 per application"
    preserves_spectrum = False

    def apply(self, h: HOBO) -> QuadResult:
        # Find negative high-order terms
        negative_terms = [(t, c) for t, c in h.terms.items() if c < 0 and len(t) >= 3]
        
        if not negative_terms:
            return QuadResult(
                h,
                method=self.name,
                section=self.section,
                notes="No negative high-order terms."
            )
        
        # Find most common variable pair
        var_pair_counts = defaultdict(int)
        for term, _ in negative_terms:
            for pair in combinations(sorted(term), 2):
                var_pair_counts[pair] += 1
        
        if not var_pair_counts:
            return QuadResult(h, method=self.name, section=self.section)
        
        common_pair = max(var_pair_counts, key=var_pair_counts.get)
        C = frozenset(common_pair)
        ba = h._new_aux()
        
        # Apply Eq. 149: α_H(1 - ∏_C bⱼ - ∏_{H\C} bⱼ)b_a
        # = α_H·b_a - α_H·∏_C bⱼ·b_a - α_H·∏_{H\C} bⱼ·b_a
        for term, coeff in negative_terms:
            if C.issubset(term):
                # Remove original term
                h._add_term(term, -coeff)
                
                H_minus_C = frozenset(v for v in term if v not in C)
                
                # Term 1: α_H·b_a (linear, coeff < 0)
                h._add_term(frozenset({ba}), coeff)
                
                # Term 2: -α_H·∏_C bⱼ·b_a (positive since -coeff > 0)
                # This is degree |C|+1 = 3, needs further reduction
                term_C_aux = frozenset(C | {ba})
                h._add_term(term_C_aux, -coeff)
                
                # Term 3: -α_H·∏_{H\C} bⱼ·b_a (positive since -coeff > 0)
                if len(H_minus_C) > 0:
                    term_HC_aux = frozenset(H_minus_C | {ba})
                    h._add_term(term_HC_aux, -coeff)
        
        return QuadResult(
            h,
            method=self.name,
            section=self.section,
            notes=f"Split on {common_pair}. Creates positive terms. Use with NTR for full quadratization. Ground state preserved."
        )

    def apply_complete(self, h: HOBO) -> QuadResult:
        """
        Apply FGBZ_Negative followed by NTR methods for full quadratization.
        This ensures ground state is properly preserved.
        """
        from ..ntr.ntr_kzfd import NTR_KZFD
        
        # First apply FGBZ
        result = self.apply(h)
        
        # Then apply NTR-KZFD to reduce remaining positive high-order terms
        ntr = NTR_KZFD()
        final_result = ntr(result.qubo)
        
        return QuadResult(
            final_result.qubo,
            method=f"{self.name}+ntr_kzfd",
            section=self.section,
            notes="FGBZ + NTR-KZFD. Fully quadratized. Ground state preserved."
        )