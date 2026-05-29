"""
reducer/ntr/ntr_abcg2.py
II-C  NTR-ABCG-2 (Anthony, Boros, Crama, Gruber, 2016)
Eq. 34: −b₁b₂...bₖ → (2k − 1)bₐ − 2Σᵢ bᵢbₐ
Symmetric with respect to all non-auxiliary variables.
"""
from ..base import HOBO, QuadResult, QuadratizationMethod

class NTR_ABCG2(QuadratizationMethod):
    name           = "ntr_abcg2"
    section        = "II-C"
    description    = "Symmetric quadratization of negative monomials (ABCG-2)"
    handles_sign   = "negative"
    handles_degree = "any"
    aux_per_term   = "1"

    def apply(self, h: HOBO) -> QuadResult:
        to_reduce = [
            (t, c) for t, c in h.terms.items()
            if len(t) >= 3 and c < 0
        ]

        for term, coeff in to_reduce:
            h._add_term(term, -coeff)
            abs_c = -coeff
            k = len(term)
            ba = h._new_aux()
            
            # (2k - 1) * bₐ
            h._add_term(frozenset({ba}), abs_c * (2 * k - 1))
            
            # -2 * Σᵢ bᵢbₐ
            for vi in term:
                h._add_term(frozenset({vi, ba}), -2 * abs_c)
            
        return QuadResult(
            h,
            method=self.name,
            section=self.section,
            notes="1 aux per negative term. Symmetric. Larger coefficients."
        )