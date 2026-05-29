"""
reducer/ntr/ntr_gbp.py
II-D  NTR-GBP (Gallagher, Batra, Parikh, 2011)
Eq. 38-40: Asymmetric cubic reduction
−b₁b₂b₃ → bₐ(−b₁ + b₂ + b₃) − b₁b₂ − b₁b₃ + b₁
"""
from ..base import HOBO, QuadResult, QuadratizationMethod

class NTR_GBP(QuadratizationMethod):
    name           = "ntr_gbp"
    section        = "II-D"
    description    = "Asymmetric cubic reduction (GBP)"
    handles_sign   = "negative"
    handles_degree = "cubic"
    aux_per_term   = "1"

    def apply(self, h: HOBO) -> QuadResult:
        to_reduce = [
            (t, c) for t, c in h.terms.items()
            if len(t) == 3 and c < 0
        ]

        for term, coeff in to_reduce:
            h._add_term(term, -coeff)
            abs_c = -coeff
            
            vars_list = sorted(list(term))
            v1, v2, v3 = vars_list
            ba = h._new_aux()
            
            # Using Eq. 38: bₐ(−b₁ + b₂ + b₃) − b₁b₂ − b₁b₃ + b₁
            # bₐ(-v1 + v2 + v3)
            h._add_term(frozenset({ba, v1}), -abs_c)
            h._add_term(frozenset({ba, v2}), abs_c)
            h._add_term(frozenset({ba, v3}), abs_c)
            
            # -v1v2 - v1v3
            h._add_term(frozenset({v1, v2}), -abs_c)
            h._add_term(frozenset({v1, v3}), -abs_c)
            
            # +v1
            h._add_term(frozenset({v1}), abs_c)
            
        return QuadResult(
            h,
            method=self.name,
            section=self.section,
            notes="1 aux per negative cubic. Asymmetric flexibility."
        )