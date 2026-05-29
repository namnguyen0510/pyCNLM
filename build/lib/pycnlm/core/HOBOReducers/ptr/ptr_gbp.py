"""
reducer/ptr/ptr_gbp.py
III-L  PTR-GBP (Gallagher, Batra, Parikh, 2011)
Eq. 76-78: Asymmetric positive cubic reduction
b₁b₂b₃ → bₐ − b₂bₐ − b₃bₐ + b₁bₐ + b₂b₃
"""
from ..base import HOBO, QuadResult, QuadratizationMethod

class PTR_GBP(QuadratizationMethod):
    name           = "ptr_gbp"
    section        = "III-L"
    description    = "Asymmetric positive cubic reduction (GBP)"
    handles_sign   = "positive"
    handles_degree = "cubic"
    aux_per_term   = "1"

    def apply(self, h: HOBO) -> QuadResult:
        to_reduce = [
            (t, c) for t, c in h.terms.items()
            if len(t) == 3 and c > 0
        ]

        for term, coeff in to_reduce:
            h._add_term(term, -coeff)
            
            vars_list = sorted(list(term))
            v1, v2, v3 = vars_list
            ba = h._new_aux()
            
            # Eq. 76: bₐ − b₂bₐ − b₃bₐ + b₁bₐ + b₂b₃
            h._add_term(frozenset({ba}), coeff)
            h._add_term(frozenset({ba, v2}), -coeff)
            h._add_term(frozenset({ba, v3}), -coeff)
            h._add_term(frozenset({ba, v1}), coeff)
            h._add_term(frozenset({v2, v3}), coeff)
            
        return QuadResult(
            h,
            method=self.name,
            section=self.section,
            notes="1 aux per positive cubic. Fewer non-submodular terms."
        )