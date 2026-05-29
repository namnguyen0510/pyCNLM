"""
reducer/ptr/ptr_kz.py
III-J  PTR-KZ (Kolmogorov & Zabih, 2004)
Eq. 74: b₁b₂b₃ → 1 − (bₐ + b₁ + b₂ + b₃) + bₐ(b₁ + b₂ + b₃) + b₁b₂ + b₁b₃ + b₂b₃
"""
from ..base import HOBO, QuadResult, QuadratizationMethod

class PTR_KZ(QuadratizationMethod):
    name           = "ptr_kz"
    section        = "III-J"
    description    = "Reduction by Minimum Selection (Cubic)"
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
            
            # Constant: c * 1
            h._add_term(frozenset(), coeff)
            
            # Linear: -c * (bₐ + v₁ + v₂ + v₃)
            h._add_term(frozenset({ba}), -coeff)
            h._add_term(frozenset({v1}), -coeff)
            h._add_term(frozenset({v2}), -coeff)
            h._add_term(frozenset({v3}), -coeff)
            
            # Quadratic (Aux): c * bₐ(v₁ + v₂ + v₃)
            h._add_term(frozenset({ba, v1}), coeff)
            h._add_term(frozenset({ba, v2}), coeff)
            h._add_term(frozenset({ba, v3}), coeff)
            
            # Quadratic (Original): c * (v₁v₂ + v₁v₃ + v₂v₃)
            h._add_term(frozenset({v1, v2}), coeff)
            h._add_term(frozenset({v1, v3}), coeff)
            h._add_term(frozenset({v2, v3}), coeff)
            
        return QuadResult(
            h,
            method=self.name,
            section=self.section,
            notes="1 aux per cubic. All 6 quadratic terms non-submodular."
        )