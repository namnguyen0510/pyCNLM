"""
reducer/ntr/ntr_abcg.py
II-B  NTR-ABCG (Anthony, Boros, Crama, Gruber, 2014)
Eq. 30: −b₁...bₖ → Σ_{i=1}^{k-1} bᵢ − Σ_{i=1}^{k-1} bᵢbₖ − Σ_{i=1}^{k} bᵢbₐ + (k−1)bₖbₐ
"""
from ..base import HOBO, QuadResult, QuadratizationMethod

class NTR_ABCG(QuadratizationMethod):
    name           = "ntr_abcg"
    section        = "II-B"
    description    = "Extended standard quadratization of negative monomials"
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
            
            vars_list = sorted(list(term))
            bk = vars_list[-1]
            others = vars_list[:-1]
            
            for vi in others:
                h._add_term(frozenset({vi}), abs_c)
            for vi in others:
                h._add_term(frozenset({vi, bk}), -abs_c)
            for vi in vars_list:
                h._add_term(frozenset({vi, ba}), -abs_c)
            h._add_term(frozenset({bk, ba}), abs_c * (k - 1))
            
        return QuadResult(
            h,
            method=self.name,
            section=self.section,
            notes="1 aux per negative term. Asymmetric (last variable special)."
        )