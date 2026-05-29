"""
reducer/ntr/ntr_kzfd.py
II-A  NTR-KZFD  (Kolmogorov & Zabih 2004; Freedman & Drineas 2005)
For each negative k-local term  c · x₁…xₖ  (c < 0),
introduce ONE auxiliary variable bₐ and substitute (Eq. 25):
−x₁…xₖ  →  (k−1)·bₐ  −  Σᵢ xᵢ·bₐ
"""
from ..base import HOBO, QuadResult, QuadratizationMethod

class NTR_KZFD(QuadratizationMethod):
    name           = "ntr_kzfd"
    section        = "II-A"
    description    = "Standard quadratization of negative monomials"
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

            h._add_term(frozenset({ba}), abs_c * (k - 1))
            for vi in term:
                h._add_term(frozenset({vi, ba}), -abs_c)

        return QuadResult(
            h,
            method=self.name,
            section=self.section,
            notes="1 aux per negative term. All quadratic couplings submodular."
        )