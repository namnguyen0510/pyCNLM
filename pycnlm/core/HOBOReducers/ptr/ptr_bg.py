"""
reducer/ptr/ptr_bg.py
III-A  PTR-BG (Boros and Gruber, 2014)
Eq. 48: b₁b₂...bₖ → (Σ_{i=1}^{k-2} b_{aᵢ}(k - i - 1 + bᵢ - Σ_{j=i+1}^k bⱼ)) + b_{k-1}bₖ

NOTE: Ground state preserved only. Requires k-2 auxiliaries per k-local term.
"""
from ..base import HOBO, QuadResult, QuadratizationMethod

class PTR_BG(QuadratizationMethod):
    name           = "ptr_bg"
    section        = "III-A"
    description    = "Recursive NTR-KZFD for positive monomials (Boros-Gruber)"
    handles_sign   = "positive"
    handles_degree = "any"
    aux_per_term   = "k-2"
    preserves_spectrum = False

    def apply(self, h: HOBO) -> QuadResult:
        to_reduce = [
            (t, c) for t, c in h.terms.items()
            if len(t) >= 3 and c > 0
        ]

        for term, coeff in to_reduce:
            h._add_term(term, -coeff)  # Remove original
            k = len(term)
            
            if k < 3:
                continue
            
            vars_list = sorted(list(term))
            aux_vars = h._new_aux_batch(k - 2)
            
            # Eq. 48: Σ_{i=1}^{k-2} b_{aᵢ}(k - i - 1 + bᵢ - Σ_{j=i+1}^k bⱼ) + b_{k-1}bₖ
            for i in range(k - 2):
                ai = aux_vars[i]
                bi = vars_list[i]
                
                # Linear: b_{aᵢ} × (k - i - 1)
                h._add_term(frozenset({ai}), coeff * (k - i - 1))
                
                # Quadratic: +b_{aᵢ}bᵢ
                h._add_term(frozenset({ai, bi}), coeff)
                
                # Quadratic: -b_{aᵢ}bⱼ for j > i
                for j in range(i + 1, k):
                    h._add_term(frozenset({ai, vars_list[j]}), -coeff)
            
            # Final term: +b_{k-1}bₖ
            h._add_term(frozenset({vars_list[-2], vars_list[-1]}), coeff)
            
        return QuadResult(
            h,
            method=self.name,
            section=self.section,
            notes="k-2 aux per term. Ground state preserved."
        )