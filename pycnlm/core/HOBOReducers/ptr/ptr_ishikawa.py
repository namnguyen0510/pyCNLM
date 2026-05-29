"""
reducer/ptr/ptr_ishikawa.py
III-B  PTR-Ishikawa (Ishikawa, 2011)
Eq. 50: Symmetric polynomial reduction
b₁...bₖ → (Σ_{i=1}^{nₖ} b_{ai}(c_{i,k}(-Σⱼ bⱼ + 2i) - 1) + Σ_{i<j} bᵢbⱼ)
where nₖ = ⌊(k-1)/2⌋
"""
from ..base import HOBO, QuadResult, QuadratizationMethod
import math

class PTR_Ishikawa(QuadratizationMethod):
    name           = "ptr_ishikawa"
    section        = "III-B"
    description    = "Ishikawa's Symmetric Reduction"
    handles_sign   = "positive"
    handles_degree = "any"
    aux_per_term   = "floor((k-1)/2)"

    def apply(self, h: HOBO) -> QuadResult:
        to_reduce = [
            (t, c) for t, c in h.terms.items()
            if len(t) >= 3 and c > 0
        ]

        for term, coeff in to_reduce:
            h._add_term(term, -coeff)
            
            k = len(term)
            vars_list = sorted(list(term))
            n_k = math.floor((k - 1) / 2)
            
            aux_indices = h._new_aux_batch(n_k)
            
            # Σ_{i<j} bᵢbⱼ (All pairwise products)
            for i in range(k):
                for j in range(i + 1, k):
                    h._add_term(frozenset({vars_list[i], vars_list[j]}), coeff)
            
            # Σ_{i=1}^{nₖ} b_{ai}(c_{i,k}(-Σ bⱼ + 2i) - 1)
            for idx, aux in enumerate(aux_indices):
                i = idx + 1
                c_ik = 1 if (i == n_k and k % 2 == 1) else 2
                
                # -1 * b_{ai}
                h._add_term(frozenset({aux}), coeff * -1)
                
                # c_{i,k} * (-Σ bⱼ) * b_{ai}
                for v in vars_list:
                    h._add_term(frozenset({aux, v}), coeff * c_ik * -1)
                
                # c_{i,k} * 2i * b_{ai}
                h._add_term(frozenset({aux}), coeff * c_ik * 2 * i)
                
        return QuadResult(
            h,
            method=self.name,
            section=self.section,
            notes="floor((k-1)/2) aux. Full spectrum reproduced."
        )