"""
reducer/arbitrary/substitution.py
V-A  Reduction by Substitution (Rosenberg 1975)
Eq. 147: bᵢbⱼbₖ → bₐbₖ + bᵢbⱼ − 2bᵢbₐ − 2bⱼbₐ + 3bₐ
"""
from ..base import HOBO, QuadResult, QuadratizationMethod

class ReductionBySubstitution(QuadratizationMethod):
    name           = "substitution"
    section        = "V-A"
    description    = "Reduction by Substitution (Rosenberg)"
    handles_sign   = "any"
    handles_degree = "any"
    aux_per_term   = "k-2 per term"

    def apply(self, h: HOBO) -> QuadResult:
        while h.degree > 2:
            term_to_reduce = None
            coeff = 0
            for t, c in h.terms.items():
                if len(t) > 2:
                    term_to_reduce = t
                    coeff = c
                    break
            
            if term_to_reduce is None:
                break
                
            vars_list = sorted(list(term_to_reduce))
            v1, v2 = vars_list[0], vars_list[1]
            rest = vars_list[2:]
            
            ba = h._new_aux()
            h._add_term(term_to_reduce, -coeff)
            
            # FIX: Convert to list before combining, then to frozenset
            new_term = frozenset([ba] + rest)
            h._add_term(new_term, coeff)
            
            penalty_scale = abs(coeff)
            h._add_term(frozenset({v1, v2}), penalty_scale)
            h._add_term(frozenset({v1, ba}), -2 * penalty_scale)
            h._add_term(frozenset({v2, ba}), -2 * penalty_scale)
            h._add_term(frozenset({ba}), 3 * penalty_scale)
            
        return QuadResult(
            h,
            method=self.name,
            section=self.section,
            notes="Recursive substitution. Full spectrum preserved."
        )