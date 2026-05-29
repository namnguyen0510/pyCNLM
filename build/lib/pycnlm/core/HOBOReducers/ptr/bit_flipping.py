"""
reducer/ptr/bit_flipping.py
III-P  Bit Flipping (Ishikawa, 2011)
Replace bᵢ with (1 - b̄ᵢ) to improve submodularity.
This is a symbolic substitution that preserves spectrum exactly.
"""
from ..base import HOBO, QuadResult, QuadratizationMethod

class BitFlipping(QuadratizationMethod):
    name           = "bit_flipping"
    section        = "III-P"
    description    = "Bit flipping to improve submodularity"
    handles_sign   = "any"
    handles_degree = "any"
    aux_per_term   = "0"

    def __init__(self, flip_vars: list = None):
        """
        Args:
            flip_vars: List of variable indices to flip (bᵢ → 1 - b̄ᵢ)
        """
        self.flip_vars = flip_vars or []

    def apply(self, h: HOBO) -> QuadResult:
        if not self.flip_vars:
            # Don't flip anything if not specified - preserves spectrum
            return QuadResult(
                h,
                method=self.name,
                section=self.section,
                notes="No variables flipped. Spectrum preserved."
            )
        
        for var in self.flip_vars:
            if var < h.n_vars:  # Only flip original variables
                self._flip_variable(h, var)
        
        return QuadResult(
            h,
            method=self.name,
            section=self.section,
            notes=f"Flipped {len(self.flip_vars)} variables. Spectrum preserved."
        )

    def _flip_variable(self, h: HOBO, var: int):
        """Replace bᵢ with (1 - b̄ᵢ) in all terms. Preserves spectrum exactly."""
        new_terms = {}
        
        for term, coeff in h.terms.items():
            if var not in term:
                # Term doesn't contain var, keep as is
                new_terms[term] = new_terms.get(term, 0) + coeff
            else:
                # bᵢ = 1 - b̄ᵢ
                # Expand: coeff * product = coeff * (1 - b̄ᵢ) * product(rest)
                rest = frozenset(v for v in term if v != var)
                
                # Term 1: coeff * 1 * product(rest)
                if rest:
                    new_terms[rest] = new_terms.get(rest, 0) + coeff
                else:
                    new_terms[frozenset()] = new_terms.get(frozenset(), 0) + coeff
                
                # Term 2: -coeff * b̄ᵢ * product(rest)
                flipped_term = frozenset(list(rest) + [var])
                new_terms[flipped_term] = new_terms.get(flipped_term, 0) - coeff
        
        h.terms = {k: v for k, v in new_terms.items() if abs(v) > 1e-12}

def main():
    terms = {
        frozenset({0, 1}): 3.0,
        frozenset({1, 2}): 1.0,
        frozenset({0, 3}): 2.0,
        frozenset({1, 3}): -4.0,
    }
    h = HOBO(terms, n_vars=4)
    
    print("Original:", h.to_string())
    print("Original energy at (0,0,0,0):", h.evaluate({0:0, 1:0, 2:0, 3:0}))
    
    method = BitFlipping(flip_vars=[1, 3])
    result = method(h)
    
    print("Flipped:", result.qubo.to_string())
    # Note: After flipping, variable 1 now represents (1 - original_b1)
    # So to compare energies, need to transform assignments

if __name__ == "__main__":
    main()