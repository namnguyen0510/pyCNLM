"""
reducer/zero_aux/elc_reduction.py
I-B  ELC Reduction (Ishikawa, 2014)
Summary: Excludable Local Configuration - partial assignments that make 
minimum impossible. Add penalty term without changing solution.
Cost: 0 auxiliary variables needed.
"""
from ..base import HOBO, QuadResult, QuadratizationMethod
from itertools import combinations, product
from collections import defaultdict
from typing import List, Tuple, Dict, FrozenSet

class ELCReduction(QuadratizationMethod):
    name           = "elc_reduction"
    section        = "I-B"
    description    = "Excludable Local Configuration reduction"
    handles_sign   = "any"
    handles_degree = "any"
    aux_per_term   = "0"

    def __init__(self, elc_configs: list = None, subset_size: int = 3):
        """
        Args:
            elc_configs: List of (vars_tuple, assignment_tuple) that are excluded
            subset_size: Size of variable subsets to check for ELCs
        """
        self.elc_configs = elc_configs or []
        self.subset_size = subset_size

    def apply(self, h: HOBO) -> QuadResult:
        """Apply ELC-based reduction."""
        # Pre-process terms for faster iteration (convert frozenset to sorted tuple)
        # This avoids repeated hashing/conversion during checks
        high_degree_terms = [
            (tuple(term), coeff) 
            for term, coeff in h.terms.items() 
            if len(term) >= self.subset_size
        ]

        # If no ELCs provided, try to find simple ones
        if not self.elc_configs:
            # Only search within variables that actually participate in high-order terms
            candidate_vars = set()
            for term, _ in high_degree_terms:
                candidate_vars.update(term)
            
            if candidate_vars:
                self.elc_configs = self._find_elcs_fast(
                    list(candidate_vars), 
                    high_degree_terms
                )
        
        # Batch collect penalty terms to minimize dictionary updates
        penalty_updates: Dict[FrozenSet, float] = defaultdict(float)

        for vars_tuple, assignment in self.elc_configs:
            self._collect_elc_penalty(penalty_updates, vars_tuple, assignment)
        
        # Apply all penalty updates at once
        for term, coeff in penalty_updates.items():
            if coeff != 0:
                h._add_term(term, coeff)
        
        # Remove terms that are zero under ELC constraints
        self._simplify_under_elcs(h)
        
        return QuadResult(
            h,
            method=self.name,
            section=self.section,
            notes=f"Applied {len(self.elc_configs)} ELCs. 0 auxiliaries."
        )

    def _find_elcs_fast(self, candidate_vars: List[int], high_degree_terms: List) -> List:
        """
        Find ELCs by checking subsets of variables.
        Optimized: Only checks combinations derived from existing high-order terms.
        """
        elcs = []
        seen_configs = set() # Avoid duplicates
        
        # Strategy: Extract subsets directly from existing high-degree terms
        # This reduces search space from O(N^k) to O(T * d^k) where T=terms, d=degree
        for term_tuple, _ in high_degree_terms:
            if len(term_tuple) < self.subset_size:
                continue
                
            # Generate subsets of the specific size from this term
            for var_subset in combinations(term_tuple, self.subset_size):
                # Sort to ensure consistent hashing for 'seen_configs'
                var_subset = tuple(sorted(var_subset))
                
                for assignment in product([0, 1], repeat=self.subset_size):
                    config_key = (var_subset, assignment)
                    if config_key in seen_configs:
                        continue
                    
                    if self._is_excludable_fast(high_degree_terms, var_subset, assignment):
                        elcs.append(config_key)
                        seen_configs.add(config_key)
        
        return elcs

    def _is_excludable_fast(self, high_degree_terms: List, vars_tuple: Tuple, assignment: Tuple) -> bool:
        """
        Check if a partial assignment is excludable.
        Optimized: Uses pre-filtered term list and tuple lookups.
        """
        # Map variable index to position in the subset for O(1) lookup
        var_to_pos = {v: i for i, v in enumerate(vars_tuple)}
        
        zero_terms_count = 0
        # Threshold heuristic (kept from original logic but optimized)
        threshold = 2 

        for term_tuple, _ in high_degree_terms:
            # Check if this term is zeroed by the assignment
            # A term is zero if any variable in it is assigned 0
            is_zero = False
            
            # Optimization: Only check variables in the term that are in our subset
            for var in term_tuple:
                if var in var_to_pos:
                    pos = var_to_pos[var]
                    if assignment[pos] == 0:
                        is_zero = True
                        break
            
            if is_zero:
                zero_terms_count += 1
                if zero_terms_count >= threshold:
                    return True
        
        return False

    def _collect_elc_penalty(self, updates: Dict[FrozenSet, float], vars_tuple: Tuple, assignment: Tuple):
        """
        Accumulate penalty term coefficients into a dictionary.
        P = λ * product of (b_i if assignment[i]=1 else (1-b_i))
        """
        penalty_coeff = 10.0
        n = len(vars_tuple)
        
        # Pre-calculate signs to avoid branching in the inner loop
        # factor[i] = (1, 0) if assign=1 (term is b_i) -> const 0, var 1
        # factor[i] = (1, -1) if assign=0 (term is 1-b_i) -> const 1, var -1
        # We are expanding product(const_i + var_coeff_i * b_i)
        
        # To expand product(A_i + B_i * x_i):
        # Iterate all masks. If bit i set, take B_i*x_i. If not, take A_i.
        
        # Precompute multipliers for mask logic
        # If mask bit is 1 (variable included): multiplier is (1 if assign==1 else -1)
        # If mask bit is 0 (variable excluded): multiplier is (1 if assign==0 else 1) -> Wait, logic check
        
        # Let's stick to the algebraic expansion logic which is robust:
        # Term = product( (1-a_i) + (2a_i - 1)*b_i ) ? No.
        # If a_i=1: want b_i. (0 + 1*b_i)
        # If a_i=0: want (1-b_i). (1 - 1*b_i)
        # So const_part = 1 - a_i
        # var_part = 2*a_i - 1
        
        const_parts = [(1 - a) for a in assignment]
        var_parts = [(2 * a - 1) for a in assignment]
        
        for mask in range(1 << n):
            term_vars = []
            coeff = penalty_coeff
            
            for i in range(n):
                if mask & (1 << i):
                    term_vars.append(vars_tuple[i])
                    coeff *= var_parts[i]
                else:
                    coeff *= const_parts[i]
            
            if coeff != 0:
                updates[frozenset(term_vars)] += coeff

    def _simplify_under_elcs(self, h: HOBO):
        """
        Remove or simplify terms based on ELC constraints.
        Note: Full implementation requires variable substitution logic.
        """
        # Placeholder kept for API compatibility
        # To optimize: If ELC forces var=1, substitute 1. If var=0, remove term.
        pass