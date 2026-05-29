"""
reducer/utils.py
Utility functions for quadratization
"""
from .base import HOBO
from itertools import product
from typing import Dict, Tuple, List

def verify_quadratization(original: HOBO, quadratized: HOBO, tolerance: float = 1e-6) -> Tuple[bool, str]:
    """
    Verify that quadratization preserves the energy spectrum.
    For each original assignment, min over auxiliaries should equal original energy.
    """
    orig_vars = list(range(original.n_vars))
    aux_vars = list(range(original.n_vars, quadratized._aux_counter))
    
    for orig_assignment in product([0, 1], repeat=len(orig_vars)):
        orig_dict = dict(zip(orig_vars, orig_assignment))
        orig_energy = original.evaluate(orig_dict)
        
        min_quad_energy = float('inf')
        if not aux_vars:
            min_quad_energy = quadratized.evaluate(orig_dict)
        else:
            for aux_assignment in product([0, 1], repeat=len(aux_vars)):
                full_assign = orig_dict.copy()
                full_assign.update(dict(zip(aux_vars, aux_assignment)))
                val = quadratized.evaluate(full_assign)
                if val < min_quad_energy:
                    min_quad_energy = val
        
        if abs(orig_energy - min_quad_energy) > tolerance:
            return False, f"Mismatch at {orig_assignment}: {orig_energy} vs {min_quad_energy}"
    
    return True, "Spectrum verified"

def count_submodular_terms(h: HOBO) -> Tuple[int, int]:
    """
    Count submodular vs non-submodular quadratic terms.
    Submodular: negative coefficient for bᵢbⱼ
    Non-submodular: positive coefficient for bᵢbⱼ
    """
    submodular = 0
    non_submodular = 0
    
    for term, coeff in h.terms.items():
        if len(term) == 2:
            if coeff < 0:
                submodular += 1
            elif coeff > 0:
                non_submodular += 1
    
    return submodular, non_submodular

def generate_random_hobo(n_vars: int, max_degree: int, seed: int = None) -> HOBO:
    """Generate a random HOBO for testing."""
    import random
    if seed is not None:
        random.seed(seed)
    
    terms = {}
    for degree in range(1, max_degree + 1):
        from itertools import combinations
        for vars_tuple in combinations(range(n_vars), degree):
            if random.random() < 0.3:  # 30% chance of term existing
                coeff = random.uniform(-10, 10)
                if abs(coeff) > 0.1:
                    terms[frozenset(vars_tuple)] = coeff
    
    return HOBO(terms, n_vars=n_vars)

def find_ground_state(h: HOBO) -> Tuple[Dict[int, int], float]:
    """Find the ground state (minimum energy assignment) by brute force."""
    min_energy = float('inf')
    best_assign = None
    
    for assignment in product([0, 1], repeat=h.n_vars):
        assign_dict = dict(enumerate(assignment))
        energy = h.evaluate(assign_dict)
        if energy < min_energy:
            min_energy = energy
            best_assign = assign_dict
    
    return best_assign, min_energy