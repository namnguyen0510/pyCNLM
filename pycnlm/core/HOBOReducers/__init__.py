"""
reducer/__init__.py
Main package initialization - exports all quadratization methods from Part I
"""

# Base classes
from .base import HOBO, QuadResult, QuadratizationMethod

# Zero auxiliary methods (Section I)
from .zero_aux.deduc_reduc import DeducReduc
from .zero_aux.elc_reduction import ELCReduction
from .zero_aux.split_reduction import SplitReduction

# NTR methods (Section II - Negative Term Reductions)
from .ntr.ntr_kzfd import NTR_KZFD
from .ntr.ntr_abcg import NTR_ABCG
from .ntr.ntr_abcg2 import NTR_ABCG2
from .ntr.ntr_gbp import NTR_GBP

# PTR methods (Section III - Positive Term Reductions)
from .ptr.ptr_bg import PTR_BG
from .ptr.ptr_ishikawa import PTR_Ishikawa
from .ptr.ptr_kz import PTR_KZ
from .ptr.ptr_gbp import PTR_GBP
from .ptr.bit_flipping import BitFlipping

# Arbitrary function methods (Section V)
from .arb.substitution import ReductionBySubstitution
from .arb.fgbz_negative import FGBZ_Negative
from .arb.fgbz_positive import FGBZ_Positive
from .arb.pairwise_covers import PairwiseCovers

# FERQ methods (Our - Fermat Quadratization)
from .ferq.ferq_method import FERQ, create_ferq_evaluator

__all__ = [
    # Base
    "HOBO", "QuadResult", "QuadratizationMethod",
    
    # Zero Aux (Section I)
    "DeducReduc", "ELCReduction", "SplitReduction",
    
    # NTR (Section II)
    "NTR_KZFD", "NTR_ABCG", "NTR_ABCG2", "NTR_GBP",
    
    # PTR (Section III)
    "PTR_BG", "PTR_Ishikawa", "PTR_KZ", "PTR_GBP", "BitFlipping",
    
    # Arbitrary (Section V)
    "ReductionBySubstitution", "FGBZ_Negative", "FGBZ_Positive", "PairwiseCovers",
    
    # FERQ (Section VII)
    "FERQ", "create_ferq_evaluator",
]

__version__ = "1.0.0"
__author__ = "Based on Dattani Survey (arXiv:1901.04405) + Nguyen & Tran 2026"