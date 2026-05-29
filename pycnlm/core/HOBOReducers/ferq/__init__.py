"""
reducer/ferq/__init__.py
Fermat Quadratization (FERQ) methods - ancilla-free reduction using Fermat quotients
"""

from .ferq_method import FERQ, create_ferq_evaluator

__all__ = ["FERQ", "create_ferq_evaluator"]