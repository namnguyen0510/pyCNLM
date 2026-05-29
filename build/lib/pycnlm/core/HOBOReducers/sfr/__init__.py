"""
pycnlm.core.HOBOReducers.sfr
============================

Symmetric-Function Reductions (SFR).

A single quadratizer is currently exported:

* :class:`SFR_ABCG1` — Anthony–Boros–Crama–Gruber variant 1.

(The SFR_BCR* variants are on the roadmap; their imports are commented
out until their modules ship.)
"""
from __future__ import annotations

from pycnlm.core.HOBOReducers.sfr.sfr_abcg1 import SFR_ABCG1

# Forthcoming variants:
# from .sfr_bcr1 import SFR_BCR1
# from .sfr_bcr2 import SFR_BCR2
# from .sfr_bcr3 import SFR_BCR3
# from .sfr_bcr4 import SFR_BCR4

__all__ = ["SFR_ABCG1"]
