"""
pycnlm.core.HOBOReducers.ptr
============================

Positive-Term-Reduction (PTR) quadratizers.

These methods replace a single positive monomial of degree :math:`d \\geq 3`
with a quadratic expression by introducing one or more auxiliary variables.
They differ in the *number* of auxiliaries, in their support pattern, and
in the depth of the resulting QUBO graph.

Currently exported:

* :class:`PTR_BG`       — Boros–Gruber.
* :class:`PTR_Ishikawa` — Ishikawa's reduction.
* :class:`PTR_KZ`       — Kolmogorov–Zabih variant.
* :class:`PTR_GBP`      — generalised Boros–Pardalos.
* :class:`BitFlipping`  — global polarity flip pre-step.

(The PTR_BCR* and PTR_CCG variants remain on the roadmap; their imports
are commented out in this module until their implementations land.)
"""
from __future__ import annotations

from pycnlm.core.HOBOReducers.ptr.ptr_bg import PTR_BG
from pycnlm.core.HOBOReducers.ptr.ptr_ishikawa import PTR_Ishikawa
from pycnlm.core.HOBOReducers.ptr.ptr_kz import PTR_KZ
from pycnlm.core.HOBOReducers.ptr.ptr_gbp import PTR_GBP
from pycnlm.core.HOBOReducers.ptr.bit_flipping import BitFlipping

# Forthcoming variants (uncomment as their modules ship):
# from .ptr_bcr1 import PTR_BCR1
# from .ptr_bcr2 import PTR_BCR2
# from .ptr_bcr3 import PTR_BCR3
# from .ptr_bcr4 import PTR_BCR4
# from .ptr_bcr5 import PTR_BCR5
# from .ptr_ccg  import PTR_CCG

__all__ = [
    "PTR_BG",
    "PTR_Ishikawa",
    "PTR_KZ",
    "PTR_GBP",
    "BitFlipping",
]
