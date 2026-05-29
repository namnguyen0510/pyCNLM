"""
pycnlm.core.AdaptCNLM
=====================

Symmetry-based qubit-reduction and embedding strategies for SAT problems
on quantum-annealer hardware (Chimera, Pegasus).

Three encoders are provided, each striking a different qubit / fidelity
trade-off:

* :class:`OrbitBasedEncoder`   — one qubit per structural orbit.
* :class:`CliqueBasedEncoder`  — qubits grouped by clique structure.
* :class:`ClusterBasedEncoder` — qubits grouped by greedy clustering.

The :class:`SymmetryDetector` discovers the orbits used by all three.

Optional dependencies
---------------------
Embedding onto physical D-Wave topologies additionally requires::

    pip install "pycnlm[dwave]"

When those packages are unavailable the encoders still function in
"logical" mode; only the hardware-embedding step is skipped.
"""
from __future__ import annotations

from pycnlm.core.AdaptCNLM.AdaptCNLM import (
    SymmetryDetector,
    OrbitBasedEncoder,
    CliqueBasedEncoder,
    ClusterBasedEncoder,
)

__all__ = [
    "SymmetryDetector",
    "OrbitBasedEncoder",
    "CliqueBasedEncoder",
    "ClusterBasedEncoder",
]
