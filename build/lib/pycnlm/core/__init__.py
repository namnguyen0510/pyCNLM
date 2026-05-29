"""
pycnlm.core
===========

Scientific sub-packages.  Three independent components live here:

* :mod:`pycnlm.core.AdaptCNLM`     — symmetry-based qubit reduction.
* :mod:`pycnlm.core.HOBOReducers`  — quadratization library.
* :mod:`pycnlm.core.LangevinCNLM`  — Langevin SDE SAT / MaxSAT solver.

These are kept as sibling sub-packages so each can evolve, be vendored, or
be benchmarked independently.  Most users should import from the top-level
:mod:`pycnlm` package, which re-exports the curated public surface.
"""
from __future__ import annotations

from pycnlm.core import AdaptCNLM, HOBOReducers, LangevinCNLM  # noqa: F401

__all__ = ["AdaptCNLM", "HOBOReducers", "LangevinCNLM"]
