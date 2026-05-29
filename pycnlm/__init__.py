"""
pycnlm
======

A unified Python toolkit for **Continuous Non-Linear Manifold (CNLM)**
methods applied to combinatorial optimization — in particular
Boolean Satisfiability (SAT) and Maximum Satisfiability (MaxSAT).

The package ships three complementary components:

* :mod:`pycnlm.langevin`
    The **CNLM-Langevin** fast–slow stochastic-differential-equation solver
    for SAT and MaxSAT in DIMACS ``.cnf`` / ``.wcnf`` format.

* :mod:`pycnlm.hobo`
    A library of **Higher-Order Binary Optimization** quadratization /
    reduction methods (NTR, PTR, SFR, FERQ, ELC, FGBZ, …) for compiling
    polynomial pseudo-Boolean objectives onto QUBO-class hardware.

* :mod:`pycnlm.adapt`
    **Adaptive CNLM** symmetry-based qubit-count reduction and
    embedding for quantum annealers (Chimera / Pegasus topologies).

Quick start
-----------
>>> import pycnlm
>>> # Solve a CNF file with the Langevin solver
>>> result = pycnlm.solve_sat_file("instance.cnf")
>>> print(result.is_sat, result.best_energy)

Refer to https://github.com/your-org/pycnlm for full documentation.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Version --------------------------------------------------------------------
# ---------------------------------------------------------------------------
from pycnlm._version import __version__, __version_info__

# ---------------------------------------------------------------------------
# Sub-package aliases --------------------------------------------------------
# The scientific code lives under a slightly deeper tree (``pycnlm.core.*``)
# to keep file-level imports inside the research modules stable.  We expose
# convenient short aliases here so end-users see a flat, idiomatic API:
# ``pycnlm.langevin``, ``pycnlm.hobo``, ``pycnlm.adapt``.
# ---------------------------------------------------------------------------
from pycnlm.core.LangevinCNLM import cnlm_langevin as langevin  # noqa: F401
from pycnlm.core import HOBOReducers as hobo  # noqa: F401
from pycnlm.core import AdaptCNLM as adapt  # noqa: F401

# Register the short aliases in sys.modules so they behave like real
# sub-modules, i.e. ``import pycnlm.langevin`` works in addition to
# ``pycnlm.langevin`` attribute access.
import sys as _sys

_sys.modules.setdefault("pycnlm.langevin", langevin)
_sys.modules.setdefault("pycnlm.hobo", hobo)
_sys.modules.setdefault("pycnlm.adapt", adapt)
del _sys

# ---------------------------------------------------------------------------
# Re-export the most common public objects at the top level so that
# ``from pycnlm import X`` works for what 99 % of users need.
# ---------------------------------------------------------------------------

# --- Shared data structures -------------------------------------------------
from pycnlm.utils.dataloader import SATInstance, parse_cnf_file

# --- CNLM-Langevin solver ---------------------------------------------------
from pycnlm.core.LangevinCNLM.cnlm_langevin import (
    DimacsParseError,
    parse_dimacs_cnf,
    parse_dimacs_wcnf,
    parse_dimacs_auto,
    SATInstance as LangevinSATInstance,
    MaxSATInstance,
    build_literal_matrix,
    CNLMLangevinSolver,
    SolverConfig,
    SolveResult,
    solve_sat_file,
    solve_maxsat_file,
    solve_folder,
)

# --- HOBO reducer library ---------------------------------------------------
from pycnlm.core.HOBOReducers import (
    HOBO,
    QuadResult,
    QuadratizationMethod,
)

# --- AdaptCNLM encoders -----------------------------------------------------
from pycnlm.core.AdaptCNLM import (
    SymmetryDetector,
    OrbitBasedEncoder,
    CliqueBasedEncoder,
    ClusterBasedEncoder,
)

# ---------------------------------------------------------------------------
# Logging --------------------------------------------------------------------
# Library convention: attach a NullHandler so users see no log output unless
# they explicitly configure logging in their application.
# ---------------------------------------------------------------------------
import logging as _logging

_logging.getLogger(__name__).addHandler(_logging.NullHandler())
del _logging

__all__ = [
    # Version
    "__version__",
    "__version_info__",
    # Sub-package shortcuts
    "langevin",
    "hobo",
    "adapt",
    # Shared
    "SATInstance",
    "parse_cnf_file",
    # Langevin solver
    "DimacsParseError",
    "parse_dimacs_cnf",
    "parse_dimacs_wcnf",
    "parse_dimacs_auto",
    "LangevinSATInstance",
    "MaxSATInstance",
    "build_literal_matrix",
    "CNLMLangevinSolver",
    "SolverConfig",
    "SolveResult",
    "solve_sat_file",
    "solve_maxsat_file",
    "solve_folder",
    # HOBO reducers
    "HOBO",
    "QuadResult",
    "QuadratizationMethod",
    # Adapt
    "SymmetryDetector",
    "OrbitBasedEncoder",
    "CliqueBasedEncoder",
    "ClusterBasedEncoder",
]
