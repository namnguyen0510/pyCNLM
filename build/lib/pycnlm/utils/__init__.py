"""
pycnlm.utils
============

Small, dependency-light utilities shared across the package.  At the
moment only the DIMACS loader lives here; future helpers (logging
configuration, run-directory management, …) belong in this namespace.
"""
from __future__ import annotations

from pycnlm.utils.dataloader import SATInstance, parse_cnf_file

__all__ = ["SATInstance", "parse_cnf_file"]
