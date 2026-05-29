"""
pycnlm.core.LangevinCNLM
========================

Container for the CNLM-Langevin solver and its benchmarking machinery.

* :mod:`pycnlm.core.LangevinCNLM.cnlm_langevin` — the solver itself.
* :mod:`pycnlm.core.LangevinCNLM.benchmark`    — benchmark drivers and
  baseline adapters.

The ``examples/`` directory holds tutorial notebooks (not part of the
Python package) and ``third_party/`` vendors reference implementations
of competing methods used in benchmarks (also not imported by default).

On import this package installs a backwards-compatibility alias so that the
historical top-level name ``cnlm_langevin`` resolves to the nested
:mod:`pycnlm.core.LangevinCNLM.cnlm_langevin` package.  See
:mod:`pycnlm.core.LangevinCNLM._cnlm_alias` for the rationale.
"""
from __future__ import annotations

# Register the ``cnlm_langevin`` -> nested-package alias as early as
# possible so that *any* downstream code (benchmark driver, adapters, user
# scripts) can ``import cnlm_langevin`` regardless of install mode.
from pycnlm.core.LangevinCNLM._cnlm_alias import install as _install_cnlm_alias

_install_cnlm_alias()

from pycnlm.core.LangevinCNLM import cnlm_langevin  # noqa: E402,F401

__all__ = ["cnlm_langevin"]
