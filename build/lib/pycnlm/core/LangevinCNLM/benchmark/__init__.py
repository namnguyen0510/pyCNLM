"""
pycnlm.core.LangevinCNLM.benchmark
==================================

Benchmark harness for comparing the CNLM-Langevin solver against
classical (PySAT, WalkSAT, …) and neural (NeuroSAT, SATNet, …) baselines
on standard SAT / MaxSAT corpora.

Adapters live in :mod:`pycnlm.core.LangevinCNLM.benchmark.adapters`; the
top-level driver entry-point is :func:`driver.run_benchmark` (see
``driver.py``).

Importing this package installs a small import-alias shim (see
:mod:`pycnlm.core.LangevinCNLM._cnlm_alias`) so that the driver and adapter
modules — which use the historical absolute name ``cnlm_langevin`` — resolve
correctly regardless of whether ``pycnlm`` was pip-installed or is being run
from a source checkout.  This is what fixes the
``ModuleNotFoundError: No module named 'cnlm_langevin'`` seen when running
``benchmark_SAT.py`` from an installed package.
"""
from __future__ import annotations

# Install the alias *before* any submodule (driver, adapters) is imported,
# because those modules do ``from cnlm_langevin import ...`` at import time.
# Guarded so that running from a legacy flat layout (where ``cnlm_langevin``
# is already importable as a real top-level package) still works.
try:
    from pycnlm.core.LangevinCNLM._cnlm_alias import install as _install_cnlm_alias

    _install_cnlm_alias()
except Exception:  # pragma: no cover - defensive fallback for odd layouts
    pass

__all__: list = []
