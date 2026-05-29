"""
Backwards-compatibility import alias for the historical top-level module
name ``cnlm_langevin``.

Why this exists
---------------
The CNLM-Langevin solver was originally developed as a stand-alone project
in which ``cnlm_langevin`` was an *importable top-level package*.  All of
the benchmark driver and adapter modules therefore use absolute imports of
the form::

    from cnlm_langevin import SATInstance, parse_dimacs_cnf
    from cnlm_langevin.core.instance import MaxSATInstance

When the project was folded into the ``pycnlm`` distribution the solver
moved to ``pycnlm.core.LangevinCNLM.cnlm_langevin``.  Those absolute imports
only resolve if ``pycnlm/core/LangevinCNLM/`` happens to be on ``sys.path``
(true when you run a script from that folder, false for an installed
wheel).  That is the source of the intermittent
``ModuleNotFoundError: No module named 'cnlm_langevin'``.

Rather than rewrite ~25 import sites across the (frozen) solver and
benchmark code, we install a :class:`importlib.abc.MetaPathFinder` that
transparently maps any import of ``cnlm_langevin`` / ``cnlm_langevin.*``
onto the real ``pycnlm.core.LangevinCNLM.cnlm_langevin[.*]`` module.

Key properties
--------------
* **Single underlying module object.**  The alias registers the *same*
  module instance under both names in ``sys.modules`` so that
  ``isinstance`` checks and class identity keep working across the two
  import paths.
* **Non-intrusive.**  If the real package cannot be imported (e.g. a stray
  ``cnlm_langevin`` genuinely exists elsewhere on ``sys.path``) the finder
  declines and the normal import machinery takes over.
* **Idempotent.**  Calling :func:`install` more than once is a no-op.
"""
from __future__ import annotations

import importlib
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import spec_from_loader
from typing import Optional, Sequence

_SHORT = "cnlm_langevin"
_LONG = "pycnlm.core.LangevinCNLM.cnlm_langevin"


class _CnlmLangevinAliasFinder(MetaPathFinder, Loader):
    """Map ``cnlm_langevin[.sub]`` imports to ``{_LONG}[.sub]``."""

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]] = None,
        target: Optional[object] = None,
    ) -> Optional[ModuleSpec]:
        if fullname != _SHORT and not fullname.startswith(_SHORT + "."):
            return None
        long_name = _LONG + fullname[len(_SHORT):]
        try:
            # Importing here both validates availability and ensures the
            # real module is cached under its canonical (long) name.
            importlib.import_module(long_name)
        except Exception:
            # Decline — let the standard finders try (and surface their
            # own, more informative, error if nothing resolves).
            return None
        return spec_from_loader(fullname, self)

    def create_module(self, spec: ModuleSpec):
        long_name = _LONG + spec.name[len(_SHORT):]
        module = sys.modules.get(long_name) or importlib.import_module(long_name)
        # Alias the *same* object under the short name.
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module) -> None:  # noqa: D401
        # The real module has already been executed under its long name.
        return None


_INSTALLED = False


def install() -> None:
    """Install the alias finder (idempotent)."""
    global _INSTALLED
    if _INSTALLED:
        return
    # Insert at the front so we are consulted before path-based finders.
    sys.meta_path.insert(0, _CnlmLangevinAliasFinder())
    _INSTALLED = True


__all__ = ["install"]
