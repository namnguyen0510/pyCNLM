"""Single source of truth for the ``pycnlm`` version string.

The version follows `Semantic Versioning 2.0.0 <https://semver.org/>`_::

    MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]

Bump:

* **MAJOR** for incompatible API changes,
* **MINOR** for additive, backwards-compatible features,
* **PATCH** for backwards-compatible bug fixes.

The build backend reads ``__version__`` from this file at install time;
do not import any heavy dependencies here.
"""
from __future__ import annotations

__version__: str = "0.1.0"

# Convenience tuple for programmatic comparison: ``pycnlm.__version_info__``.
__version_info__: tuple = tuple(
    int(p) if p.isdigit() else p
    for p in __version__.replace("-", ".").replace("+", ".").split(".")
)

__all__ = ["__version__", "__version_info__"]
