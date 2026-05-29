"""
cnlm_langevin.core.analysis
===========================
Pre-solve diagnostic analyses on a SAT/MaxSAT instance.

Currently exposed:

* :func:`analyze_gradient_snr` — signal-to-noise ratio of the
  per-clause confidence gradient over uniform random x, as a
  function of the confidence c.  Useful for choosing a c-schedule.

The companion plot lives in :mod:`cnlm_langevin.core.viz` as
``plot_gradient_snr``.
"""
from __future__ import annotations

from typing import Optional, Union, Iterable

import numpy as np

from .instance import SATInstance, MaxSATInstance


def _dense_L(L) -> np.ndarray:
    if hasattr(L, "toarray"):
        return np.asarray(L.toarray(), dtype=float)
    return np.asarray(L, dtype=float)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def analyze_gradient_snr(
    instance: Union[SATInstance, MaxSATInstance],
    c_grid: Optional[Iterable[float]] = None,
    n_samples: int = 2000,
    eps: float = 0.5,
    seed: int = 0,
    aggregate: str = "mean",
) -> dict:
    """
    Compute the signal-to-noise ratio of the confidence-gradient

        ∂F̃_j/∂c_j  =  -w_j · σ(c·s̃_j(x)) · s̃_j(x)

    over uniform random x ∈ (0,1)^n, sweeping c on a logarithmic grid.

    For each c on the grid:

      * ``mu_j(c) = E_x[∂F_j/∂c_j]``
      * ``sd_j(c) = std_x[∂F_j/∂c_j]``
      * ``snr_j(c) = |mu_j(c)| / sd_j(c)``

    The returned scalar SNR is the per-clause SNR aggregated according
    to ``aggregate``: ``"mean"`` (default), ``"median"`` or ``"max"``.

    Parameters
    ----------
    instance
        SATInstance or MaxSATInstance — its ``L``, ``n_neg`` and (for MaxSAT)
        ``weights`` are used.
    c_grid
        Iterable of c values.  Default: 25 points on log-grid 0.1 → 50.
    n_samples
        Number of uniform-random x ∈ (0,1)^n samples used per c.
    eps
        SDNF margin ε ∈ (0,1).
    seed
        RNG seed.
    aggregate
        How to reduce per-clause SNR to a single scalar:
        ``"mean"`` | ``"median"`` | ``"max"``.

    Returns
    -------
    dict
        ``{'c_grid', 'snr', 'snr_per_clause', 'n_samples', 'aggregate'}``
        suitable for ``cnlm.plot_gradient_snr``.
    """
    rng = np.random.default_rng(seed)
    n = instance.n_vars
    m = instance.n_clauses

    L_dense = _dense_L(instance.L)
    n_neg = np.asarray(instance.n_neg, dtype=float)
    if hasattr(instance, "weights") and instance.weights is not None:
        weights = np.asarray(instance.weights, dtype=float)
    else:
        weights = np.ones(m, dtype=float)

    if c_grid is None:
        c_grid = np.logspace(-1, np.log10(50.0), 25)
    c_grid = np.asarray(list(c_grid), dtype=float)

    X = rng.uniform(0.0, 1.0, size=(int(n_samples), n))     # (N, n)
    S = X @ L_dense.T + (n_neg - 1.0 + eps)                  # (N, m)

    snr_per_clause = np.zeros((len(c_grid), m), dtype=float)
    for ci, c in enumerate(c_grid):
        nu = _sigmoid(c * S)                                 # (N, m)
        grad = -weights[None, :] * nu * S                    # (N, m)
        mu = grad.mean(axis=0)
        sd = grad.std(axis=0) + 1e-12
        snr_per_clause[ci] = np.abs(mu) / sd

    if aggregate == "mean":
        snr = snr_per_clause.mean(axis=1)
    elif aggregate == "median":
        snr = np.median(snr_per_clause, axis=1)
    elif aggregate == "max":
        snr = snr_per_clause.max(axis=1)
    else:
        raise ValueError(
            f"aggregate must be 'mean'|'median'|'max', got {aggregate!r}"
        )

    return {
        "c_grid": c_grid,
        "snr": snr,
        "snr_per_clause": snr_per_clause,
        "n_samples": int(n_samples),
        "aggregate": aggregate,
    }


__all__ = ["analyze_gradient_snr"]
