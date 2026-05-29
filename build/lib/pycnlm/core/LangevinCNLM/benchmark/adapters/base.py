"""
Unified adapter interface for benchmarking neural / classical SAT/MaxSAT
solvers against CNLM-Langevin.

Every concrete adapter subclasses :class:`BaseAdapter` and implements

    * `name`            — short identifier (e.g. "neurosat")
    * `kind`            — "neural" | "classical" | "ours"
    * `supports`        — set of {"SAT", "MaxSAT"}
    * `available()`     — True iff the adapter can actually run; otherwise
                          set ``self.unavailable_reason`` to a human string
    * `solve(instance, timeout_s=...) -> SolveOutcome`

Every neural baseline is allowed to be unavailable (missing weights,
missing CUDA, incompatible TF version, …) — the benchmark driver will
just skip it and record the reason.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional, Set, Union

import numpy as np


@dataclass
class SolveOutcome:
    """Unified return type for any adapter's solve() call."""
    solver: str
    instance: str
    problem_type: str            # "SAT" | "MaxSAT"
    available: bool

    # primary metrics (only meaningful if available and no error)
    is_SAT: Optional[bool] = None
    n_satisfied: Optional[int] = None
    n_clauses: Optional[int] = None
    sat_score: Optional[float] = None        # n_satisfied / n_clauses
    cost: Optional[float] = None             # MaxSAT objective: Σ_unsat w_j (soft)
    n_hard_sat: Optional[int] = None
    n_hard_total: Optional[int] = None
    n_soft_sat: Optional[int] = None
    n_soft_total: Optional[int] = None

    runtime_s: float = 0.0
    timed_out: bool = False
    error: Optional[str] = None              # exception string if any
    unavailable_reason: Optional[str] = None  # set if available=False

    # the actual assignment (kept short — first few hundred bits at most)
    assignment: Optional[list] = None

    # arbitrary per-solver extras
    extras: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        """Flatten for CSV writing."""
        d = asdict(self)
        # avoid blowing up CSV with the full assignment
        if d.get("assignment") is not None and len(d["assignment"]) > 64:
            d["assignment"] = (
                "[" + ",".join(str(x) for x in d["assignment"][:32])
                + ",...,"
                + ",".join(str(x) for x in d["assignment"][-32:])
                + "]"
            )
        # extras → JSON string; coerce numpy scalars to Python primitives
        import json
        def _coerce(v):
            if hasattr(v, "item"):
                try: return v.item()
                except Exception: return str(v)
            if isinstance(v, dict):
                return {k: _coerce(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [_coerce(x) for x in v]
            return v
        d["extras"] = json.dumps(_coerce(d.get("extras", {})))
        return d


class BaseAdapter:
    name: str = "base"
    kind: str = "classical"          # "neural" | "classical" | "ours"
    supports: Set[str] = {"SAT"}     # which problem types this adapter handles

    def __init__(self, **kwargs):
        self.config = dict(kwargs)
        self.unavailable_reason: Optional[str] = None

    # ----------------------------------------------------------------- API
    def available(self) -> bool:
        """Override.  Set self.unavailable_reason and return False if not."""
        return True

    def solve(self, instance, timeout_s: float = 120.0) -> SolveOutcome:
        raise NotImplementedError

    # ----------------------------------------------------------- Helpers
    @staticmethod
    def _verify(assignment: np.ndarray, instance) -> tuple:
        """
        Independently verify an assignment against the instance.
        Returns (n_satisfied, sat_mask, hard_sat, soft_sat, cost) where
        cost is the MaxSAT cost (Σ unsat soft weights) or 0.0 for SAT.
        """
        from cnlm_langevin.core.instance import (
            evaluate_clauses_bool_vectorized,
            MaxSATInstance,
        )
        x = np.asarray(assignment, dtype=bool).ravel()
        if x.size != instance.n_vars:
            # truncate or pad with False
            x2 = np.zeros(instance.n_vars, dtype=bool)
            x2[: min(x.size, instance.n_vars)] = x[: min(x.size, instance.n_vars)]
            x = x2
        sat_mask = evaluate_clauses_bool_vectorized(
            instance.L, np.asarray(instance.n_neg), x
        )
        n_sat = int(sat_mask.sum())
        if isinstance(instance, MaxSATInstance):
            hard = instance.is_hard
            hard_sat = int(sat_mask[hard].sum())
            soft_sat = int(sat_mask[~hard].sum())
            cost = float(instance.weights[~hard][~sat_mask[~hard]].sum())
            return n_sat, sat_mask, hard_sat, soft_sat, cost
        return n_sat, sat_mask, None, None, 0.0

    def _run_with_timeout(self, fn, timeout_s: float):
        """
        Run ``fn()`` in a daemon thread, return its result.  If the call
        outlives ``timeout_s``, return ('TIMEOUT', None).  Note: the
        thread keeps running in the background — adapters should expose
        their own best-effort cancellation if possible.
        """
        import threading
        result = {"value": None, "error": None}

        def target():
            try:
                result["value"] = fn()
            except Exception as exc:  # noqa: BLE001
                result["error"] = (
                    f"{type(exc).__name__}: {exc}\n"
                    + traceback.format_exc(limit=3)
                )

        th = threading.Thread(target=target, daemon=True)
        th.start()
        th.join(timeout=timeout_s)
        if th.is_alive():
            return ("TIMEOUT", None)
        if result["error"]:
            return ("ERROR", result["error"])
        return ("OK", result["value"])

    # ----------------------------------------------------------- repr
    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name} kind={self.kind}>"
