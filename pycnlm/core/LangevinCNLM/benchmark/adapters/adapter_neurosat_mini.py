"""
NeuroSAT-mini adapter — wraps the small in-package PyTorch GNN trained
in this session (see ``cnlm_langevin/baselines/neurosat_mini.py``).

The trained checkpoint ships with the package at
``cnlm_langevin/baselines/weights/neurosat_mini.pt``, so this adapter
is **always available** as long as PyTorch is installed.
"""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np

from .base import BaseAdapter, SolveOutcome


class NeuroSATMiniAdapter(BaseAdapter):
    """In-package NeuroSAT-style GNN, trained from scratch on synthetic 3-SAT."""
    name = "neurosat_mini"
    kind = "neural"             # genuine neural baseline (trained)
    supports = {"SAT", "MaxSAT"}

    def __init__(self, ckpt_path: str = None, n_random_rounds: int = 16,
                 n_restarts: int = 4, **kwargs):
        super().__init__(ckpt_path=ckpt_path,
                         n_random_rounds=n_random_rounds,
                         n_restarts=n_restarts, **kwargs)

    def available(self) -> bool:
        try:
            import torch  # noqa: F401
        except Exception as exc:
            self.unavailable_reason = (
                f"PyTorch not installed (`pip install torch`).  Import failed: {exc}"
            )
            return False
        # locate the in-package weights
        from cnlm_langevin.baselines.neurosat_mini import DEFAULT_CKPT
        ckpt = self.config.get("ckpt_path") or DEFAULT_CKPT
        ckpt = Path(ckpt)
        if not ckpt.exists():
            self.unavailable_reason = (
                f"NeuroSAT-mini weights not found at {ckpt}. "
                f"Train via `python -m cnlm_langevin.baselines.neurosat_mini`."
            )
            return False
        self._ckpt = ckpt
        return True

    def solve(self, instance, timeout_s: float = 120.0) -> SolveOutcome:
        from cnlm_langevin.baselines.neurosat_mini import (
            load_neurosat_mini, predict_assignment,
        )
        from cnlm_langevin.core.instance import MaxSATInstance
        is_max = isinstance(instance, MaxSATInstance)
        n_clauses = instance.n_clauses

        t0 = time.perf_counter()
        try:
            model = load_neurosat_mini(self._ckpt, device="cpu")
        except Exception as exc:
            return SolveOutcome(
                solver=self.name, instance=getattr(instance, "name", ""),
                problem_type=("MaxSAT" if is_max else "SAT"), available=True,
                runtime_s=time.perf_counter() - t0,
                error=f"failed to load checkpoint: {exc}",
            )

        # multiple restarts: each restart re-runs predict_assignment with
        # different rounding noise and we keep the best result
        best_x = None
        best_n_sat = -1
        n_restarts = int(self.config["n_restarts"])
        for r in range(n_restarts):
            try:
                x = predict_assignment(
                    model, instance, device="cpu",
                    n_random_rounds=int(self.config["n_random_rounds"]),
                )
            except Exception as exc:
                return SolveOutcome(
                    solver=self.name, instance=getattr(instance, "name", ""),
                    problem_type=("MaxSAT" if is_max else "SAT"), available=True,
                    runtime_s=time.perf_counter() - t0,
                    error=f"inference error: {exc}",
                )
            n_sat, _, _, _, _ = self._verify(x, instance)
            if n_sat > best_n_sat:
                best_n_sat = n_sat
                best_x = x
            if best_n_sat == n_clauses:
                break

        runtime = time.perf_counter() - t0
        if best_x is None:
            best_x = np.zeros(instance.n_vars, dtype=bool)
        n_sat, _, h_sat, s_sat, cost = self._verify(best_x, instance)

        out = SolveOutcome(
            solver=self.name, instance=getattr(instance, "name", ""),
            problem_type=("MaxSAT" if is_max else "SAT"),
            available=True, runtime_s=runtime,
            is_SAT=(n_sat == n_clauses if not is_max
                    else (h_sat == int(instance.is_hard.sum()))),
            n_satisfied=n_sat, n_clauses=n_clauses,
            sat_score=n_sat / max(n_clauses, 1),
            assignment=best_x.astype(int).tolist(),
            extras={"checkpoint": str(self._ckpt),
                    "n_random_rounds": self.config["n_random_rounds"],
                    "n_restarts": n_restarts},
        )
        if is_max:
            out.cost = cost
            out.n_hard_sat = h_sat
            out.n_hard_total = int(instance.is_hard.sum())
            out.n_soft_sat = s_sat
            out.n_soft_total = int((~instance.is_hard).sum())
        return out
