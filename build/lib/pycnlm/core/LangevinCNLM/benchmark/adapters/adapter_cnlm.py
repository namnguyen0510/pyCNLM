"""CNLM-Langevin (this paper's method) adapter."""
from __future__ import annotations

import time
import numpy as np

from .base import BaseAdapter, SolveOutcome


class CNLMAdapter(BaseAdapter):
    name = "cnlm_langevin"
    kind = "ours"
    supports = {"SAT", "MaxSAT"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # default to a moderate-cost configuration
        self.config.setdefault("n_steps", 1500)
        self.config.setdefault("n_chains", 24)
        self.config.setdefault("dt", 0.05)
        self.config.setdefault("seed", 0)
        self.config.setdefault("use_slow_sde", True)
        self.config.setdefault("early_stop_when_sat", True)

    def available(self) -> bool:
        try:
            import cnlm_langevin  # noqa: F401
            return True
        except Exception as exc:
            self.unavailable_reason = f"import error: {exc}"
            return False

    def solve(self, instance, timeout_s: float = 120.0) -> SolveOutcome:
        from cnlm_langevin import (
            CNLMLangevinSolver, SolverConfig, MaxSATInstance,
        )
        problem_type = "MaxSAT" if isinstance(instance, MaxSATInstance) else "SAT"
        cfg = SolverConfig(**{k: v for k, v in self.config.items()
                              if k in SolverConfig.__dataclass_fields__})

        t0 = time.perf_counter()
        status, val = self._run_with_timeout(
            lambda: CNLMLangevinSolver(instance, cfg).solve(),
            timeout_s=timeout_s,
        )
        runtime = time.perf_counter() - t0

        out = SolveOutcome(
            solver=self.name, instance=getattr(instance, "name", ""),
            problem_type=problem_type, available=True,
            runtime_s=runtime,
        )
        if status == "TIMEOUT":
            out.timed_out = True
            return out
        if status == "ERROR":
            out.error = val
            return out

        res = val
        out.is_SAT = bool(res.is_SAT)
        out.n_satisfied = int(res.n_satisfied)
        out.n_clauses = int(res.n_clauses)
        out.sat_score = float(res.sat_score)
        out.cost = float(res.cost) if res.cost is not None else None
        out.n_hard_sat = res.n_hard_sat
        out.n_hard_total = res.n_hard_total
        out.n_soft_sat = res.n_soft_sat
        out.n_soft_total = res.n_soft_total
        out.assignment = res.assignment.astype(int).tolist()
        out.extras = {
            "best_chain": int(res.best_chain),
            "converged_step": res.converged_step,
            "n_chains": res.n_chains,
            "n_steps": res.n_steps,
        }
        return out
