"""
Trivial random-restart + greedy-flip baseline.

Lower-bound reference solver: pick a random assignment, perform greedy
hill-climbing on the (weighted) sat count until a local optimum, then
restart.  Useful as a sanity floor against learned baselines.
"""
from __future__ import annotations

import time
import numpy as np

from .base import BaseAdapter, SolveOutcome


class RandomRestartGreedyAdapter(BaseAdapter):
    name = "random_restart_greedy"
    kind = "classical"
    supports = {"SAT", "MaxSAT"}

    def __init__(self, n_restarts: int = 200, max_flips_per_restart: int = 1000,
                 seed: int = 0, **kwargs):
        super().__init__(n_restarts=n_restarts,
                         max_flips_per_restart=max_flips_per_restart,
                         seed=seed, **kwargs)

    def available(self) -> bool:
        return True

    def solve(self, instance, timeout_s: float = 120.0) -> SolveOutcome:
        from cnlm_langevin.core.instance import (
            evaluate_clauses_bool_vectorized, MaxSATInstance,
        )
        is_max = isinstance(instance, MaxSATInstance)
        n = instance.n_vars
        m = instance.n_clauses

        if is_max:
            w_hard = 1e6 * float(instance.weights[~instance.is_hard].max() if (~instance.is_hard).any() else 1.0)
            w = np.where(instance.is_hard, w_hard, instance.weights).astype(np.float64)
        else:
            w = np.ones(m, dtype=np.float64)

        rng = np.random.default_rng(self.config["seed"])
        n_restarts = int(self.config["n_restarts"])
        max_flips = int(self.config["max_flips_per_restart"])

        def mask_of(x):
            return evaluate_clauses_bool_vectorized(
                instance.L, np.asarray(instance.n_neg), x
            )

        deadline = time.perf_counter() + timeout_s
        best_score = -np.inf  # weighted sat count; higher better
        best_x = None
        flips_total = 0
        restart = 0
        for restart in range(n_restarts):
            if time.perf_counter() >= deadline:
                break
            x = rng.integers(0, 2, n).astype(bool)
            mask = mask_of(x)
            score = float(w[mask].sum())
            for _ in range(max_flips):
                if time.perf_counter() >= deadline:
                    break
                flips_total += 1
                # try every variable, take the best single-flip
                best_dv, best_ds = None, 0.0
                for v in range(n):
                    x[v] = ~x[v]
                    new_mask = mask_of(x)
                    new_score = float(w[new_mask].sum())
                    ds = new_score - score
                    if ds > best_ds:
                        best_ds = ds
                        best_dv = v
                    x[v] = ~x[v]
                if best_dv is None:
                    break  # local optimum
                x[best_dv] = ~x[best_dv]
                mask = mask_of(x)
                score = float(w[mask].sum())
            if score > best_score:
                best_score = score
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(n, dtype=bool)
        n_sat, sat_mask, h_sat, s_sat, cost = self._verify(best_x, instance)

        problem_type = "MaxSAT" if is_max else "SAT"
        out = SolveOutcome(
            solver=self.name, instance=getattr(instance, "name", ""),
            problem_type=problem_type, available=True,
            runtime_s=time.perf_counter() - (deadline - timeout_s),
            timed_out=(time.perf_counter() >= deadline and n_sat < m),
            is_SAT=(n_sat == m if not is_max else (h_sat == int(instance.is_hard.sum()))),
            n_satisfied=n_sat, n_clauses=m,
            sat_score=n_sat / max(m, 1),
            assignment=best_x.astype(int).tolist(),
        )
        if is_max:
            out.cost = cost
            out.n_hard_sat = h_sat
            out.n_hard_total = int(instance.is_hard.sum())
            out.n_soft_sat = s_sat
            out.n_soft_total = int((~instance.is_hard).sum())
        out.extras = {"flips": flips_total, "restarts_used": restart + 1}
        return out
