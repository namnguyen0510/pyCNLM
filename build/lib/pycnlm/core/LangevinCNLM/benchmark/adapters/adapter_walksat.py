"""
Classical WalkSAT — stochastic local search.

A reference, well-known SLS baseline that always runs (no deps beyond
numpy).  WalkSAT picks a random unsatisfied clause, then with probability
p flips a random variable in it, otherwise flips the variable in the
clause that minimises the number of newly broken clauses (greedy).

For MaxSAT we use the standard generalisation:

  * hard clauses count infinitely (we score infeasible flips as +∞)
  * soft clauses are re-weighted by their soft weight
  * the metric ranked is the *weighted* breaks count

Reference: Selman, Kautz & Cohen, "Local Search Strategies for SAT", 1994.
"""
from __future__ import annotations

import time
import numpy as np

from .base import BaseAdapter, SolveOutcome


class WalkSATAdapter(BaseAdapter):
    name = "walksat"
    kind = "classical"
    supports = {"SAT", "MaxSAT"}

    def __init__(self, max_flips: int = 100_000, p: float = 0.5,
                 n_restarts: int = 20, seed: int = 0,
                 **kwargs):
        super().__init__(max_flips=max_flips, p=p,
                         n_restarts=n_restarts, seed=seed, **kwargs)

    def available(self) -> bool:
        return True

    def solve(self, instance, timeout_s: float = 120.0) -> SolveOutcome:
        from cnlm_langevin.core.instance import (
            evaluate_clauses_bool_vectorized, MaxSATInstance,
        )

        problem_type = "MaxSAT" if isinstance(instance, MaxSATInstance) else "SAT"
        n = instance.n_vars
        m = instance.n_clauses
        is_max = isinstance(instance, MaxSATInstance)

        # weights for "weighted breaks": hard clauses get a huge weight
        if is_max:
            w_hard = 1e6 * float(instance.weights[~instance.is_hard].max() if (~instance.is_hard).any() else 1.0)
            w = np.where(instance.is_hard, w_hard, instance.weights).astype(np.float64)
        else:
            w = np.ones(m, dtype=np.float64)

        clauses = [list(c) for c in instance.clauses]
        # var → clauses containing it (as positive or negative literal)
        var_to_clauses = [[] for _ in range(n)]
        for j, cl in enumerate(clauses):
            for lit in cl:
                v = abs(lit) - 1
                if 0 <= v < n:
                    var_to_clauses[v].append(j)

        rng = np.random.default_rng(self.config["seed"])
        max_flips = int(self.config["max_flips"])
        p = float(self.config["p"])
        n_restarts = int(self.config["n_restarts"])

        def evaluate_mask(x: np.ndarray) -> np.ndarray:
            return evaluate_clauses_bool_vectorized(
                instance.L, np.asarray(instance.n_neg), x
            )

        def weighted_unsat(mask: np.ndarray) -> float:
            return float(w[~mask].sum())

        best_score = np.inf  # weighted unsat; lower is better
        best_x = None
        deadline = time.perf_counter() + timeout_s
        flips_total = 0

        for restart in range(n_restarts):
            if time.perf_counter() >= deadline:
                break
            x = rng.integers(0, 2, n).astype(bool)
            mask = evaluate_mask(x)

            for _ in range(max_flips):
                if time.perf_counter() >= deadline:
                    break
                flips_total += 1
                unsat_idx = np.where(~mask)[0]
                if unsat_idx.size == 0:
                    break  # full SAT
                # pick an unsatisfied clause weighted by importance
                w_us = w[unsat_idx]
                pj = w_us / w_us.sum()
                j = int(rng.choice(unsat_idx, p=pj))
                cl = clauses[j]
                lits = [abs(l) - 1 for l in cl if 0 < abs(l) <= n]

                if rng.random() < p or not lits:
                    # noise step: flip a random var from this clause
                    v = int(rng.choice(lits)) if lits else int(rng.integers(0, n))
                else:
                    # greedy step: flip the var that minimises (breaks - makes)
                    best_delta = np.inf
                    cands = []
                    for v in lits:
                        x[v] = ~x[v]
                        new_mask = evaluate_mask(x)
                        delta = (
                            float(w[mask & ~new_mask].sum())  # broken
                            - float(w[~mask & new_mask].sum())  # made
                        )
                        x[v] = ~x[v]   # undo
                        if delta < best_delta:
                            best_delta = delta
                            cands = [v]
                        elif delta == best_delta:
                            cands.append(v)
                    v = int(rng.choice(cands))

                x[v] = ~x[v]
                mask = evaluate_mask(x)
                cur_score = weighted_unsat(mask)
                if cur_score < best_score:
                    best_score = cur_score
                    best_x = x.copy()
                if best_score == 0.0:
                    break
            if best_score == 0.0:
                break

        # finalise
        if best_x is None:
            best_x = np.zeros(n, dtype=bool)
        n_sat, sat_mask, h_sat, s_sat, cost = self._verify(best_x, instance)

        out = SolveOutcome(
            solver=self.name, instance=getattr(instance, "name", ""),
            problem_type=problem_type, available=True,
            runtime_s=min(timeout_s, time.perf_counter() - (deadline - timeout_s)),
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
