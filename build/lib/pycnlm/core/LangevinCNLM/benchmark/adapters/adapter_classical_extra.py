"""
Survey Propagation (SP) — classical message-passing SAT solver.

Reference:
    Mézard, Parisi, Zecchina (Science 2002),
    "Analytic and algorithmic solution of random satisfiability problems."

This is the canonical neural-flavoured *non-learned* SAT baseline (a
factor-graph belief-propagation variant).  Highly competitive on
random 3-SAT near the SAT/UNSAT phase boundary.

We implement vanilla SP-decimation:

  1. Run survey-propagation message updates until convergence;
  2. Pick the most-biased variable, fix it according to its bias;
  3. Simplify the formula and repeat;
  4. When BP-style messages dominate or the residual formula gets small,
     hand off to a few rounds of WalkSAT-style local search.

No training required.  Always available with just NumPy.
"""
from __future__ import annotations

import time
from typing import Tuple
import numpy as np

from .base import BaseAdapter, SolveOutcome


class SurveyPropagationAdapter(BaseAdapter):
    name = "survey_propagation"
    kind = "classical"
    supports = {"SAT", "MaxSAT"}    # MaxSAT works as a best-effort

    def __init__(self, max_sweeps: int = 200, eps: float = 1e-3,
                 rho: float = 0.0, decimate_frac: float = 0.05,
                 seed: int = 0, **kwargs):
        super().__init__(max_sweeps=max_sweeps, eps=eps, rho=rho,
                         decimate_frac=decimate_frac, seed=seed, **kwargs)

    def available(self) -> bool:
        return True

    def solve(self, instance, timeout_s: float = 120.0) -> SolveOutcome:
        from cnlm_langevin.core.instance import (
            evaluate_clauses_bool_vectorized, MaxSATInstance,
        )
        is_max = isinstance(instance, MaxSATInstance)
        n = instance.n_vars
        m = instance.n_clauses

        L = instance.L
        if hasattr(L, "toarray"):
            L_dense = np.asarray(L.toarray(), dtype=np.int8)
        else:
            L_dense = np.asarray(L, dtype=np.int8)

        rng = np.random.default_rng(self.config["seed"])
        deadline = time.perf_counter() + timeout_s

        # build clause-to-vars and var-to-clauses adjacency
        # cls[j] = list of (var, sign) pairs; var[i] = list of (clause, sign)
        cls = [[] for _ in range(m)]
        vars_ = [[] for _ in range(n)]
        for j in range(m):
            row = L_dense[j]
            for i in range(n):
                if row[i] == 1:
                    cls[j].append((i, +1))
                    vars_[i].append((j, +1))
                elif row[i] == -1:
                    cls[j].append((i, -1))
                    vars_[i].append((j, -1))

        # SP messages η_{j→i}: probability that clause j sends a "warning"
        # to variable i, indicating i must satisfy j.
        # Initialise randomly in (0,1).
        eta = {}
        for j in range(m):
            for i, _ in cls[j]:
                eta[(j, i)] = float(rng.uniform(0.05, 0.95))

        # current assignment & alive masks
        x = np.full(n, -1, dtype=np.int8)     # -1 = unset, 0/1 otherwise
        clause_alive = np.ones(m, dtype=bool)
        var_alive = np.ones(n, dtype=bool)

        max_sweeps = int(self.config["max_sweeps"])
        eps = float(self.config["eps"])
        decim_frac = float(self.config["decimate_frac"])

        def sp_update_sweep():
            """One sweep of SP message updates.  Returns max change."""
            max_diff = 0.0
            for j in range(m):
                if not clause_alive[j]:
                    continue
                lits = cls[j]
                for i, sg_i in lits:
                    if not var_alive[i]:
                        continue
                    # u_{k→j}(i) = product over k ≠ j of (warnings from k that
                    # variable i must take a value contradicting clause j)
                    # Simplified version: aggregate over other clauses where i
                    # appears with the opposite sign.
                    # See Braunstein-Mézard-Zecchina 2005 for the full update.
                    prod_pos, prod_neg = 1.0, 1.0
                    for jj, sg in vars_[i]:
                        if jj == j or not clause_alive[jj]:
                            continue
                        e = eta[(jj, i)]
                        if sg == sg_i:
                            prod_pos *= (1.0 - e)
                        else:
                            prod_neg *= (1.0 - e)
                    # u^+ favors clause j unsatisfied via i; u^- the opposite
                    pi_u = (1.0 - prod_pos) * prod_neg
                    pi_s = (1.0 - prod_neg) * prod_pos
                    pi_0 = prod_pos * prod_neg
                    denom = pi_u + pi_s + pi_0
                    if denom < 1e-12:
                        new_eta = 0.0
                    else:
                        new_eta = pi_u / denom
                    diff = abs(new_eta - eta[(j, i)])
                    if diff > max_diff:
                        max_diff = diff
                    eta[(j, i)] = new_eta
            return max_diff

        def variable_biases():
            """Return (bias_pos, bias_neg, bias_zero) per *alive* variable."""
            biases = np.zeros((n, 3))   # cols: +, -, free
            for i in range(n):
                if not var_alive[i]:
                    continue
                p_plus, p_minus = 1.0, 1.0
                for jj, sg in vars_[i]:
                    if not clause_alive[jj]:
                        continue
                    e = eta[(jj, i)]
                    if sg > 0:
                        # clause j will be satisfied if x_i = 1
                        p_minus *= (1.0 - e)
                    else:
                        p_plus *= (1.0 - e)
                # un-normalised marginal probabilities of x_i being +/-/free
                Pi_plus = (1.0 - p_plus) * p_minus
                Pi_minus = p_plus * (1.0 - p_minus)
                Pi_zero = p_plus * p_minus
                Z = Pi_plus + Pi_minus + Pi_zero
                if Z < 1e-12:
                    biases[i] = (1/3, 1/3, 1/3)
                else:
                    biases[i] = (Pi_plus / Z, Pi_minus / Z, Pi_zero / Z)
            return biases

        # main decimation loop
        for outer in range(n):
            if time.perf_counter() >= deadline:
                break
            if not var_alive.any():
                break
            # converge SP messages
            converged = False
            for sw in range(max_sweeps):
                if time.perf_counter() >= deadline:
                    break
                d = sp_update_sweep()
                if d < eps:
                    converged = True
                    break
            biases = variable_biases()

            # check trivial paramagnetic (all biases ≈ uniform) — fallback to BP/local-search
            alive_idx = np.where(var_alive)[0]
            if alive_idx.size == 0:
                break
            mags = biases[alive_idx, 0] - biases[alive_idx, 1]
            # decimate the top-magnitude variables
            n_to_fix = max(1, int(decim_frac * alive_idx.size))
            order = np.argsort(-np.abs(mags))
            for k in range(min(n_to_fix, alive_idx.size)):
                i = int(alive_idx[order[k]])
                # set x_i = 1 if Π+ > Π−, else 0
                x[i] = 1 if biases[i, 0] >= biases[i, 1] else 0
                var_alive[i] = False
                # simplify: kill clauses satisfied by this assignment, drop literals from the rest
                for jj, sg in vars_[i]:
                    if not clause_alive[jj]:
                        continue
                    if (sg > 0 and x[i] == 1) or (sg < 0 and x[i] == 0):
                        clause_alive[jj] = False
                # drop the messages associated with i
                # (no need to rebuild — the alive masks gate the updates)

            if not converged:
                # paramagnetic — just exit and hand off
                break

        # for any remaining unset variables, fill with 0 (then run SLS to clean up)
        for i in range(n):
            if x[i] == -1:
                x[i] = 0

        x_bool = x.astype(bool)
        # quick WalkSAT polish: try a small budget of greedy-or-noise flips
        x_bool = self._walksat_polish(instance, x_bool,
                                      max_flips=200 * n,
                                      deadline=deadline,
                                      rng=rng)

        n_sat, _, h_sat, s_sat, cost = self._verify(x_bool, instance)
        out = SolveOutcome(
            solver=self.name, instance=getattr(instance, "name", ""),
            problem_type=("MaxSAT" if is_max else "SAT"),
            available=True, runtime_s=time.perf_counter() - (deadline - timeout_s),
            timed_out=(time.perf_counter() >= deadline and n_sat < m),
            is_SAT=(n_sat == m if not is_max
                    else (h_sat == int(instance.is_hard.sum()))),
            n_satisfied=n_sat, n_clauses=m,
            sat_score=n_sat / max(m, 1),
            assignment=x_bool.astype(int).tolist(),
        )
        if is_max:
            out.cost = cost
            out.n_hard_sat = h_sat
            out.n_hard_total = int(instance.is_hard.sum())
            out.n_soft_sat = s_sat
            out.n_soft_total = int((~instance.is_hard).sum())
        out.extras = {"sweeps_per_decimation": self.config["max_sweeps"]}
        return out

    @staticmethod
    def _walksat_polish(instance, x: np.ndarray, max_flips: int,
                        deadline: float, rng) -> np.ndarray:
        """Quick WalkSAT-style polish on the SP-decimated assignment."""
        from cnlm_langevin.core.instance import evaluate_clauses_bool_vectorized
        L = instance.L; n_neg = np.asarray(instance.n_neg)
        n = instance.n_vars
        clauses = [list(c) for c in instance.clauses]

        def mask_of(xx):
            return evaluate_clauses_bool_vectorized(L, n_neg, xx)

        m_arr = mask_of(x)
        for _ in range(max_flips):
            if time.perf_counter() >= deadline:
                break
            unsat_idx = np.where(~m_arr)[0]
            if unsat_idx.size == 0:
                break
            j = int(rng.choice(unsat_idx))
            cl = clauses[j]
            lits = [abs(l) - 1 for l in cl if 0 < abs(l) <= n]
            if not lits:
                continue
            if rng.random() < 0.4:
                v = int(rng.choice(lits))
            else:
                # greedy: flip the variable that breaks fewest sat clauses
                best_v, best_delta = lits[0], -np.inf
                for v in lits:
                    x[v] = ~x[v]
                    nm = mask_of(x)
                    delta = int(nm.sum()) - int(m_arr.sum())
                    x[v] = ~x[v]
                    if delta > best_delta:
                        best_delta = delta; best_v = v
                v = best_v
            x[v] = ~x[v]
            m_arr = mask_of(x)
        return x


class SimulatedAnnealingAdapter(BaseAdapter):
    """Classical simulated-annealing baseline for SAT/MaxSAT."""
    name = "simulated_annealing"
    kind = "classical"
    supports = {"SAT", "MaxSAT"}

    def __init__(self, n_steps: int = 50_000, T_init: float = 5.0,
                 T_final: float = 0.05, seed: int = 0, **kwargs):
        super().__init__(n_steps=n_steps, T_init=T_init, T_final=T_final,
                         seed=seed, **kwargs)

    def available(self) -> bool:
        return True

    def solve(self, instance, timeout_s: float = 120.0) -> SolveOutcome:
        from cnlm_langevin.core.instance import (
            evaluate_clauses_bool_vectorized, MaxSATInstance,
        )
        is_max = isinstance(instance, MaxSATInstance)
        n = instance.n_vars; m = instance.n_clauses
        L = instance.L; n_neg = np.asarray(instance.n_neg)

        if is_max:
            w_hard = 1e6 * float(instance.weights[~instance.is_hard].max()
                                  if (~instance.is_hard).any() else 1.0)
            w = np.where(instance.is_hard, w_hard, instance.weights).astype(np.float64)
        else:
            w = np.ones(m, dtype=np.float64)

        rng = np.random.default_rng(self.config["seed"])
        steps = int(self.config["n_steps"])
        T0 = float(self.config["T_init"])
        T1 = float(self.config["T_final"])
        x = rng.integers(0, 2, n).astype(bool)

        def score(xx):
            return float(w[evaluate_clauses_bool_vectorized(L, n_neg, xx)].sum())

        cur = score(x)
        best_x = x.copy(); best_score = cur
        deadline = time.perf_counter() + timeout_s

        for step in range(steps):
            if time.perf_counter() >= deadline:
                break
            T = T0 * (T1 / T0) ** (step / max(steps - 1, 1))
            v = int(rng.integers(0, n))
            x[v] = ~x[v]
            new = score(x)
            d = new - cur          # higher = better
            if d > 0 or rng.random() < np.exp(d / max(T, 1e-6)):
                cur = new
                if cur > best_score:
                    best_score = cur; best_x = x.copy()
            else:
                x[v] = ~x[v]

        n_sat, _, h_sat, s_sat, cost = self._verify(best_x, instance)
        out = SolveOutcome(
            solver=self.name, instance=getattr(instance, "name", ""),
            problem_type=("MaxSAT" if is_max else "SAT"),
            available=True, runtime_s=time.perf_counter() - (deadline - timeout_s),
            timed_out=(time.perf_counter() >= deadline and n_sat < m),
            is_SAT=(n_sat == m if not is_max
                    else (h_sat == int(instance.is_hard.sum()))),
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
        out.extras = {"steps": steps}
        return out
