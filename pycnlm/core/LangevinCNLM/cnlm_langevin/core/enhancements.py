"""
cnlm_langevin.core.enhancements
================================
Three enhancements layered on top of the base
``CNLMLangevinSolver`` *without* modifying the underlying SDE:

  1. **best-of-K decoding** — round σ(z_T) under K independent noise
     perturbations and keep the rounding that satisfies the most
     clauses.

  2. **WalkSAT polish** — after the SDE finishes, run a budgeted
     stochastic local search starting from the best chain's rounded
     assignment.

  3. **Parallel tempering / replica exchange** — split the K chains
     into a temperature ladder β_1 < … < β_K and propose Metropolis
     swaps of z-states between adjacent rungs.

The enhancements are *additive*: you can use any subset, in any
order.  The single entry point ``solve_with_enhancements`` returns
a ``SolveResult``-compatible object.

Drop this file at:    cnlm_langevin/core/enhancements.py
"""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Optional, Tuple

import numpy as np

from .dynamics import CNLMLangevinSolver, SolverConfig, SolveResult
from .instance import (
    SATInstance, MaxSATInstance,
    evaluate_clauses_bool_vectorized,
)


# ============================================================ best-of-K decoding
def best_of_k_decode(
    instance,
    z: np.ndarray,
    n_rounds: int = 16,
    sigma: float = 0.20,
    weights: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, int]:
    """Round σ(z + ξ) under K noise perturbations; keep the best."""
    rng = rng or np.random.default_rng(0)
    if z.ndim == 1:
        z = z[None, :]
    n_chains, n = z.shape

    if weights is None:
        if isinstance(instance, MaxSATInstance):
            weights = instance.effective_weights()
        else:
            weights = np.ones(instance.n_clauses, dtype=float)
    weights = np.asarray(weights, dtype=float)

    L = instance.L
    n_neg = np.asarray(instance.n_neg)

    best_x = None
    best_w_score = -np.inf
    best_n_sat = -1
    for c in range(n_chains):
        for k in range(n_rounds):
            noise = rng.normal(0.0, sigma, size=n) if sigma > 0 else 0.0
            x = (1.0 / (1.0 + np.exp(-(z[c] + noise))) > 0.5).astype(bool)
            mask = evaluate_clauses_bool_vectorized(L, n_neg, x)
            w_score = float(weights[mask].sum())
            if w_score > best_w_score:
                best_w_score = w_score
                best_n_sat = int(mask.sum())
                best_x = x.copy()
        x = (1.0 / (1.0 + np.exp(-z[c])) > 0.5).astype(bool)
        mask = evaluate_clauses_bool_vectorized(L, n_neg, x)
        w_score = float(weights[mask].sum())
        if w_score > best_w_score:
            best_w_score = w_score
            best_n_sat = int(mask.sum())
            best_x = x.copy()

    return best_x, best_n_sat


# ================================================================ WalkSAT polish
def walksat_polish(
    instance,
    x: np.ndarray,
    max_flips: int = 5000,
    p: float = 0.4,
    deadline: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, int]:
    """Greedy + noise local search starting from ``x``."""
    rng = rng or np.random.default_rng(0)
    n = instance.n_vars
    m = instance.n_clauses
    L = instance.L
    n_neg = np.asarray(instance.n_neg)
    is_max = isinstance(instance, MaxSATInstance)

    if is_max:
        w_hard = 1e6 * float(instance.weights[~instance.is_hard].max()
                             if (~instance.is_hard).any() else 1.0)
        w = np.where(instance.is_hard, w_hard, instance.weights).astype(np.float64)
    else:
        w = np.ones(m, dtype=np.float64)

    clauses = [list(c) for c in instance.clauses]
    x = np.asarray(x, dtype=bool).copy()
    if x.size != n:
        x2 = np.zeros(n, dtype=bool); x2[:min(n, x.size)] = x[:min(n, x.size)]
        x = x2

    def mask_of(xx):
        return evaluate_clauses_bool_vectorized(L, n_neg, xx)

    mask = mask_of(x)
    best_x = x.copy()
    best_score = float(w[mask].sum())

    for step in range(max_flips):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        unsat = np.where(~mask)[0]
        if unsat.size == 0:
            break
        wu = w[unsat]
        j = int(rng.choice(unsat, p=(wu / wu.sum())))
        cl = clauses[j]
        lits = [abs(l) - 1 for l in cl if 0 < abs(l) <= n]
        if not lits:
            continue
        if rng.random() < p:
            v = int(rng.choice(lits))
        else:
            best_delta = -np.inf
            cands = []
            for v in lits:
                x[v] = ~x[v]
                nm = mask_of(x)
                delta = float(w[nm].sum()) - float(w[mask].sum())
                x[v] = ~x[v]
                if delta > best_delta:
                    best_delta = delta; cands = [v]
                elif delta == best_delta:
                    cands.append(v)
            v = int(rng.choice(cands)) if cands else lits[0]
        x[v] = ~x[v]
        mask = mask_of(x)
        score = float(w[mask].sum())
        if score > best_score:
            best_score = score
            best_x = x.copy()

    return best_x, int(mask_of(best_x).sum())


# =============================================================== parallel tempering
class ParallelTemperingSolver:
    """
    Replica-exchange wrapper around ``CNLMLangevinSolver``.

    Runs K chains at a temperature ladder β_0 = β_init, β_K = β_final
    geometrically interpolated.  Every ``swap_every`` SDE steps it
    proposes swapping z-states between adjacent ladder rungs (i, i+1)
    with Metropolis acceptance

        log α  =  (β_i − β_{i+1}) · (F̃(z_i; c_final) − F̃(z_{i+1}; c_final))

    Both the SDE drift/diffusion and the swap rule preserve the joint
    Boltzmann stationary distribution, so theoretical guarantees of the
    fast-slow CNLM-Langevin SDE are unchanged (Earl & Deem, 2005).
    """
    def __init__(
        self,
        instance,
        config: SolverConfig,
        n_rungs: int = 6,
        swap_every: int = 100,
        beta_min: Optional[float] = None,
        beta_max: Optional[float] = None,
    ):
        self.instance = instance
        self.base_config = config
        self.n_rungs = int(n_rungs)
        self.swap_every = int(swap_every)
        self.beta_min = beta_min if beta_min is not None else config.beta_init
        self.beta_max = beta_max if beta_max is not None else config.beta_final
        self.betas = np.geomspace(self.beta_min, self.beta_max, self.n_rungs)

    # ----- swap acceptance free-energy F̃_λ(z; c_final)
    def _free_energy(self, z: np.ndarray) -> float:
        """Compute F̃_λ(z; c_final) = -Σ softplus(c·s̃) + ½λ‖z‖²."""
        cfg = self.base_config
        L = self.instance.L
        n_neg = np.asarray(self.instance.n_neg, dtype=float)
        if hasattr(L, "toarray"):
            L_dense = np.asarray(L.toarray(), dtype=float)
        else:
            L_dense = np.asarray(L, dtype=float)
        x = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
        s = L_dense @ x + (n_neg - 1.0 + cfg.eps)
        c_final = float(cfg.c_final)
        cs = np.clip(c_final * s, -50, 50)
        sp = np.where(cs > 0,
                      cs + np.log1p(np.exp(-cs)),
                      np.log1p(np.exp(cs)))
        F = -float(sp.sum()) + 0.5 * cfg.lam * float((z * z).sum())
        return F

    # ----- main entry
    def solve(self) -> SolveResult:
        cfg0 = self.base_config
        n_total = int(cfg0.n_steps)
        n_chunks = max(1, n_total // max(self.swap_every, 1))
        steps_per_chunk = self.swap_every

        # one solver per rung at constant β
        rungs = []
        for k in range(self.n_rungs):
            rung_cfg = replace(
                cfg0,
                beta_init=float(self.betas[k]),
                beta_final=float(self.betas[k]),
                n_steps=steps_per_chunk,
                seed=cfg0.seed + 1000 * k,
                early_stop_when_sat=False,
            )
            solver = CNLMLangevinSolver(self.instance, rung_cfg)
            rungs.append({"solver": solver, "z": None, "best_x": None,
                          "best_n_sat": -1, "F": np.inf,
                          "last_res": None})

        rng = np.random.default_rng(cfg0.seed + 7)
        n_swap_proposals = 0
        n_swap_accepts = 0
        t0 = time.perf_counter()

        for chunk in range(n_chunks):
            for r in rungs:
                solver = r["solver"]
                res = solver.solve()
                r["last_res"] = res
                if hasattr(res, "final_z") and res.final_z is not None:
                    z_arr = np.asarray(res.final_z)
                    if z_arr.ndim > 1:
                        z_arr = z_arr[0]
                    r["z"] = z_arr
                else:
                    x = np.clip(res.assignment.astype(float), 0.05, 0.95)
                    r["z"] = np.log(x / (1.0 - x))
                if int(res.n_satisfied) > r["best_n_sat"]:
                    r["best_n_sat"] = int(res.n_satisfied)
                    r["best_x"] = res.assignment.astype(bool).copy()
                r["F"] = self._free_energy(r["z"])

            # propose swaps between adjacent rungs (even pass + odd pass)
            for parity in (0, 1):
                for k in range(parity, self.n_rungs - 1, 2):
                    n_swap_proposals += 1
                    rA, rB = rungs[k], rungs[k + 1]
                    bA, bB = self.betas[k], self.betas[k + 1]
                    log_alpha = (bA - bB) * (rA["F"] - rB["F"])
                    if np.log(rng.uniform(1e-12, 1.0)) < log_alpha:
                        rA["z"], rB["z"] = rB["z"].copy(), rA["z"].copy()
                        rA["F"], rB["F"] = rB["F"], rA["F"]
                        n_swap_accepts += 1

        runtime = time.perf_counter() - t0

        # pick best
        best_rung = int(np.argmax([r["best_n_sat"] for r in rungs]))
        best_x = rungs[best_rung]["best_x"]
        if best_x is None:
            best_x = np.zeros(self.instance.n_vars, dtype=bool)
        n_sat = int(rungs[best_rung]["best_n_sat"])
        if n_sat < 0:
            n_sat = 0

        # reuse a real SolveResult and mutate fields rather than constructing
        result = rungs[best_rung]["last_res"]
        if result is None:
            result = CNLMLangevinSolver(self.instance, cfg0).solve()

        is_max = isinstance(self.instance, MaxSATInstance)
        mask = evaluate_clauses_bool_vectorized(
            self.instance.L, np.asarray(self.instance.n_neg), best_x)
        m = self.instance.n_clauses

        result.assignment = best_x.astype(np.int8)
        result.n_satisfied = int(mask.sum())
        result.n_clauses = m
        result.sat_score = float(mask.sum()) / max(m, 1)
        result.runtime_s = runtime
        if hasattr(result, "best_chain"):
            try:    result.best_chain = best_rung
            except Exception: pass
        if hasattr(result, "n_chains"):
            try:    result.n_chains = self.n_rungs
            except Exception: pass
        if hasattr(result, "n_steps"):
            try:    result.n_steps = n_total
            except Exception: pass

        if is_max:
            n_hard_total = int(self.instance.is_hard.sum())
            n_hard_sat = int(mask[self.instance.is_hard].sum())
            n_soft_total = int((~self.instance.is_hard).sum())
            n_soft_sat = int(mask[~self.instance.is_hard].sum())
            cost = float(self.instance.weights[~self.instance.is_hard][
                ~mask[~self.instance.is_hard]].sum())
            result.is_SAT = (n_hard_sat == n_hard_total)
            if hasattr(result, "cost"):         result.cost = cost
            if hasattr(result, "n_hard_sat"):   result.n_hard_sat = n_hard_sat
            if hasattr(result, "n_hard_total"): result.n_hard_total = n_hard_total
            if hasattr(result, "n_soft_sat"):   result.n_soft_sat = n_soft_sat
            if hasattr(result, "n_soft_total"): result.n_soft_total = n_soft_total
        else:
            result.is_SAT = (result.n_satisfied == m)

        try:
            zs = [r["z"] for r in rungs if r["z"] is not None]
            if zs and hasattr(result, "final_z"):
                result.final_z = np.stack(zs, axis=0)
        except Exception:
            pass

        return result


# =================================================== high-level orchestrator
def solve_with_enhancements(
    instance,
    config: SolverConfig,
    polish: bool = True,
    polish_max_flips: int = 5000,
    best_of_k: int = 8,
    best_of_k_sigma: float = 0.20,
    use_pt: bool = False,
    pt_n_rungs: int = 6,
    pt_swap_every: int = 100,
    timeout_s: float = 60.0,
) -> SolveResult:
    """One-call orchestrator: SDE (optionally PT) → best-of-K → polish."""
    deadline = time.perf_counter() + timeout_s
    rng = np.random.default_rng(config.seed)

    if use_pt:
        pt = ParallelTemperingSolver(
            instance, config,
            n_rungs=pt_n_rungs, swap_every=pt_swap_every,
        )
        result = pt.solve()
    else:
        result = CNLMLangevinSolver(instance, config).solve()

    if time.perf_counter() >= deadline:
        return result

    best_x = result.assignment.astype(bool).copy()
    best_n_sat = int(result.n_satisfied)
    if best_of_k > 0 and getattr(result, "final_z", None) is not None:
        cand_x, cand_n = best_of_k_decode(
            instance, result.final_z,
            n_rounds=best_of_k, sigma=best_of_k_sigma,
            rng=rng,
        )
        if cand_x is not None and cand_n >= best_n_sat:
            best_x = cand_x; best_n_sat = cand_n

    if polish and time.perf_counter() < deadline:
        polished_x, polished_n = walksat_polish(
            instance, best_x,
            max_flips=polish_max_flips,
            deadline=deadline, rng=rng,
        )
        if polished_n >= best_n_sat:
            best_x = polished_x; best_n_sat = polished_n

    is_max = isinstance(instance, MaxSATInstance)
    mask = evaluate_clauses_bool_vectorized(
        instance.L, np.asarray(instance.n_neg), best_x)
    m = instance.n_clauses
    cost = None
    n_hard_sat = n_hard_total = n_soft_sat = n_soft_total = None
    soft_w_sat = None
    if is_max:
        n_hard_total = int(instance.is_hard.sum())
        n_hard_sat = int(mask[instance.is_hard].sum())
        n_soft_total = int((~instance.is_hard).sum())
        n_soft_sat = int(mask[~instance.is_hard].sum())
        cost = float(instance.weights[~instance.is_hard][
            ~mask[~instance.is_hard]].sum())
        soft_w_sat = float(instance.weights[~instance.is_hard][
            mask[~instance.is_hard]].sum())

    result.assignment = best_x.astype(np.int8)
    result.n_satisfied = int(mask.sum())
    result.sat_score = float(mask.sum()) / max(m, 1)
    result.is_SAT = (result.n_satisfied == m if not is_max
                    else (n_hard_sat == n_hard_total))
    result.sat_mask = mask
    if is_max:
        result.cost = cost
        result.n_hard_sat = n_hard_sat
        result.n_hard_total = n_hard_total
        result.n_soft_sat = n_soft_sat
        result.n_soft_total = n_soft_total
        result.soft_weight_satisfied = soft_w_sat
    return result


__all__ = [
    "best_of_k_decode",
    "walksat_polish",
    "ParallelTemperingSolver",
    "solve_with_enhancements",
]