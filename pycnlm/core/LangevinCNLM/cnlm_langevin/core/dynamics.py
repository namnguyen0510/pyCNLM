"""
CNLM-Langevin (fast-slow) solver.

Faithful Euler-Maruyama discretisation of the coupled Itô SDE
(Definition 4.1 in the paper):

    dz_t  = -∇_z F~_λ(z_t; c_t) dt + sqrt(2/β(t)) dW^z_t       # FAST
    dρ_t  = -η ∇_ρ L(z_t; e^{ρ_t}) dt + sqrt(2η/β_c) dW^ρ_t     # SLOW
    c_t   = exp(ρ_t)

with the lifted, regularised free energy

    F~_λ(z; c) = -Σ_j w_j ln(1 + exp(c_j s~_j(z))) + (λ/2)||z||²

and the CNF clause score

    s~_j(z) = Σ_i L[j,i] σ(z_i) + (|N_j| - 1 + ε)
            = (L σ(z))_j + b_j

The only nonzero entries of L are ±1 corresponding to literal polarity.

Two annealing modes are supported and can be combined:
  *  deterministic:  β(t), c(t) follow user schedules;
  *  stochastic:     ρ_t evolves under its own Langevin SDE.

Multiple independent chains are vectorised across an extra "K" dimension
so that NumPy's BLAS releases the GIL and a single solver call already
runs many parallel walkers.  Cross-instance parallelism is provided by
`solver.solve_folder`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .instance import (
    SATInstance,
    MaxSATInstance,
    matvec_L,
    matvec_LT,
    evaluate_clauses_bool_vectorized,
)


# ============================================================================
# numerics helpers
# ============================================================================
def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid, vectorised."""
    out = np.empty_like(z)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def _sigmoid_deriv_from_sigma(sx: np.ndarray) -> np.ndarray:
    """σ'(z) = σ(z)(1-σ(z)) given σ(z)."""
    return sx * (1.0 - sx)


def _logsigmoid(z: np.ndarray) -> np.ndarray:
    """log σ(z) numerically stable."""
    return -np.logaddexp(0.0, -z)


def _softplus_pos(z: np.ndarray) -> np.ndarray:
    """ln(1+e^z) numerically stable, vectorised."""
    # numpy's logaddexp is robust
    return np.logaddexp(0.0, z)


# ============================================================================
# configuration
# ============================================================================
@dataclass
class SolverConfig:
    """
    Hyper-parameters of the CNLM-Langevin solver.

    Annealing schedules (for fast mode β and slow mode c):
      * 'log'   — log(1+t)-style cooling                     (β recommended)
      * 'lin'   — linear ramp                                (c recommended)
      * 'poly'  — polynomial t^p ramp
      * 'const' — constant
    """
    # --- discretisation
    n_steps: int = 1500             # total Euler-Maruyama steps
    dt: float = 0.05                # time step (z dynamics)
    n_chains: int = 16              # parallel Langevin walkers
    seed: Optional[int] = None
    eps: float = 0.5                # SDNF margin (must be in (0,1))
    lam: float = 1e-3               # quadratic regulariser λ ≥ 0
    z_init_scale: float = 0.5       # σ of Gaussian initialisation of z
    z_clip: float = 30.0            # clip |z| to keep σ-derivatives non-degenerate

    # --- fast schedule (β)
    beta_init: float = 1.0
    beta_final: float = 80.0
    beta_schedule: str = "log"      # 'log' | 'lin' | 'poly' | 'const'
    beta_poly_p: float = 1.0

    # --- slow schedule (c)
    c_init: float = 1.0
    c_final: float = 60.0
    c_schedule: str = "lin"
    c_poly_p: float = 1.5

    # --- slow SDE on ρ = log c (use_slow_sde=True turns on noisy ρ)
    use_slow_sde: bool = False
    eta: float = 0.05               # slow learning rate η
    beta_c: float = 50.0            # inverse temperature on ρ
    c_min: float = 1e-2
    c_max: float = 1e3

    # --- restarts and convergence
    restart_on_stuck: bool = True   # kick a chain that's been stuck for too long
    stuck_patience: int = 200       # steps without best-improvement before kick
    early_stop_when_sat: bool = True  # stop if any chain reaches all-SAT

    # --- bookkeeping / verbosity
    record_history_every: int = 1   # record every k steps
    record_assignment_every: int = 0  # 0 disables to save memory; e.g. 10 stores every 10 steps
    verbose: bool = False

    # --- MaxSAT specifics
    hard_scale: float = 1e3         # hard-clause weight multiplier (relative to max soft)


# ============================================================================
# results
# ============================================================================
@dataclass
class SolveResult:
    """Container for solver outputs."""
    # primary
    assignment: np.ndarray          # (n,) Boolean assignment of best chain
    is_SAT: bool                    # for SAT: full satisfaction; for MaxSAT: hard satisfaction
    n_satisfied: int
    n_clauses: int
    sat_score: float                # n_satisfied / n_clauses

    # MaxSAT extras (populated only for MaxSAT)
    cost: Optional[float] = None
    soft_weight_satisfied: Optional[float] = None
    n_hard_sat: Optional[int] = None
    n_hard_total: Optional[int] = None
    n_soft_sat: Optional[int] = None
    n_soft_total: Optional[int] = None

    # diagnostics
    best_chain: int = 0
    n_chains: int = 0
    n_steps: int = 0
    runtime_s: float = 0.0
    converged_step: Optional[int] = None  # step at which best chain hit current best

    # per-step histories (each is an ndarray of length n_recorded_steps)
    history_steps: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    history_beta: np.ndarray = field(default_factory=lambda: np.zeros(0))
    history_c_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    history_c_min: np.ndarray = field(default_factory=lambda: np.zeros(0))
    history_c_max: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # per-step per-chain stats: shape (n_recorded_steps, n_chains)
    history_free_energy: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    history_n_sat: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=int))
    history_best_n_sat: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    # final per-chain assignments (all chains)
    final_x_all: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int8))
    final_n_sat_all: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    # optional full assignment trajectory (recorded if cfg.record_assignment_every > 0)
    history_x: Optional[np.ndarray] = None  # (T, n_chains, n) float in (0,1)

    # final per-clause status (best chain)
    sat_mask: Optional[np.ndarray] = None

    # config snapshot
    config: dict = field(default_factory=dict)
    instance_name: str = ""
    problem_type: str = "SAT"
    n_vars: int = 0


# ============================================================================
# schedules
# ============================================================================
def _schedule(t_norm: np.ndarray, kind: str, v0: float, v1: float, p: float = 1.0) -> np.ndarray:
    """
    Build a monotone schedule v(t) for t_norm ∈ [0,1].
    """
    if kind == "const":
        return np.full_like(t_norm, v1, dtype=np.float64)
    if kind == "lin":
        return v0 + (v1 - v0) * t_norm
    if kind == "log":
        # Log-cooling:  v0 + (v1-v0) * log(1+α t) / log(1+α)  with α=10
        alpha = 10.0
        x = np.log1p(alpha * t_norm) / np.log1p(alpha)
        return v0 + (v1 - v0) * x
    if kind == "poly":
        return v0 + (v1 - v0) * np.power(t_norm, p)
    raise ValueError(f"unknown schedule kind: {kind}")


# ============================================================================
# the solver
# ============================================================================
class CNLMLangevinSolver:
    """
    Vectorised, multi-chain CNLM-Langevin (fast-slow) solver.

    Parameters
    ----------
    instance : SATInstance | MaxSATInstance
    config   : SolverConfig
    """

    def __init__(
        self,
        instance: Union[SATInstance, MaxSATInstance],
        config: Optional[SolverConfig] = None,
    ):
        self.instance = instance
        self.config = config or SolverConfig()
        self._rng = np.random.default_rng(self.config.seed)

        # Detect problem type and per-clause weights
        if isinstance(instance, MaxSATInstance):
            self.problem_type = "MaxSAT"
            self.weights = instance.effective_weights(hard_scale=self.config.hard_scale).astype(np.float64)
            self.is_hard = instance.is_hard.copy()
        elif isinstance(instance, SATInstance):
            self.problem_type = "SAT"
            self.weights = np.ones(instance.n_clauses, dtype=np.float64)
            self.is_hard = np.ones(instance.n_clauses, dtype=bool)
        else:
            raise TypeError(f"Unsupported instance type: {type(instance)}")

        self.n = instance.n_vars
        self.m = instance.n_clauses
        self.L = instance.L
        self.n_neg = instance.n_neg.astype(np.float64)
        self.width = instance.width

        # offset b_j = |N_j| - 1 + ε
        self.b = self.n_neg - 1.0 + float(self.config.eps)

    # ------------------------------------------------------------------
    # core kernel: vectorised score, energy, gradient over K parallel chains
    # ------------------------------------------------------------------
    def _scores(self, sigma_z: np.ndarray) -> np.ndarray:
        """
        sigma_z : (K, n)
        returns s : (K, m)
        s_{k,j} = sum_i L[j,i] sigma_z[k,i] + b_j
        """
        # (K, n) @ (n, m) => (K, m)
        # sigma_z @ L.T  works for both dense and sparse L (sparse on the right)
        if hasattr(self.L, "toarray"):  # sparse
            # sparse on the left is fastest:  sigma_z @ L.T == (L @ sigma_z.T).T
            tmp = self.L @ sigma_z.T   # (m, K)
            s = tmp.T + self.b[None, :]
        else:
            s = sigma_z @ self.L.T + self.b[None, :]
        return s

    def _free_energy(self, z: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Compute F~_λ(z; c) for each chain plus auxiliary fields.

        z : (K, n), c : (K, m) or (m,) — broadcastable
        Returns F (K,) and a dict with 'sigma_z', 's', 'nu', 'sat_mask'.
        """
        sigma_z = _sigmoid(z)
        s = self._scores(sigma_z)
        c_arr = np.broadcast_to(c, s.shape)
        cs = c_arr * s
        nu = _sigmoid(cs)

        # F~_λ = -Σ_j w_j softplus(c_j s_j) + (λ/2)|z|²
        sp = _softplus_pos(cs)                         # (K, m)
        F = -(self.weights[None, :] * sp).sum(axis=1)
        F = F + 0.5 * self.config.lam * (z * z).sum(axis=1)

        return F, dict(sigma_z=sigma_z, s=s, nu=nu)

    def _grad_z(self, z: np.ndarray, sigma_z: np.ndarray, nu: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        ∇_z F~_λ for each chain (vectorised).

        ∂F/∂z_i = -Σ_j w_j c_j ν_j (∂s_j/∂z_i) + λ z_i
                = -[L^T (w ⊙ c ⊙ ν)]_i · σ'(z_i) + λ z_i

        z : (K, n)
        nu, c : (K, m) (or c broadcastable)
        returns (K, n)
        """
        c_arr = np.broadcast_to(c, nu.shape)
        wcv = self.weights[None, :] * c_arr * nu       # (K, m)

        # (K, m) @ (m, n)
        if hasattr(self.L, "toarray"):                  # sparse path
            # (L.T @ wcv.T).T  = wcv @ L
            LT_wcv = (self.L.T @ wcv.T).T               # (K, n)
        else:
            LT_wcv = wcv @ self.L                       # (K, n)

        sig_prime = _sigmoid_deriv_from_sigma(sigma_z)  # (K, n)
        grad = -LT_wcv * sig_prime + self.config.lam * z
        return grad

    def _grad_rho(self, nu: np.ndarray, s: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        ∇_ρ F~_λ where ρ_j = log c_j (so c_j = e^{ρ_j} and ∂c_j/∂ρ_j = c_j).

        ∂F/∂ρ_j = ∂F/∂c_j · c_j = -w_j ν_j s_j · c_j
        Returns (K, m)
        """
        c_arr = np.broadcast_to(c, nu.shape)
        return -self.weights[None, :] * nu * s * c_arr

    # ------------------------------------------------------------------
    # discrete-state evaluation per chain
    # ------------------------------------------------------------------
    def _eval_assignments(self, x_int: np.ndarray) -> np.ndarray:
        """
        x_int : (K, n) integer/bool
        returns satisfied counts, weighted-sat for MaxSAT, sat_masks (K, m)
        """
        # vectorised:  (L x_int.T).T  in {-something..+something}, sat iff (Lx)_j + |N_j| ≥ 1
        if hasattr(self.L, "toarray"):
            Lx = (self.L @ x_int.T.astype(np.float32)).T   # (K, m)
        else:
            Lx = x_int.astype(np.float32) @ self.L.T
        sat_mask = Lx + self.n_neg[None, :] >= 1.0 - 1e-9
        return sat_mask

    def _objective_score(self, sat_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (n_total_sat per chain, ranking score per chain) where the
        ranking score reflects the actual problem objective:
          * SAT     : raw count of satisfied clauses
          * MaxSAT  : weighted sat count, with hard clauses dominating
        Higher is better.
        """
        K = sat_mask.shape[0]
        n_sat = sat_mask.sum(axis=1).astype(np.int64)

        if self.problem_type == "SAT":
            return n_sat, n_sat.astype(np.float64)

        # MaxSAT: weighted soft + huge bonus for hard satisfaction.
        soft_mask = ~self.is_hard
        hard_mask = self.is_hard
        w = self.instance.weights.astype(np.float64).copy()
        # neutralise non-finite top weight for ranking
        w_safe = w.copy()
        w_safe[~np.isfinite(w_safe)] = 0.0

        # weighted soft score
        soft_score = (sat_mask[:, soft_mask].astype(np.float64) *
                      w_safe[soft_mask][None, :]).sum(axis=1) if soft_mask.any() else np.zeros(K)

        # hard clause penalty (dominates):
        # We add a huge bonus of (n_unsat_hard) * BIG with negative sign
        BIG = max(1.0, w_safe[soft_mask].sum() if soft_mask.any() else 1.0) * 1e6 + 1.0
        n_hard_sat = sat_mask[:, hard_mask].sum(axis=1) if hard_mask.any() else np.zeros(K, dtype=int)
        n_hard_total = int(hard_mask.sum())
        hard_penalty = (n_hard_total - n_hard_sat) * (-BIG)

        rank = soft_score + hard_penalty
        return n_sat, rank

    # ------------------------------------------------------------------
    # main solve
    # ------------------------------------------------------------------
    def solve(self) -> SolveResult:
        cfg = self.config
        K, n, m = cfg.n_chains, self.n, self.m
        rng = self._rng
        t0 = time.perf_counter()

        # ---- schedules
        T = cfg.n_steps
        t_norm = np.linspace(0.0, 1.0, T, dtype=np.float64)
        beta_sched = _schedule(t_norm, cfg.beta_schedule, cfg.beta_init, cfg.beta_final, cfg.beta_poly_p)
        c_sched = _schedule(t_norm, cfg.c_schedule, cfg.c_init, cfg.c_final, cfg.c_poly_p)

        # ---- state initialisation
        z = rng.standard_normal((K, n)).astype(np.float64) * cfg.z_init_scale

        # confidence: deterministic schedule by default; if use_slow_sde, ρ has its own SDE
        if cfg.use_slow_sde:
            rho = np.full((K, m), np.log(cfg.c_init), dtype=np.float64)

        # tracking
        record_every = max(1, int(cfg.record_history_every))
        record_x_every = int(cfg.record_assignment_every)
        n_records = (T + record_every - 1) // record_every

        hist_steps = np.zeros(n_records, dtype=np.int64)
        hist_beta = np.zeros(n_records, dtype=np.float64)
        hist_c_mean = np.zeros(n_records, dtype=np.float64)
        hist_c_min = np.zeros(n_records, dtype=np.float64)
        hist_c_max = np.zeros(n_records, dtype=np.float64)
        hist_F = np.zeros((n_records, K), dtype=np.float64)
        hist_n_sat = np.zeros((n_records, K), dtype=np.int64)
        hist_best = np.zeros(n_records, dtype=np.int64)

        if record_x_every > 0:
            n_x_records = (T + record_x_every - 1) // record_x_every
            hist_x = np.zeros((n_x_records, K, n), dtype=np.float32)
            x_idx = 0
        else:
            hist_x = None

        best_n_sat_chain = np.zeros(K, dtype=np.int64)
        best_x_chain = np.zeros((K, n), dtype=np.int8)
        best_rank_chain = np.full(K, -np.inf, dtype=np.float64)
        steps_since_improve = np.zeros(K, dtype=np.int64)

        global_best_step: Optional[int] = None
        global_best_rank = -np.inf

        rec = 0
        for t in range(T):
            beta = float(beta_sched[t])

            if cfg.use_slow_sde:
                c_arr = np.exp(rho)                                    # (K, m)
            else:
                c_scalar = float(c_sched[t])
                c_arr = np.full(m, c_scalar, dtype=np.float64)         # broadcast (m,)

            # ----- gradient evaluation (single batched call)
            F, aux = self._free_energy(z, c_arr if cfg.use_slow_sde else c_arr)
            sigma_z = aux["sigma_z"]
            s = aux["s"]
            nu = aux["nu"]

            grad_z = self._grad_z(z, sigma_z, nu, c_arr)

            # ----- z step (Euler-Maruyama)
            noise_scale = np.sqrt(2.0 * cfg.dt / max(beta, 1e-12))
            z = z - cfg.dt * grad_z + noise_scale * rng.standard_normal((K, n))

            # clip z to avoid σ saturation totally killing gradients
            np.clip(z, -cfg.z_clip, cfg.z_clip, out=z)

            # ----- ρ step (only if slow SDE active)
            if cfg.use_slow_sde:
                grad_rho = self._grad_rho(nu, s, c_arr)
                noise_rho = np.sqrt(2.0 * cfg.eta * cfg.dt / max(cfg.beta_c, 1e-12))
                rho = rho - cfg.eta * cfg.dt * grad_rho + noise_rho * rng.standard_normal((K, m))
                # clamp confidence to [c_min, c_max]
                np.clip(rho, np.log(cfg.c_min), np.log(cfg.c_max), out=rho)

            # ----- evaluate Boolean rounding for monitoring / best tracking
            x_int = (sigma_z > 0.5).astype(np.int8)
            sat_mask = self._eval_assignments(x_int)
            n_sat_per = sat_mask.sum(axis=1).astype(np.int64)
            _, rank_per = self._objective_score(sat_mask)

            # update best per chain (rank-based)
            improved = rank_per > best_rank_chain
            if improved.any():
                best_rank_chain[improved] = rank_per[improved]
                best_n_sat_chain[improved] = n_sat_per[improved]
                best_x_chain[improved] = x_int[improved]
                steps_since_improve[improved] = 0
            steps_since_improve[~improved] += 1

            # update global best step (for converged_step diagnostic)
            cur_global_max = float(rank_per.max())
            if cur_global_max > global_best_rank + 1e-9:
                global_best_rank = cur_global_max
                global_best_step = t

            # ----- restart logic
            if cfg.restart_on_stuck and cfg.stuck_patience > 0:
                stuck = steps_since_improve >= cfg.stuck_patience
                if stuck.any():
                    n_kick = int(stuck.sum())
                    # fresh random start (preserve best assignment in best_x_chain)
                    z[stuck] = rng.standard_normal((n_kick, n)) * cfg.z_init_scale
                    if cfg.use_slow_sde:
                        rho[stuck] = np.log(cfg.c_init)
                    steps_since_improve[stuck] = 0

            # ----- recording
            if (t % record_every == 0) or (t == T - 1):
                hist_steps[rec] = t
                hist_beta[rec] = beta
                if cfg.use_slow_sde:
                    c_eff = np.exp(rho).mean(axis=0)
                else:
                    c_eff = c_arr
                hist_c_mean[rec] = float(c_eff.mean())
                hist_c_min[rec] = float(c_eff.min())
                hist_c_max[rec] = float(c_eff.max())
                hist_F[rec] = F
                hist_n_sat[rec] = n_sat_per
                hist_best[rec] = int(best_n_sat_chain.max())
                rec += 1

            if hist_x is not None and (t % record_x_every == 0):
                hist_x[x_idx] = sigma_z.astype(np.float32)
                x_idx += 1

            # ----- early-stop on full SAT (only meaningful for SAT)
            if cfg.early_stop_when_sat and self.problem_type == "SAT":
                if best_n_sat_chain.max() == m:
                    # truncate histories
                    hist_steps = hist_steps[:rec]
                    hist_beta = hist_beta[:rec]
                    hist_c_mean = hist_c_mean[:rec]
                    hist_c_min = hist_c_min[:rec]
                    hist_c_max = hist_c_max[:rec]
                    hist_F = hist_F[:rec]
                    hist_n_sat = hist_n_sat[:rec]
                    hist_best = hist_best[:rec]
                    if hist_x is not None:
                        hist_x = hist_x[:x_idx]
                    break

            if cfg.verbose and (t % max(1, T // 10) == 0):
                print(
                    f"step {t:5d}/{T} | β={beta:7.2f} | "
                    f"c̄={hist_c_mean[max(0, rec-1)]:7.2f} | "
                    f"best n_sat = {best_n_sat_chain.max()}/{m}"
                )

        # ---- finalisation: pick best chain
        # final evaluation using the best (snapshotted) assignments
        sat_mask_final = self._eval_assignments(best_x_chain)
        n_sat_final, rank_final = self._objective_score(sat_mask_final)
        best_idx = int(np.argmax(rank_final))
        best_x = best_x_chain[best_idx]
        best_n_sat = int(n_sat_final[best_idx])
        best_sat_mask = sat_mask_final[best_idx]

        # build SolveResult
        runtime = time.perf_counter() - t0
        result = SolveResult(
            assignment=best_x.astype(np.int8),
            is_SAT=False,
            n_satisfied=best_n_sat,
            n_clauses=m,
            sat_score=best_n_sat / max(1, m),
            best_chain=best_idx,
            n_chains=K,
            n_steps=hist_steps[-1] + 1 if rec > 0 else 0,
            runtime_s=runtime,
            converged_step=global_best_step,
            history_steps=hist_steps[:rec],
            history_beta=hist_beta[:rec],
            history_c_mean=hist_c_mean[:rec],
            history_c_min=hist_c_min[:rec],
            history_c_max=hist_c_max[:rec],
            history_free_energy=hist_F[:rec],
            history_n_sat=hist_n_sat[:rec],
            history_best_n_sat=hist_best[:rec],
            final_x_all=best_x_chain.copy(),
            final_n_sat_all=n_sat_final.copy(),
            history_x=hist_x,
            sat_mask=best_sat_mask,
            config=asdict(self.config),
            instance_name=getattr(self.instance, "name", ""),
            problem_type=self.problem_type,
            n_vars=self.n,
        )

        # populate problem-specific fields
        if self.problem_type == "SAT":
            result.is_SAT = best_n_sat == m
        else:
            ev = self.instance.evaluate(best_x.astype(np.int8))
            result.is_SAT = ev["is_SAT_hard"]
            result.cost = ev["cost"]
            result.soft_weight_satisfied = ev["soft_weight_satisfied"]
            result.n_hard_sat = ev["n_hard_sat"]
            result.n_hard_total = ev["n_hard_total"]
            result.n_soft_sat = ev["n_soft_sat"]
            result.n_soft_total = ev["n_soft_total"]

        return result
