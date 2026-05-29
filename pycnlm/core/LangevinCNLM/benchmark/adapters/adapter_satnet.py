"""
SATNet-style SDP coordinate descent (Mixing Method) — pure-NumPy CPU port.

This adapter uses the **non-learned** form of SATNet: we set the clause-
membership matrix directly from the CNF (no learning), and run the
SDP mixing-method coordinate descent (Wang, Chang, Kolter 2019,
"The Mixing Method") to obtain an approximate MAXSAT solution.

This means the adapter works WITHOUT the satnet C++/CUDA extension,
without GPU, and without any pretrained weights.  It is a faithful
representative of the SATNet *core algorithm* applied as a stand-alone
solver, and serves as a meaningful "neural-flavoured" baseline.

If you have ``satnet`` installed (`pip install satnet`) and want to use
the official optimised CUDA implementation, use
``SATNetOfficialAdapter`` (see the same file at the bottom).

Reference:
    P.-W. Wang, P. L. Donti, B. Wilder, J. Z. Kolter.
    "SATNet: Bridging deep learning and logical reasoning using a
    differentiable satisfiability solver." ICML 2019.

Code at: https://github.com/locuslab/SATNet
"""
from __future__ import annotations

import time
import numpy as np

from .base import BaseAdapter, SolveOutcome


def _mixing_method_sdp_maxsat(
    S: np.ndarray,            # (n+1, m) signed clause-membership; row 0 = "true"
    weights: np.ndarray,      # (m,) clause weights, ≥ 0
    k: int = 32,              # rank of the SDP relaxation
    max_iter: int = 200,
    tol: float = 1e-3,
    seed: int = 0,
) -> np.ndarray:
    """
    Solve the MAXSAT SDP relaxation by the mixing method.

    Variables live as unit vectors v_i ∈ S^{k-1}, i = 0..n.  The objective
    (weighted relaxed indicator that each clause is satisfied) reduces to
    iterative coordinate descent updates:

        v_i ← -ω · g_i / ‖g_i‖,   g_i = Σ_j w_j (S[i,j] · Σ_{ℓ≠i} S[ℓ,j] v_ℓ)

    The ω constant is rolled into the normalisation.  After convergence,
    each variable is rounded against a random hyperplane through v_0.
    """
    rng = np.random.default_rng(seed)
    nplus1, m = S.shape
    V = rng.normal(size=(nplus1, k))
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12

    # pre-compute weighted columns
    Sw = S * weights[None, :]            # (n+1, m)

    for it in range(max_iter):
        max_diff = 0.0
        # update each variable's vector once per sweep (Gauss-Seidel)
        # P[j] = Σ_i S[i,j] v_i  (current relaxed-clause vector)
        P = S.T @ V                       # (m, k)
        for i in range(nplus1):
            # remove i's contribution
            Pi = P - np.outer(S[:, :].T @ np.eye(nplus1)[i], np.zeros(k))
            # cleaner: explicit
            contrib = np.outer(S[i, :], V[i])     # (m, k)
            Pi = P - contrib
            g = Sw[i, :] @ Pi                     # (k,)
            norm = np.linalg.norm(g)
            if norm < 1e-12:
                continue
            v_new = -g / norm
            max_diff = max(max_diff, float(np.linalg.norm(V[i] - v_new)))
            V[i] = v_new
            # rebuild P with the updated row to maintain G-S correctness
            P = Pi + np.outer(S[i, :], V[i])

        if max_diff < tol:
            break

    # round: for each variable v_i (i=1..n) measure sign(v_i · v_0) under a
    # random hyperplane.  Standard randomised rounding for the SDP.
    v0 = V[0]
    # one shot: best of several random hyperplanes — improves rounding
    best_x = None
    best_obj = -np.inf
    for _ in range(20):
        h = rng.normal(size=k)
        sign_v0 = np.sign(V[0] @ h)
        signs = np.sign(V[1:, :] @ h)
        x = (signs * sign_v0 > 0)        # i is True iff v_i and v_0 land same side
        # objective = Σ_j w_j · 1[clause satisfied]
        x_full = np.concatenate([[True], x])  # add x_0 = True
        # clause j satisfied if Σ_i S[i,j] x_i has at least one matching positive literal
        # but for a bare CNF this is: clause sat iff for some i with S[i,j]=+1: x_i,
        # or some i with S[i,j]=-1: ¬x_i.  Vectorised:
        # Σ_i max(0, S[i,j] · (2x_i-1))  positive count → sat iff > 0
        signed_x = (2 * x_full.astype(int) - 1)         # ±1
        match = ((S * signed_x[:, None]) > 0).any(axis=0)   # (m,)
        obj = float(weights[match].sum())
        if obj > best_obj:
            best_obj = obj
            best_x = x.copy()

    return best_x.astype(bool)


def _build_S_for_instance(instance) -> np.ndarray:
    """
    Build the SATNet (n+1, m) signed clause matrix.  Row 0 corresponds to a
    fixed `true` variable; rows 1..n to actual variables.  Column j has +1
    in row i if literal +x_i appears in clause j, -1 if -x_i appears, 0
    otherwise.  (Row 0 is left zero — we don't add the SATNet "true" coupling.)
    """
    L = instance.L
    if hasattr(L, "toarray"):
        L_dense = np.asarray(L.toarray(), dtype=float)
    else:
        L_dense = np.asarray(L, dtype=float)
    n, m = instance.n_vars, instance.n_clauses
    S = np.zeros((n + 1, m), dtype=float)
    S[1:, :] = L_dense.T            # L is (m, n) ⇒ L.T is (n, m)
    return S


class SATNetSDPAdapter(BaseAdapter):
    """
    SATNet's mixing-method MAXSAT SDP, in pure NumPy (no training, no CUDA).
    """
    name = "satnet_sdp_numpy"
    kind = "neural"      # neural-flavoured: same algorithm as the SATNet layer
    supports = {"SAT", "MaxSAT"}

    def __init__(self, k: int = 32, max_iter: int = 200, tol: float = 1e-3,
                 seed: int = 0, n_restarts: int = 4, **kwargs):
        super().__init__(k=k, max_iter=max_iter, tol=tol, seed=seed,
                         n_restarts=n_restarts, **kwargs)

    def available(self) -> bool:
        return True

    def solve(self, instance, timeout_s: float = 120.0) -> SolveOutcome:
        from cnlm_langevin.core.instance import MaxSATInstance
        is_max = isinstance(instance, MaxSATInstance)
        n, m = instance.n_vars, instance.n_clauses
        S = _build_S_for_instance(instance)

        if is_max:
            # for MaxSAT, scale up hard clauses
            w_hard = 1e3 * float(instance.weights[~instance.is_hard].max() if (~instance.is_hard).any() else 1.0)
            w = np.where(instance.is_hard, w_hard, instance.weights).astype(np.float64)
        else:
            w = np.ones(m, dtype=np.float64)

        deadline = time.perf_counter() + timeout_s
        best_n_sat = -1
        best_x = None
        n_restarts = int(self.config["n_restarts"])

        for r in range(n_restarts):
            if time.perf_counter() >= deadline:
                break
            try:
                x = _mixing_method_sdp_maxsat(
                    S, w,
                    k=int(self.config["k"]),
                    max_iter=int(self.config["max_iter"]),
                    tol=float(self.config["tol"]),
                    seed=int(self.config["seed"]) + r,
                )
            except Exception as exc:
                return SolveOutcome(
                    solver=self.name, instance=getattr(instance, "name", ""),
                    problem_type=("MaxSAT" if is_max else "SAT"),
                    available=True, error=f"{type(exc).__name__}: {exc}",
                )
            n_sat, _, _, _, _ = self._verify(x, instance)
            if n_sat > best_n_sat:
                best_n_sat = n_sat
                best_x = x.copy()
            if best_n_sat == m:
                break

        if best_x is None:
            best_x = np.zeros(n, dtype=bool)
        n_sat, sat_mask, h_sat, s_sat, cost = self._verify(best_x, instance)

        out = SolveOutcome(
            solver=self.name, instance=getattr(instance, "name", ""),
            problem_type=("MaxSAT" if is_max else "SAT"),
            available=True,
            runtime_s=time.perf_counter() - (deadline - timeout_s),
            timed_out=False,
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
        out.extras = {"k": self.config["k"], "n_restarts": n_restarts}
        return out


class SATNetOfficialAdapter(BaseAdapter):
    """
    Wrapper around the official ``satnet`` PyPI package (locuslab/SATNet).
    Requires CUDA + a working `satnet` install.  Used in inference-only
    mode by setting the SATNet weight matrix S directly from the CNF.
    """
    name = "satnet_official"
    kind = "neural"
    supports = {"SAT", "MaxSAT"}

    def __init__(self, max_iter: int = 100, **kwargs):
        super().__init__(max_iter=max_iter, **kwargs)

    def available(self) -> bool:
        try:
            import torch                                # noqa: F401
            import satnet                               # noqa: F401
            if not __import__("torch").cuda.is_available():
                self.unavailable_reason = (
                    "satnet imports OK but CUDA is unavailable (the official "
                    "build has only a partial CPU path; use satnet_sdp_numpy)"
                )
                return False
            return True
        except Exception as exc:
            self.unavailable_reason = (
                f"satnet not installed (`pip install satnet` and CUDA toolkit "
                f"required): {exc}"
            )
            return False

    def solve(self, instance, timeout_s: float = 120.0) -> SolveOutcome:
        import torch
        import satnet
        from cnlm_langevin.core.instance import MaxSATInstance

        is_max = isinstance(instance, MaxSATInstance)
        n, m = instance.n_vars, instance.n_clauses

        # Build the SATNet S matrix (n+1, m).  Use the same convention as
        # _build_S_for_instance.
        S_np = _build_S_for_instance(instance)
        device = torch.device("cuda")
        S = torch.from_numpy(S_np).float().to(device)

        # SATNet expects an `nn.Module` — we instantiate a layer with
        # n input bits and use S directly as the (frozen) weight.
        layer = satnet.SATNet(n=n, m=m, aux=0).to(device)
        with torch.no_grad():
            # SATNet's weight is the `S` parameter; shape (n+aux+1, m)
            layer.S.copy_(S)

        # forward pass on a single all-0.5 input (no known bits)
        z = torch.full((1, n), 0.5, device=device)
        is_input = torch.zeros((1, n), dtype=torch.int, device=device)

        t0 = time.perf_counter()
        try:
            out_z = layer(z, is_input)
        except Exception as exc:
            return SolveOutcome(
                solver=self.name, instance=getattr(instance, "name", ""),
                problem_type=("MaxSAT" if is_max else "SAT"),
                available=True, error=f"{type(exc).__name__}: {exc}",
                runtime_s=time.perf_counter() - t0,
            )
        x = (out_z[0].detach().cpu().numpy() > 0.5).astype(bool)
        runtime = time.perf_counter() - t0

        n_sat, _, h_sat, s_sat, cost = self._verify(x, instance)
        out = SolveOutcome(
            solver=self.name, instance=getattr(instance, "name", ""),
            problem_type=("MaxSAT" if is_max else "SAT"),
            available=True, runtime_s=runtime,
            is_SAT=(n_sat == m if not is_max else (h_sat == int(instance.is_hard.sum()))),
            n_satisfied=n_sat, n_clauses=m,
            sat_score=n_sat / max(m, 1),
            assignment=x.astype(int).tolist(),
        )
        if is_max:
            out.cost = cost
            out.n_hard_sat = h_sat
            out.n_hard_total = int(instance.is_hard.sum())
            out.n_soft_sat = s_sat
            out.n_soft_total = int((~instance.is_hard).sum())
        return out
