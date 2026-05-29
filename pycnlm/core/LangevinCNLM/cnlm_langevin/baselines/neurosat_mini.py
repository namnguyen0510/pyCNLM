"""
NeuroSAT-mini — a compact in-package re-implementation of NeuroSAT
(Selsam et al., ICLR 2019), small enough to be trained from scratch on
CPU in a few minutes.

Architecture: classic NeuroSAT bipartite message-passing on the
literal-clause graph.

  * Each literal ℓ has an embedding L[ℓ] ∈ R^d
  * Each clause j has an embedding C[j] ∈ R^d
  * For T iterations:
      C[j]   ← GRU_C(  C[j],  Σ_{ℓ ∈ j} MLP_LtoC(L[ℓ]) )
      L[ℓ]  ← GRU_L(  L[ℓ], [ Σ_{j ∋ ℓ} MLP_CtoL(C[j]),  L[ℓ_flip] ] )
  * After T iters:  vote_i = mean( MLP_vote(L[2i]) , -MLP_vote(L[2i+1]) )
                    p_i   = sigmoid( vote_i )
                    x_i   = round( p_i )

Trained with the **assignment-supervision** objective: BCE between the
predicted p_i and a planted satisfying assignment.
"""
from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

WEIGHTS_DIR = Path(__file__).parent / "weights"
DEFAULT_CKPT = WEIGHTS_DIR / "neurosat_mini.pt"


# -------------------------------------------------------------- model
class NeuroSATMini(nn.Module):
    def __init__(self, dim: int = 64, n_iter: int = 16, mlp_hidden: int = 64):
        super().__init__()
        self.dim = dim
        self.n_iter = n_iter

        # learned initial states for literals and clauses
        self.L_init = nn.Parameter(torch.randn(1, dim) * 0.1)
        self.C_init = nn.Parameter(torch.randn(1, dim) * 0.1)

        # message MLPs
        self.L2C_msg = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, dim),
        )
        self.C2L_msg = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, dim),
        )

        # GRU updates (NeuroSAT uses LayerNormBasicLSTMCell; GRU is simpler & fine)
        self.C_update = nn.GRUCell(dim, dim)
        self.L_update = nn.GRUCell(2 * dim, dim)   # in: msg_from_C ⊕ flip-state

        # vote MLP at the end → scalar per literal
        self.vote = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, M_pos: torch.Tensor, M_neg: torch.Tensor):
        """
        Forward on a single CNF instance (no batching for simplicity).

        M_pos  shape (m, n)  — 1 if variable i appears positively in clause j
        M_neg  shape (m, n)  — 1 if variable i appears negatively in clause j

        Literal layout: index 2i = positive lit of var i, 2i+1 = negative lit.
        Clauses connect to either positive or negative literals via M_pos/M_neg.

        Returns p ∈ R^n   (predicted prob. that x_i = True)
        """
        device = M_pos.device
        m, n = M_pos.shape

        L = self.L_init.expand(2 * n, self.dim).contiguous()  # (2n, d)
        C = self.C_init.expand(m, self.dim).contiguous()      # (m, d)

        # build the (m, 2n) literal–clause adjacency once
        # column 2i is the positive literal of var i, column 2i+1 the negative
        A = torch.zeros(m, 2 * n, device=device, dtype=M_pos.dtype)
        A[:, 0::2] = M_pos
        A[:, 1::2] = M_neg

        for _ in range(self.n_iter):
            # 1) clauses receive sums of literal messages
            L_msgs = self.L2C_msg(L)                # (2n, d)
            C_in = A @ L_msgs                       # (m, d)
            C = self.C_update(C_in, C)

            # 2) literals receive clause messages + their flipped state
            C_msgs = self.C2L_msg(C)                # (m, d)
            L_from_C = A.t() @ C_msgs               # (2n, d)
            L_flip = L.clone()
            L_flip[0::2], L_flip[1::2] = L[1::2].clone(), L[0::2].clone()
            L_in = torch.cat([L_from_C, L_flip], dim=1)
            L = self.L_update(L_in, L)

        # vote: pos lit votes for "True", neg lit votes for "False"
        scores_pos = self.vote(L[0::2]).squeeze(-1)   # (n,)
        scores_neg = self.vote(L[1::2]).squeeze(-1)   # (n,)
        logit = scores_pos - scores_neg               # (n,)
        return torch.sigmoid(logit)


# ---------------------------------------------------------- data utils
def random_3sat(n_vars: int, n_clauses: int, rng: np.random.Generator):
    """Random 3-SAT with a planted satisfying assignment."""
    plant = rng.integers(0, 2, n_vars).astype(bool)
    M_pos = np.zeros((n_clauses, n_vars), dtype=np.float32)
    M_neg = np.zeros((n_clauses, n_vars), dtype=np.float32)
    for j in range(n_clauses):
        idx = rng.choice(n_vars, 3, replace=False)
        sat_already = False
        signs = []
        for v in idx:
            sg = int(rng.integers(0, 2))
            signs.append(sg)
            if (sg == 1 and plant[v]) or (sg == 0 and not plant[v]):
                sat_already = True
        if not sat_already:
            # force first literal to satisfy
            v = idx[0]
            signs[0] = 1 if plant[v] else 0
        for v, sg in zip(idx, signs):
            if sg == 1:
                M_pos[j, v] = 1.0
            else:
                M_neg[j, v] = 1.0
    return M_pos, M_neg, plant


# ----------------------------------------------------------- training
def _soft_sat_loss(p: torch.Tensor, M_pos: torch.Tensor, M_neg: torch.Tensor,
                   eps: float = 1e-6) -> torch.Tensor:
    """
    Differentiable surrogate for "fraction of clauses satisfied" given
    variable probabilities p ∈ (0,1)^n.

    For each clause j:
        P(j sat) = 1 - Π_{i ∈ pos(j)} (1 - p_i) · Π_{i ∈ neg(j)} p_i
    Loss = - mean_j log( P(j sat) ).
    """
    # log_unsat[j] = Σ_{i pos} log(1 - p_i) + Σ_{i neg} log(p_i)
    log_unsat = (M_pos * torch.log1p(-p + eps).unsqueeze(0)).sum(dim=1) \
              + (M_neg * torch.log(p + eps).unsqueeze(0)).sum(dim=1)
    # log P(sat) = log(1 - exp(log_unsat))    — stable form
    # log1mexp(x) for x < 0
    log_unsat = torch.clamp(log_unsat, max=-eps)
    log_sat = torch.log1p(-torch.exp(log_unsat))
    return -log_sat.mean()


# ----------------------------------------------------------- training
def train_neurosat_mini(
    n_iters_train: int = 2500,
    n_vars_range=(8, 18),
    clause_var_ratio: float = 4.2,
    dim: int = 64,
    n_msg_iters: int = 16,
    lr: float = 2e-3,
    seed: int = 0,
    device: str = "cpu",
    save_path: Path = DEFAULT_CKPT,
    verbose: bool = True,
):
    """Train NeuroSAT-mini on synthetic 3-SAT with the unsupervised soft-SAT loss."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    model = NeuroSATMini(dim=dim, n_iter=n_msg_iters).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses, accs = [], []

    model.train()
    for step in range(1, n_iters_train + 1):
        n_vars = int(rng.integers(n_vars_range[0], n_vars_range[1] + 1))
        n_clauses = int(round(clause_var_ratio * n_vars))
        Mp, Mn, plant = random_3sat(n_vars, n_clauses, rng)
        Mp_t = torch.from_numpy(Mp).to(device)
        Mn_t = torch.from_numpy(Mn).to(device)
        p = model(Mp_t, Mn_t).clamp(1e-6, 1.0 - 1e-6)
        loss = _soft_sat_loss(p, Mp_t, Mn_t)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))

        # also track real # satisfied clauses with rounded p
        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                x = (p > 0.5).cpu().numpy().astype(bool)
                # how many clauses satisfied?
                n_sat = 0
                for j in range(n_clauses):
                    pos_lits = np.where(Mp[j] == 1.0)[0]
                    neg_lits = np.where(Mn[j] == 1.0)[0]
                    if any(x[v] for v in pos_lits) or any(not x[v] for v in neg_lits):
                        n_sat += 1
                acc = n_sat / max(n_clauses, 1)
            accs.append(acc)
            if verbose:
                recent = float(np.mean(losses[-100:]))
                print(f"  step {step:5d}/{n_iters_train}: loss(100)={recent:.4f}  "
                      f"sat={n_sat}/{n_clauses} ({acc:.3f})  "
                      f"n={n_vars} m={n_clauses}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "config": {"dim": dim, "n_msg_iters": n_msg_iters,
                   "n_iters_train": n_iters_train,
                   "n_vars_range": list(n_vars_range),
                   "clause_var_ratio": clause_var_ratio,
                   "loss": "soft_sat"},
        "losses": losses,
        "accs": accs,
    }, save_path)
    if verbose:
        print(f"\nsaved checkpoint -> {save_path}  ({save_path.stat().st_size/1024:.1f} KB)")
    return model, losses


def load_neurosat_mini(ckpt_path: Path = DEFAULT_CKPT, device: str = "cpu"):
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"NeuroSAT-mini checkpoint not found at {ckpt_path}. "
            f"Run `python -m cnlm_langevin.baselines.neurosat_mini` to train."
        )
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["config"]
    model = NeuroSATMini(dim=cfg["dim"], n_iter=cfg["n_msg_iters"]).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def predict_assignment(model: NeuroSATMini, instance, device: str = "cpu",
                       n_random_rounds: int = 8) -> np.ndarray:
    """Run NeuroSAT-mini on a SATInstance, return a Boolean assignment.
    Tries a small number of random roundings of the predicted probabilities
    and keeps the one satisfying the most clauses.
    """
    n = instance.n_vars
    m = instance.n_clauses
    L = instance.L
    if hasattr(L, "toarray"):
        L_dense = np.asarray(L.toarray(), dtype=np.float32)
    else:
        L_dense = np.asarray(L, dtype=np.float32)

    M_pos = (L_dense == 1.0).astype(np.float32)
    M_neg = (L_dense == -1.0).astype(np.float32)
    Mp = torch.from_numpy(M_pos).to(device)
    Mn = torch.from_numpy(M_neg).to(device)
    with torch.no_grad():
        p = model(Mp, Mn).cpu().numpy()

    # try multiple roundings: round + small noise
    from cnlm_langevin.core.instance import evaluate_clauses_bool_vectorized
    rng = np.random.default_rng(0)
    best_x = (p > 0.5).astype(bool)
    best_n = int(evaluate_clauses_bool_vectorized(
        instance.L, np.asarray(instance.n_neg), best_x).sum())
    for _ in range(n_random_rounds):
        x = (p + rng.normal(0, 0.15, size=p.shape) > 0.5).astype(bool)
        n_sat = int(evaluate_clauses_bool_vectorized(
            instance.L, np.asarray(instance.n_neg), x).sum())
        if n_sat > best_n:
            best_n = n_sat
            best_x = x
    return best_x


if __name__ == "__main__":
    # train and save when run directly
    train_neurosat_mini()
