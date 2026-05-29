"""
Visualization helpers for CNLM-Langevin (fast-slow) solver output.

All plot functions accept a SolveResult (and sometimes the instance)
and write a PDF (or return the matplotlib Figure).  Style mirrors the
paper figures.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, List

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from .dynamics import SolveResult
from .instance import SATInstance, MaxSATInstance


# ---------------------------------------------------------------------- style
PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.4,
    "axes.grid": True,
    "grid.alpha": 0.25,
}

COL_SOFT = "#2E5EAA"
COL_HARD = "#C0392B"
COL_TRUE = "#1E8449"
COL_FALSE = "#A93226"
COL_NEU = "#7B7B9D"
COL_GOLD = "#D4AC0D"


def _apply_style():
    mpl.rcParams.update(PAPER_STYLE)


# =========================================================================== 1
def plot_assignment_trajectory(
    result: SolveResult,
    save: Optional[Union[str, Path]] = None,
    chain: Optional[int] = None,
    max_vars: int = 30,
):
    """
    Plot x_i(t) = σ(z_i(t)) for the best chain (or any chosen chain).
    Requires the solver to have been run with record_assignment_every > 0.
    """
    _apply_style()
    if result.history_x is None or result.history_x.size == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "(no x trajectory recorded)\nset record_assignment_every>0",
                transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        if save:
            fig.savefig(save)
        return fig

    if chain is None:
        chain = int(result.best_chain)
    n_steps_rec, n_chains, n = result.history_x.shape
    n_show = min(n, max_vars)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    cmap = cm.viridis(np.linspace(0.1, 0.9, n_show))
    t_axis = np.arange(n_steps_rec)
    for i in range(n_show):
        ax.plot(t_axis, result.history_x[:, chain, i], color=cmap[i], lw=1.0, alpha=0.85)
    ax.axhline(0.5, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("recorded step")
    ax.set_ylabel(r"$x_i(t)=\sigma(z_i(t))$")
    title = (f"(a) Continuous embedding trajectory — chain {chain}"
             + (f"  (showing {n_show}/{n} vars)" if n_show < n else ""))
    ax.set_title(title, weight="bold")
    if save:
        fig.savefig(save)
    return fig


# =========================================================================== 2
def plot_energy_curve(result: SolveResult, save: Optional[Union[str, Path]] = None):
    """Free-energy F~_λ over time, all chains, plus mean± std."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    F = result.history_free_energy
    if F.size == 0:
        ax.text(0.5, 0.5, "no history", transform=ax.transAxes, ha="center")
        if save: fig.savefig(save)
        return fig
    steps = result.history_steps
    K = F.shape[1]
    cmap = cm.viridis(np.linspace(0.1, 0.85, K))
    for k in range(K):
        ax.plot(steps, F[:, k], color=cmap[k], lw=0.7, alpha=0.4)
    ax.plot(steps, F.mean(axis=1), color=COL_HARD, lw=1.8, label=r"mean over chains")
    ax.fill_between(steps, F.mean(axis=1) - F.std(axis=1),
                    F.mean(axis=1) + F.std(axis=1),
                    color=COL_HARD, alpha=0.15)
    ax.plot(steps, F.min(axis=1), color="black", lw=1.0, ls="--", label="best chain")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\widetilde{F}_\lambda(\mathbf{z}_t;\mathbf{c}_t)$")
    ax.set_title("(b) CNLM-Langevin free energy along the SDE", weight="bold")
    ax.legend(loc="upper right")
    if save:
        fig.savefig(save)
    return fig


# =========================================================================== 3
def plot_clause_satisfaction(result: SolveResult, save: Optional[Union[str, Path]] = None):
    """Number of satisfied clauses over time: per-chain + best ever + total."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    if result.history_n_sat.size == 0:
        ax.text(0.5, 0.5, "no history", transform=ax.transAxes, ha="center")
        if save: fig.savefig(save)
        return fig
    steps = result.history_steps
    H = result.history_n_sat
    K = H.shape[1]
    cmap = cm.plasma(np.linspace(0.1, 0.85, K))
    for k in range(K):
        ax.plot(steps, H[:, k], color=cmap[k], lw=0.7, alpha=0.45)
    ax.plot(steps, result.history_best_n_sat, color=COL_TRUE, lw=2.2, label="running best")
    ax.axhline(result.n_clauses, color="black", lw=0.7, ls=":", label=f"all clauses ({result.n_clauses})")
    ax.set_xlabel("step")
    ax.set_ylabel("# satisfied clauses")
    title = f"(c) Clause satisfaction trajectory  —  final {result.n_satisfied}/{result.n_clauses}"
    ax.set_title(title, weight="bold")
    ax.legend(loc="lower right")
    if save:
        fig.savefig(save)
    return fig


# =========================================================================== 4
def plot_confidence_evolution(result: SolveResult, save: Optional[Union[str, Path]] = None):
    """c(t) and β(t) annealing schedules along the run."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    steps = result.history_steps
    if steps.size == 0:
        ax.text(0.5, 0.5, "no history", transform=ax.transAxes, ha="center")
        if save: fig.savefig(save)
        return fig
    ax.plot(steps, result.history_beta, color=COL_SOFT, lw=1.6, label=r"$\beta(t)$")
    ax.plot(steps, result.history_c_mean, color=COL_HARD, lw=1.6, label=r"$\bar c(t)$")
    ax.fill_between(steps, result.history_c_min, result.history_c_max,
                    color=COL_HARD, alpha=0.15, label=r"$c$ chain range")
    ax.set_xlabel("step")
    ax.set_ylabel("annealing parameters")
    ax.set_title(r"(d) Annealing schedule:  $\beta(t)$ and $c(t)$", weight="bold")
    ax.legend(loc="best")
    if save:
        fig.savefig(save)
    return fig


# =========================================================================== 5
def plot_schedule(result: SolveResult, save: Optional[Union[str, Path]] = None):
    """β vs c phase plot (annealing geodesic)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(4.5, 4))
    if result.history_beta.size == 0:
        ax.text(0.5, 0.5, "no history", transform=ax.transAxes, ha="center")
        if save: fig.savefig(save)
        return fig
    ax.plot(result.history_c_mean, result.history_beta,
            color="black", lw=1.6, alpha=0.8)
    ax.scatter(result.history_c_mean[0], result.history_beta[0],
               s=80, marker="o", color=COL_SOFT, edgecolor="black",
               zorder=4, label="start")
    ax.scatter(result.history_c_mean[-1], result.history_beta[-1],
               s=120, marker="*", color=COL_GOLD, edgecolor="black",
               zorder=4, label="end")
    ax.set_xlabel(r"confidence $\bar c$ (log)")
    ax.set_ylabel(r"inverse temperature $\beta$ (log)")
    try:
        ax.set_xscale("log"); ax.set_yscale("log")
    except ValueError:
        pass
    ax.set_title("(e) Annealing geodesic in $(c, \\beta)$", weight="bold")
    ax.legend()
    if save:
        fig.savefig(save)
    return fig


# =========================================================================== 6
def plot_chain_diversity(result: SolveResult, save: Optional[Union[str, Path]] = None):
    """Distribution of final n_sat across chains."""
    _apply_style()
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.4))
    n_sat_all = result.final_n_sat_all
    K = n_sat_all.size

    ax = axs[0]
    if K > 0:
        # histogram of n_sat
        bins = np.arange(int(n_sat_all.min()), int(n_sat_all.max()) + 2) - 0.5
        ax.hist(n_sat_all, bins=bins, color=COL_SOFT, edgecolor="black")
        ax.axvline(result.n_satisfied, color=COL_HARD, lw=2.0, label=f"best={result.n_satisfied}")
        ax.axvline(result.n_clauses, color="black", lw=0.7, ls=":", label=f"max={result.n_clauses}")
        ax.set_xlabel("# satisfied clauses (per chain)")
        ax.set_ylabel("count")
        ax.set_title(f"(f) Chain diversity at final step (K={K})", weight="bold")
        ax.legend()
    else:
        ax.axis("off")

    ax = axs[1]
    # show final assignments as a heatmap
    final = result.final_x_all                      # (K, n)
    n_show = min(60, final.shape[1])
    K_show = min(K, 32)
    if final.size > 0:
        # reorder chains by n_sat desc
        order = np.argsort(-result.final_n_sat_all)[:K_show]
        ax.imshow(final[order, :n_show], aspect="auto",
                  cmap="bwr", interpolation="nearest", vmin=0, vmax=1)
        ax.set_xlabel(f"variable index (showing {n_show}/{final.shape[1]})")
        ax.set_ylabel("chain (sorted by n_sat)")
        ax.set_title("Final assignments per chain", weight="bold")
        ax.grid(False)
    else:
        ax.axis("off")
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


# =========================================================================== 7
def plot_score_distribution(
    result: SolveResult,
    instance,
    save: Optional[Union[str, Path]] = None,
):
    """
    Per-clause: which are satisfied vs not, separated by hardness for MaxSAT.
    Bottom panel: clause-width vs satisfaction.
    """
    _apply_style()
    sat_mask = result.sat_mask
    if sat_mask is None:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no sat mask", transform=ax.transAxes, ha="center")
        ax.axis("off")
        if save: fig.savefig(save)
        return fig

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))

    ax = axs[0]
    counts = [int((~sat_mask).sum()), int(sat_mask.sum())]
    bars = ax.bar(["unsatisfied", "satisfied"], counts,
                  color=[COL_FALSE, COL_TRUE], edgecolor="black")
    for b, v in zip(bars, counts):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(),
                str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("count")
    ax.set_title("(g) Per-clause satisfaction (best chain)", weight="bold")

    ax = axs[1]
    width = instance.width
    if width.size:
        unique_w = np.unique(width)
        sat_per_w = np.zeros(len(unique_w))
        tot_per_w = np.zeros(len(unique_w))
        for i, w in enumerate(unique_w):
            sel = (width == w)
            tot_per_w[i] = sel.sum()
            sat_per_w[i] = sat_mask[sel].sum()
        rate = sat_per_w / np.maximum(1, tot_per_w)
        ax.bar(unique_w, rate, color=COL_SOFT, edgecolor="black")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("clause width $|C_j|$")
        ax.set_ylabel("satisfaction rate")
        ax.set_title("Satisfaction rate by clause width", weight="bold")
    else:
        ax.axis("off")
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


# =========================================================================== 8 (MaxSAT)
def plot_maxsat_breakdown(
    result: SolveResult,
    instance,
    save: Optional[Union[str, Path]] = None,
):
    """Hard vs soft, weighted satisfaction view."""
    _apply_style()
    if result.problem_type != "MaxSAT" or not isinstance(instance, MaxSATInstance):
        fig, ax = plt.subplots(figsize=(5, 2.2))
        ax.text(0.5, 0.5, "MaxSAT-only plot — instance is SAT.",
                transform=ax.transAxes, ha="center")
        ax.axis("off")
        if save: fig.savefig(save)
        return fig

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))

    # left: hard vs soft satisfaction counts
    ax = axs[0]
    cats = []
    sat_v = []
    unsat_v = []
    if result.n_hard_total:
        cats.append("hard")
        sat_v.append(result.n_hard_sat or 0)
        unsat_v.append((result.n_hard_total or 0) - (result.n_hard_sat or 0))
    if result.n_soft_total:
        cats.append("soft")
        sat_v.append(result.n_soft_sat or 0)
        unsat_v.append((result.n_soft_total or 0) - (result.n_soft_sat or 0))
    xpos = np.arange(len(cats))
    ax.bar(xpos, sat_v, color=COL_TRUE, edgecolor="black", label="sat")
    ax.bar(xpos, unsat_v, bottom=sat_v, color=COL_FALSE, edgecolor="black", label="unsat")
    ax.set_xticks(xpos)
    ax.set_xticklabels(cats)
    ax.set_ylabel("clauses")
    ax.set_title("(h) MaxSAT hard/soft satisfaction", weight="bold")
    ax.legend()

    # right: weighted soft cost
    ax = axs[1]
    sw = result.soft_weight_satisfied or 0.0
    cost = result.cost or 0.0
    ax.bar(["weight sat", "cost (unsat)"],
           [sw, cost], color=[COL_TRUE, COL_FALSE], edgecolor="black")
    ax.set_ylabel("weight")
    total = sw + cost
    if total > 0:
        rate = sw / total
        ax.set_title(f"Soft weight  sat={sw:.2f}  cost={cost:.2f}  rate={rate:.3f}",
                     weight="bold")
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


# =========================================================================== summary card
def plot_solution_summary(
    result: SolveResult,
    instance,
    save: Optional[Union[str, Path]] = None,
):
    """A single-figure dashboard with the most important diagnostics."""
    _apply_style()
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.45)

    # title text panel
    ax0 = fig.add_subplot(gs[0, :])
    ax0.axis("off")
    name = result.instance_name or "instance"
    head = (f"CNLM-Langevin (fast-slow)  —  {result.problem_type}  —  {name}\n"
            f"n={result.n_vars}  m={result.n_clauses}  "
            f"satisfied={result.n_satisfied}/{result.n_clauses}  "
            f"score={result.sat_score:.4f}  "
            f"{'SAT' if result.is_SAT else 'UNK'}  "
            f"runtime={result.runtime_s:.2f}s  K={result.n_chains}  steps={result.n_steps}"
    )
    ax0.text(0.0, 0.5, head, fontsize=11, family="monospace", va="center")

    # F over time
    ax1 = fig.add_subplot(gs[1, 0])
    if result.history_free_energy.size:
        steps = result.history_steps
        F = result.history_free_energy
        for k in range(F.shape[1]):
            ax1.plot(steps, F[:, k], lw=0.6, alpha=0.4, color=cm.viridis(k/max(1,F.shape[1]-1)))
        ax1.plot(steps, F.min(axis=1), color=COL_HARD, lw=1.6, label="best")
        ax1.set_xlabel("step"); ax1.set_ylabel(r"$\widetilde F_\lambda$")
        ax1.set_title("Free energy", weight="bold"); ax1.legend(fontsize=8)

    # n_sat over time
    ax2 = fig.add_subplot(gs[1, 1])
    if result.history_n_sat.size:
        ax2.plot(result.history_steps, result.history_best_n_sat, color=COL_TRUE, lw=2)
        ax2.axhline(result.n_clauses, ls=":", color="black", lw=0.7)
        ax2.set_xlabel("step"); ax2.set_ylabel("# sat")
        ax2.set_title("Running best", weight="bold")

    # schedule
    ax3 = fig.add_subplot(gs[1, 2])
    if result.history_beta.size:
        ax3.plot(result.history_steps, result.history_beta, color=COL_SOFT, label=r"$\beta$")
        ax3.plot(result.history_steps, result.history_c_mean, color=COL_HARD, label=r"$\bar c$")
        ax3.set_xlabel("step"); ax3.legend(fontsize=8)
        ax3.set_title("Annealing", weight="bold")

    # chain diversity
    ax4 = fig.add_subplot(gs[2, 0])
    if result.final_n_sat_all.size:
        ax4.hist(result.final_n_sat_all,
                 bins=np.arange(int(result.final_n_sat_all.min()),
                                int(result.final_n_sat_all.max())+2)-0.5,
                 color=COL_SOFT, edgecolor="black")
        ax4.axvline(result.n_satisfied, color=COL_HARD, lw=2)
        ax4.set_xlabel("# sat per chain"); ax4.set_ylabel("count")
        ax4.set_title("Chain spread", weight="bold")

    # per-clause sat vs unsat
    ax5 = fig.add_subplot(gs[2, 1])
    if result.sat_mask is not None:
        cnt = [int((~result.sat_mask).sum()), int(result.sat_mask.sum())]
        ax5.bar(["unsat", "sat"], cnt, color=[COL_FALSE, COL_TRUE], edgecolor="black")
        ax5.set_ylabel("clauses")
        ax5.set_title("Final clause status", weight="bold")

    # MaxSAT breakdown if applicable
    ax6 = fig.add_subplot(gs[2, 2])
    if result.problem_type == "MaxSAT" and result.n_hard_total is not None:
        ax6.bar(["hard sat", "hard unsat", "soft sat", "soft unsat"],
                [result.n_hard_sat or 0,
                 (result.n_hard_total or 0) - (result.n_hard_sat or 0),
                 result.n_soft_sat or 0,
                 (result.n_soft_total or 0) - (result.n_soft_sat or 0)],
                color=[COL_TRUE, COL_FALSE, COL_TRUE, COL_FALSE], edgecolor="black")
        ax6.tick_params(axis='x', rotation=15, labelsize=8)
        cost_str = f"cost={result.cost:.2f}" if result.cost is not None else ""
        ax6.set_title(f"MaxSAT  {cost_str}", weight="bold")
    else:
        ax6.axis("off")
        ax6.text(0.5, 0.5, "(SAT instance)" if result.problem_type == "SAT" else "",
                 transform=ax6.transAxes, ha="center", va="center")

    if save:
        fig.savefig(save)
    return fig


# =========================================================================== save-all
# ============================================================================
# paper-style figures (added in v1.1)
# ============================================================================
def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def _dense_L(L) -> np.ndarray:
    if hasattr(L, "toarray"):
        return np.asarray(L.toarray(), dtype=float)
    return np.asarray(L, dtype=float)


def _interp_c_to_x_axis(result: SolveResult, T_x: int) -> np.ndarray:
    """Interpolate the c(t) schedule onto the recorded-x time axis."""
    h_steps = result.history_steps.astype(float)
    h_c = result.history_c_mean.astype(float)
    if h_steps.size == 0 or T_x <= 0:
        c_final = float(result.config.get("c_final", 60.0))
        return np.full(max(T_x, 1), c_final)
    K_x = max(1, int(result.config.get("record_assignment_every", 1)))
    t_x = np.arange(T_x) * float(K_x)
    return np.interp(t_x, h_steps, h_c)


def plot_clause_veritron_heatmap(
    result: SolveResult,
    instance,
    chain: Optional[int] = None,
    save: Optional[Union[str, Path]] = None,
    max_clauses_in_lineplot: int = 12,
):
    """
    Heatmap of the per-clause Veritron output

        ν_j(t) = σ(c_t · s̃_j(z_t))

    along the recorded trajectory of one chain (default = best chain),
    with an aggregate satisfaction-dynamics strip on the bottom.

    Requires the solver to have been run with ``record_assignment_every > 0``.
    """
    _apply_style()
    if result.history_x is None or result.history_x.size == 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5,
                "no x trajectory recorded\n(set SolverConfig.record_assignment_every > 0)",
                transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        if save:
            fig.savefig(save)
        return fig

    if chain is None:
        chain = int(result.best_chain)

    x_traj = result.history_x[:, chain, :]  # (T, n)
    T, n = x_traj.shape
    L_dense = _dense_L(instance.L)
    n_neg = np.asarray(instance.n_neg, dtype=float)
    eps = float(result.config.get("eps", 0.5))

    # s_j(t) = Σ_i L[j,i] x_i(t) + (n_neg[j] - 1 + ε)
    s_traj = x_traj @ L_dense.T + (n_neg - 1.0 + eps)  # (T, m)
    c_at_x = _interp_c_to_x_axis(result, T)
    nu = _sigmoid(c_at_x[:, None] * s_traj)              # (T, m)

    m = nu.shape[1]
    fig = plt.figure(figsize=(9.0, 5.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.2], hspace=0.20)
    ax_h = fig.add_subplot(gs[0])
    ax_l = fig.add_subplot(gs[1], sharex=ax_h)

    cmap = cm.RdBu_r
    im = ax_h.imshow(
        nu.T,
        aspect="auto", origin="lower", cmap=cmap, vmin=0.0, vmax=1.0,
        extent=[0, T - 1, 0.5, m + 0.5],
        interpolation="nearest",
    )
    ax_h.set_ylabel("CNF clause")
    if m <= 12:
        ax_h.set_yticks(np.arange(1, m + 1))
        ax_h.set_yticklabels([rf"$\varphi_{{{j+1}}}$" for j in range(m)])
    else:
        step = max(1, m // 8)
        ticks = np.arange(1, m + 1, step)
        ax_h.set_yticks(ticks)
        ax_h.set_yticklabels([str(int(t)) for t in ticks])
    ax_h.set_title(
        rf"Per-clause Veritron output $\nu_j(t)=\sigma(c_t\,\tilde s_j(z_t))$  —  chain {chain}",
        weight="bold",
    )
    ax_h.tick_params(labelbottom=False)
    ax_h.grid(False)
    cb = fig.colorbar(im, ax=ax_h, fraction=0.04, pad=0.02)
    cb.set_label(r"$\nu_j$")

    # bottom panel
    t_axis = np.arange(T)
    if m <= max_clauses_in_lineplot:
        for j in range(m):
            col = cmap(0.85) if nu[-1, j] > 0.5 else cmap(0.15)
            ax_l.plot(t_axis, nu[:, j], color=col, lw=0.7, alpha=0.45)
    nu_mean = nu.mean(axis=1)
    ax_l.plot(t_axis, nu_mean, color="black", lw=2.0, label=r"mean $\bar\nu(t)$")

    all_sat = (nu > 0.5).all(axis=1)
    if all_sat.any():
        y0 = -0.06
        ax_l.fill_between(
            t_axis, y0, y0 + 0.05, where=all_sat,
            color=COL_TRUE, alpha=0.85, label=r"all clauses sat. ($\nu>0.5$)",
        )
    ax_l.set_ylim(-0.08, 1.06)
    ax_l.set_xlabel("recorded step")
    ax_l.set_ylabel(r"$\nu_j(t)$  (smoothed)")
    ax_l.set_title("Per-clause satisfaction dynamics", fontsize=10)
    ax_l.legend(loc="lower right", fontsize=8, ncol=2)

    if save:
        fig.savefig(save)
    return fig


def plot_loss_landscape_sweep(
    instance,
    c_values=(0.5, 1.0, 5.0, 10.0),
    project: Optional[tuple] = None,
    anchor: Optional[np.ndarray] = None,
    z_range: float = 6.0,
    grid: int = 60,
    eps: float = 0.5,
    lam: float = 1e-3,
    star: Optional[tuple] = None,
    save: Optional[Union[str, Path]] = None,
):
    """
    Sweep ``c_values`` and plot the lifted free energy F̃(z; c) on a 2-D
    slice of variable space, with -∇F̃ streamlines overlaid.  This visualises
    how the landscape morphs from convex (c small) to a sharp V-basin
    (c large), as in the paper.

    For ``n_vars == 2``, the full landscape is plotted.  For larger n,
    two variables are projected onto (z₁, z₂) and the rest are anchored at
    ``anchor`` (default 0.5, i.e. unit-cube centre).

    Parameters
    ----------
    star : (z1, z2) | None
        coordinates of a "satisfying corner" marker (yellow star).
    """
    _apply_style()
    n = instance.n_vars
    L_dense = _dense_L(instance.L)
    n_neg = np.asarray(instance.n_neg, dtype=float)
    if hasattr(instance, "weights") and instance.weights is not None:
        weights = np.asarray(instance.weights, dtype=float)
    else:
        weights = np.ones(instance.n_clauses, dtype=float)

    if n == 2:
        ii, jj = 0, 1
        anchor_x = None
    else:
        if project is None:
            project = (0, 1)
        ii, jj = project
        anchor_x = (np.full(n, 0.5) if anchor is None
                    else np.asarray(anchor, dtype=float).copy())

    grid_z = np.linspace(-z_range, z_range, grid)
    Z1, Z2 = np.meshgrid(grid_z, grid_z, indexing="xy")
    X1 = _sigmoid(Z1)
    X2 = _sigmoid(Z2)

    def _F_and_grad(c_val: float):
        if n == 2:
            x = np.stack([X1, X2], axis=-1)                          # (G,G,2)
        else:
            x = np.broadcast_to(anchor_x, (grid, grid, n)).copy()
            x[..., ii] = X1
            x[..., jj] = X2
        s = np.einsum("ghn,mn->ghm", x, L_dense) + (n_neg - 1.0 + eps)
        cs = np.clip(c_val * s, -50.0, 50.0)
        # stable softplus
        sp = np.where(cs > 0, cs + np.log1p(np.exp(-cs)),
                              np.log1p(np.exp(cs)))
        F = -(weights * sp).sum(axis=-1) + 0.5 * lam * (Z1 * Z1 + Z2 * Z2)
        nu = _sigmoid(cs)                                            # (G,G,m)
        # ∂F/∂x_a = -Σ_j w_j ν_j c L[j,a]
        dF_dx = -np.einsum("ghm,mn->ghn", weights * nu, c_val * L_dense)
        sigp1 = X1 * (1.0 - X1)
        sigp2 = X2 * (1.0 - X2)
        if n == 2:
            dz1 = dF_dx[..., 0] * sigp1 + lam * Z1
            dz2 = dF_dx[..., 1] * sigp2 + lam * Z2
        else:
            dz1 = dF_dx[..., ii] * sigp1 + lam * Z1
            dz2 = dF_dx[..., jj] * sigp2 + lam * Z2
        return F, -dz1, -dz2  # streamline vector is -∇F

    P = len(c_values)
    fig, axes = plt.subplots(
        1, P, figsize=(3.4 * P + 0.6, 3.4),
        sharex=True, sharey=True, constrained_layout=True,
    )
    if P == 1:
        axes = [axes]

    # global colour limits so panels are comparable
    Fs = []
    for c_val in c_values:
        Fc, _, _ = _F_and_grad(float(c_val))
        Fs.append(Fc)
    vmin = min(F.min() for F in Fs)
    vmax = max(F.max() for F in Fs)

    cmap_f = cm.Reds_r
    im = None
    for ax, c_val, F in zip(axes, c_values, Fs):
        _, U, V = _F_and_grad(float(c_val))
        im = ax.pcolormesh(Z1, Z2, F, cmap=cmap_f, vmin=vmin, vmax=vmax,
                           shading="auto")
        ax.streamplot(Z1, Z2, U, V, color="black", linewidth=0.6,
                      density=1.1, arrowsize=0.9)
        if star is not None:
            ax.plot(star[0], star[1], marker="*", ms=18,
                    color=COL_GOLD, mec="k", mew=0.8, zorder=5)
        ax.set_title(rf"$c={c_val:g}$")
        ax.set_xlabel(r"$z_1$")
        ax.set_xlim(-z_range, z_range)
        ax.set_ylim(-z_range, z_range)
        ax.grid(False)
    axes[0].set_ylabel(r"$z_2$")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015, fraction=0.04)
    cb.set_label(r"$\widetilde F$")
    fig.suptitle(
        r"Loss-landscape geometry sweeps from convex ($c$ small) to "
        r"V-basin ($c$ large) — streamlines show $-\nabla\widetilde F$",
        weight="bold", y=1.04,
    )
    if save:
        fig.savefig(save)
    return fig


def plot_sde_trajectory_2d(
    result: SolveResult,
    instance,
    project: tuple = (0, 1),
    n_panels: int = 3,
    z_range: float = 6.0,
    star: Optional[tuple] = None,
    save: Optional[Union[str, Path]] = None,
):
    """
    Hexbin density of the chain ensemble in (z_i, z_j) space at three
    epochs of the annealed solve (early / mid / late), corresponding
    to growing (β, c).  Yellow star marks a "satisfying corner" if given;
    otherwise inferred from ``result.assignment``.
    """
    _apply_style()
    if result.history_x is None or result.history_x.size == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5,
                "no x trajectory recorded\n(set SolverConfig.record_assignment_every > 0)",
                transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        if save:
            fig.savefig(save)
        return fig

    n = instance.n_vars
    i, j = project
    if i >= n or j >= n:
        raise ValueError(f"project=({i},{j}) out of range for n={n}")

    # (T, K, n) → take only (i, j) → (T, K, 2)
    x_ij = result.history_x[..., [i, j]]
    T, K, _ = x_ij.shape

    # split into n_panels equal time epochs
    edges = np.linspace(0, T, n_panels + 1, dtype=int)

    # work in z-space via logit, with clipping
    def to_z(x):
        x = np.clip(x, 1e-3, 1.0 - 1e-3)
        return np.log(x / (1.0 - x))

    # "satisfying corner" star coordinates — clip into plot box
    if star is None:
        try:
            x_star = result.assignment.astype(float)
            # Boolean → place star at ±(0.85·z_range) so it's visible inside the box
            r = 0.85 * z_range
            star_z = (
                +r if x_star[i] > 0.5 else -r,
                +r if x_star[j] > 0.5 else -r,
            )
        except Exception:
            star_z = None
    else:
        star_z = star

    # average c, β per epoch
    c_at_x = _interp_c_to_x_axis(result, T)
    h_steps = result.history_steps.astype(float)
    h_b = result.history_beta.astype(float) if result.history_beta.size else None
    if h_b is not None and h_b.size > 0:
        K_x = max(1, int(result.config.get("record_assignment_every", 1)))
        t_x = np.arange(T) * float(K_x)
        beta_at_x = np.interp(t_x, h_steps, h_b)
    else:
        beta_at_x = np.full(T, float(result.config.get("beta_final", 80.0)))

    fig, axes = plt.subplots(
        1, n_panels, figsize=(3.6 * n_panels, 3.7),
        sharex=True, sharey=True, constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]
    extent = [-z_range, z_range, -z_range, z_range]
    for k, ax in enumerate(axes):
        seg = x_ij[edges[k]:edges[k + 1]].reshape(-1, 2)        # (Ts·K, 2)
        Zseg = to_z(seg)
        # clip into plot box for hexbin
        Zseg = Zseg[(np.abs(Zseg[:, 0]) < z_range) & (np.abs(Zseg[:, 1]) < z_range)]
        if len(Zseg) == 0:
            ax.text(0.5, 0.5, "(empty)", transform=ax.transAxes,
                    ha="center", va="center")
        else:
            ax.hexbin(Zseg[:, 0], Zseg[:, 1], gridsize=40, cmap="Reds",
                      mincnt=1, extent=extent)
        # overlay sample chain trajectory (one chain) on top
        chain0_seg = x_ij[edges[k]:edges[k + 1], 0, :]          # (Ts, 2)
        Zchain0 = to_z(chain0_seg)
        ax.plot(Zchain0[:, 0], Zchain0[:, 1], color="white", lw=0.5,
                alpha=0.6)
        if star_z is not None:
            ax.plot(star_z[0], star_z[1], marker="*", ms=18,
                    color=COL_GOLD, mec="k", mew=0.8, zorder=5,
                    label="satisfying corner" if k == n_panels - 1 else None)
        c_avg = c_at_x[edges[k]:edges[k + 1]].mean() if edges[k + 1] > edges[k] else c_at_x[-1]
        b_avg = beta_at_x[edges[k]:edges[k + 1]].mean() if edges[k + 1] > edges[k] else beta_at_x[-1]
        ax.set_title(rf"$c={c_avg:.1f},\ \beta={b_avg:.1f}$")
        ax.set_xlim(-z_range, z_range)
        ax.set_ylim(-z_range, z_range)
        ax.set_xlabel(rf"$z_{{{i+1}}}$")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(rf"$z_{{{j+1}}}$")
    if star_z is not None:
        axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "CNLM-Langevin SDE — chain density across annealing epochs",
        weight="bold", y=1.03,
    )
    if save:
        fig.savefig(save)
    return fig


def plot_convergence_to_corners(
    result: SolveResult,
    instance,
    chain: Optional[int] = None,
    max_n_for_full_enum: int = 12,
    max_corners_in_legend: int = 8,
    save: Optional[Union[str, Path]] = None,
):
    """
    Two-panel "annealed CNLM-Langevin" trajectory figure (paper Fig. k):

      (left)  x_i(t) = σ(z_i(t)) for every variable along the best chain.
      (right) ‖x(t) − x*‖₂ to every Boolean corner x* ∈ {0,1}^n,
              colour-coded SAT vs UNSAT.  c(t) overlaid on the right axis.

    For ``n_vars > max_n_for_full_enum`` (default 12) the right panel falls
    back to the distance to the best assignment found.
    """
    _apply_style()
    if result.history_x is None or result.history_x.size == 0:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.text(0.5, 0.5,
                "no x trajectory recorded\n(set SolverConfig.record_assignment_every > 0)",
                transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        if save:
            fig.savefig(save)
        return fig

    if chain is None:
        chain = int(result.best_chain)
    x_traj = result.history_x[:, chain, :]                    # (T, n)
    T, n = x_traj.shape

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    # ---- LEFT: x_i(t) per variable
    ax_l = axes[0]
    if n <= 8:
        cmap_v = cm.tab10(np.linspace(0, 1, max(n, 2)))
        for k in range(n):
            ax_l.plot(np.arange(T), x_traj[:, k], color=cmap_v[k % 10],
                      lw=1.6, label=rf"$x_{{{k+1}}}(t)$")
        ax_l.legend(loc="lower right", fontsize=8, ncol=min(n, 4))
    else:
        cmap_v = cm.viridis(np.linspace(0.1, 0.9, n))
        for k in range(n):
            ax_l.plot(np.arange(T), x_traj[:, k], color=cmap_v[k],
                      lw=0.8, alpha=0.7)
    ax_l.axhline(0.5, color="gray", lw=0.5, ls="--")
    ax_l.set_xlabel("recorded step")
    ax_l.set_ylabel(r"$x_i = \sigma(z_i)$")
    ax_l.set_title(rf"Annealed CNLM-Langevin on a {n}-var instance",
                   weight="bold")
    ax_l.set_ylim(-0.02, 1.02)

    # ---- RIGHT: distances to corners
    ax_r = axes[1]
    if n <= max_n_for_full_enum:
        from itertools import product as _product
        corners = np.array(list(_product([0, 1], repeat=n)),
                           dtype=np.float64)                  # (2^n, n)

        # SAT mask via vectorised Boolean evaluator
        from .instance import evaluate_clauses_bool_vectorized
        sat_per_corner = np.zeros(len(corners), dtype=bool)
        for k, c_corner in enumerate(corners):
            mask = evaluate_clauses_bool_vectorized(
                instance.L, np.asarray(instance.n_neg),
                c_corner.astype(bool),
            )
            if hasattr(instance, "is_hard"):
                sat_per_corner[k] = bool(mask[instance.is_hard].all())
            else:
                sat_per_corner[k] = bool(mask.all())

        dists = np.linalg.norm(
            x_traj[:, None, :] - corners[None, :, :], axis=2,
        )                                                     # (T, 2^n)

        idx_unsat = np.where(~sat_per_corner)[0]
        idx_sat = np.where(sat_per_corner)[0]
        # UNSAT corners faintly
        for ci in idx_unsat:
            ax_r.plot(np.arange(T), dists[:, ci],
                      color="lightgray", lw=0.7, alpha=0.55)
        # SAT corners coloured
        if len(idx_sat):
            colors = cm.tab10(np.linspace(0, 1, max(len(idx_sat), 2)))
            for k, ci in enumerate(idx_sat):
                bits = "(" + ",".join(str(int(b)) for b in corners[ci]) + ")"
                ax_r.plot(
                    np.arange(T), dists[:, ci],
                    color=colors[k % 10], lw=1.8,
                    label=bits if k < max_corners_in_legend else None,
                )
            ax_r.legend(loc="upper right", fontsize=7,
                        title="SAT corners", ncol=2)
        ax_r.set_title("Convergence trajectory", weight="bold")
    else:
        x_star = result.assignment.astype(float)
        d = np.linalg.norm(x_traj - x_star, axis=1)
        ax_r.plot(np.arange(T), d, color=COL_TRUE, lw=2.0,
                  label="distance to best assignment")
        ax_r.legend(loc="upper right", fontsize=8)
        ax_r.set_title(
            f"Convergence to best corner (n={n} too large to enumerate)",
            weight="bold",
        )
    ax_r.set_xlabel("recorded step")
    ax_r.set_ylabel(r"$\|x(t) - x^*\|_2$")

    # twin axis: c(t)
    c_at_x = _interp_c_to_x_axis(result, T)
    ax_c = ax_r.twinx()
    ax_c.plot(np.arange(T), c_at_x, color="gray", lw=1.0, ls="--",
              alpha=0.85, label=r"$c(t)$")
    ax_c.set_ylabel(r"$c(t)$", color="gray")
    ax_c.tick_params(axis="y", labelcolor="gray")
    ax_c.spines["right"].set_visible(True)
    ax_c.grid(False)

    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


def plot_gradient_snr(snr_data: dict, save: Optional[Union[str, Path]] = None):
    """
    Plot the signal-to-noise ratio of the per-clause confidence-gradient
    ∂F_j/∂c_j over uniform random x, as a function of c.

    Input ``snr_data`` is the dict returned by
    :func:`cnlm_langevin.core.analysis.analyze_gradient_snr`.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    c = np.asarray(snr_data["c_grid"])
    snr = np.asarray(snr_data["snr"])
    n_samples = snr_data.get("n_samples", "?")
    ax.plot(
        c, snr, color=COL_HARD, marker="o", ms=4, lw=1.6,
        label=r"$|\mathbb{E}[\partial_c F_j]|/\sigma$",
    )
    ax.set_xscale("log")
    ax.set_xlabel("confidence $c$")
    ax.set_ylabel("SNR")
    ax.set_title(
        f"Signal-to-noise of confidence gradient "
        f"(over uniform random x, {n_samples} samples per c)",
        weight="bold", fontsize=10,
    )
    ax.legend(loc="best", fontsize=9)
    if save:
        fig.savefig(save)
    return fig


# ============================================================================
def save_all_plots(out_dir: Path, result: SolveResult, instance) -> None:
    """Save every meaningful plot to ``out_dir`` as PDF."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # always save (closes after)
    figs: List = []
    try:
        figs.append(plot_solution_summary(result, instance, save=out_dir/"00_summary.pdf"))
        figs.append(plot_energy_curve(result, save=out_dir/"01_free_energy.pdf"))
        figs.append(plot_clause_satisfaction(result, save=out_dir/"02_n_sat.pdf"))
        figs.append(plot_confidence_evolution(result, save=out_dir/"03_schedule.pdf"))
        figs.append(plot_schedule(result, save=out_dir/"04_geodesic.pdf"))
        figs.append(plot_chain_diversity(result, save=out_dir/"05_chains.pdf"))
        figs.append(plot_score_distribution(result, instance, save=out_dir/"06_clause_distribution.pdf"))
        if result.problem_type == "MaxSAT":
            figs.append(plot_maxsat_breakdown(result, instance, save=out_dir/"07_maxsat.pdf"))
        if result.history_x is not None and result.history_x.size > 0:
            figs.append(plot_assignment_trajectory(result, save=out_dir/"08_x_trajectory.pdf"))
            try:
                figs.append(plot_clause_veritron_heatmap(
                    result, instance, save=out_dir/"09_clause_veritron_heatmap.pdf"))
            except Exception:
                pass
            try:
                figs.append(plot_convergence_to_corners(
                    result, instance, save=out_dir/"10_convergence_to_corners.pdf"))
            except Exception:
                pass
            if instance.n_vars >= 2:
                try:
                    figs.append(plot_sde_trajectory_2d(
                        result, instance, save=out_dir/"11_sde_trajectory_2d.pdf"))
                except Exception:
                    pass
        # static (no result needed) — landscape sweep, only for small instances
        if instance.n_vars <= 64 and instance.n_clauses <= 256:
            try:
                figs.append(plot_loss_landscape_sweep(
                    instance, save=out_dir/"12_loss_landscape_sweep.pdf"))
            except Exception:
                pass
    finally:
        for f in figs:
            try:
                plt.close(f)
            except Exception:
                pass
