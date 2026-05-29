"""
cnlm_langevin
=============
A faithful implementation of the CNLM-Langevin (fast-slow) method
for SAT and MaxSAT solving, as described in the CNLM paper.

The method discretises the coupled Itô SDE

    dz_t   = -∇_z F~_λ(z_t; c_t) dt + sqrt(2/β) dW^z_t       (fast)
    dρ_t   = -η ∇_ρ L(z_t; e^{ρ_t}) dt + sqrt(2η/β_c) dW^ρ_t  (slow)

with c_t = e^{ρ_t} ≥ 0 and the lifted free energy

    F~_λ(z; c) = -Σ_j w_j ln(1 + exp(c_j s~_j(z))) + (λ/2)||z||²

over a continuous embedding x = σ(z).
"""
from .core.parser import (
    DimacsParseError,
    parse_dimacs_cnf,
    parse_dimacs_wcnf,
    parse_dimacs_auto,
)
from .core.instance import (
    SATInstance,
    MaxSATInstance,
    build_literal_matrix,
)
from .core.dynamics import (
    CNLMLangevinSolver,
    SolveResult,
    SolverConfig,
)
from .core.solver import (
    solve_sat_file,
    solve_maxsat_file,
    solve_folder,
)
from .core.viz import (
    plot_assignment_trajectory,
    plot_energy_curve,
    plot_clause_satisfaction,
    plot_confidence_evolution,
    plot_schedule,
    plot_chain_diversity,
    plot_solution_summary,
    plot_score_distribution,
    plot_maxsat_breakdown,
    save_all_plots,
    plot_clause_veritron_heatmap,
    plot_loss_landscape_sweep,
    plot_sde_trajectory_2d,
    plot_convergence_to_corners,
    plot_gradient_snr,
)
from .core.analysis import (
    analyze_gradient_snr,
)

__version__ = "1.1.0"
__all__ = [
    "DimacsParseError",
    "parse_dimacs_cnf",
    "parse_dimacs_wcnf",
    "parse_dimacs_auto",
    "SATInstance",
    "MaxSATInstance",
    "build_literal_matrix",
    "CNLMLangevinSolver",
    "SolveResult",
    "SolverConfig",
    "solve_sat_file",
    "solve_maxsat_file",
    "solve_folder",
    "plot_assignment_trajectory",
    "plot_energy_curve",
    "plot_clause_satisfaction",
    "plot_confidence_evolution",
    "plot_schedule",
    "plot_chain_diversity",
    "plot_solution_summary",
    "plot_score_distribution",
    "plot_maxsat_breakdown",
    "save_all_plots",
    "plot_clause_veritron_heatmap",
    "plot_loss_landscape_sweep",
    "plot_sde_trajectory_2d",
    "plot_convergence_to_corners",
    "plot_gradient_snr",
    "analyze_gradient_snr",
]
