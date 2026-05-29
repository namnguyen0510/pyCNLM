#!/usr/bin/env python3
"""
Optuna hyperparameter tuner for CNLM-Langevin.
==============================================

Searches over the hyperparameters that the *theory* leaves free —
the discretisation step ``dt``, the regulariser ``λ``, the SDNF margin
``ε``, the schedules ``β_init→β_final`` and ``c_init→c_final``, plus the
slow-SDE rates ``η`` and ``β_c`` — and finds the configuration that
maximises a held-out *weighted sat-score*  while  staying within a
runtime budget.

The tuner does NOT change the SDE itself: every trial's solver is the
exact ``CNLMLangevinSolver`` (optionally with the orchestrator from
``cnlm_langevin.core.enhancements`` for polish / PT / best-of-K).

Drop this file at:    cnlm_langevin/tools/tune_cnlm.py

Run:
    python -m cnlm_langevin.tools.tune_cnlm \\
        --train-folder /path/to/cnf_validation_set \\
        --n-trials    100 \\
        --time-budget 1800 \\
        --out         cnlm_langevin/tools/tuned_config.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np


# ---------- defer optuna import so the module is loadable without it
def _require_optuna():
    try:
        import optuna  # noqa: F401
        return optuna
    except ImportError as exc:
        raise ImportError(
            "optuna is required to run the tuner.  Install with "
            "`pip install optuna`."
        ) from exc


def load_validation_instances(train_folder: Path, problem_type: str = "SAT"):
    """Load every .cnf / .wcnf in `train_folder` (recursively) into instances."""
    from cnlm_langevin import (
        parse_dimacs_cnf, parse_dimacs_wcnf,
        SATInstance, MaxSATInstance,
    )
    train_folder = Path(train_folder)
    if problem_type == "SAT":
        files = sorted(train_folder.rglob("*.cnf"))
        return [SATInstance.from_parsed(parse_dimacs_cnf(f), name=f.name)
                for f in files]
    files = sorted(train_folder.rglob("*.wcnf"))
    return [MaxSATInstance.from_parsed(parse_dimacs_wcnf(f), name=f.name)
            for f in files]


def evaluate_config(
    instances,
    cfg_dict: dict,
    use_enhancements: bool,
    polish_max_flips: int,
    best_of_k: int,
    use_pt: bool,
    timeout_per_inst: float,
) -> dict:
    """
    Run the (possibly enhanced) solver on every instance with the given
    hyperparameters and return aggregate metrics.
    """
    from cnlm_langevin import (
        CNLMLangevinSolver, SolverConfig, MaxSATInstance,
    )
    if use_enhancements:
        from cnlm_langevin.core.enhancements import solve_with_enhancements

    base_cfg = SolverConfig(**cfg_dict)
    sat_scores = []
    runtimes = []
    full_solved = 0
    costs = []
    for inst in instances:
        t0 = time.perf_counter()
        if use_enhancements:
            res = solve_with_enhancements(
                inst, base_cfg,
                polish=(polish_max_flips > 0),
                polish_max_flips=polish_max_flips,
                best_of_k=best_of_k, use_pt=use_pt,
                timeout_s=timeout_per_inst,
            )
        else:
            res = CNLMLangevinSolver(inst, base_cfg).solve()
        runtimes.append(time.perf_counter() - t0)
        sat_scores.append(float(res.sat_score))
        if res.is_SAT:
            full_solved += 1
        if isinstance(inst, MaxSATInstance):
            costs.append(float(res.cost) if res.cost is not None else 0.0)

    return {
        "mean_sat_score": float(np.mean(sat_scores)),
        "min_sat_score": float(np.min(sat_scores)),
        "fraction_full_solved": full_solved / max(len(instances), 1),
        "mean_runtime_s": float(np.mean(runtimes)),
        "mean_cost": float(np.mean(costs)) if costs else None,
    }


def make_objective(instances, args, problem_type: str):
    """Build an Optuna objective closure that scores a hyperparameter trial."""
    optuna = _require_optuna()

    def objective(trial: "optuna.Trial") -> float:
        # ------ search space (faithful to the theory: only "free" knobs)
        cfg_dict = dict(
            n_steps=trial.suggest_int("n_steps", 600, 3000, step=200),
            n_chains=trial.suggest_int("n_chains", 8, 48, step=4),
            dt=trial.suggest_float("dt", 0.01, 0.20, log=True),
            lam=trial.suggest_float("lam", 1e-5, 1e-1, log=True),
            eps=trial.suggest_float("eps", 0.10, 0.90),
            beta_init=trial.suggest_float("beta_init", 0.1, 5.0, log=True),
            beta_final=trial.suggest_float("beta_final", 10.0, 200.0, log=True),
            beta_schedule=trial.suggest_categorical(
                "beta_schedule", ["log", "lin"]),
            c_init=trial.suggest_float("c_init", 0.5, 5.0),
            c_final=trial.suggest_float("c_final", 10.0, 200.0, log=True),
            c_schedule=trial.suggest_categorical(
                "c_schedule", ["lin", "poly"]),
            z_init_scale=trial.suggest_float("z_init_scale", 0.1, 2.0, log=True),
            use_slow_sde=trial.suggest_categorical("use_slow_sde", [True, False]),
            eta=trial.suggest_float("eta", 0.01, 0.5, log=True),
            beta_c=trial.suggest_float("beta_c", 5.0, 100.0, log=True),
            early_stop_when_sat=True,
            seed=0,
            record_assignment_every=0,    # disable for speed during tuning
        )
        # ensure beta_final > beta_init and c_final > c_init
        if cfg_dict["beta_final"] <= cfg_dict["beta_init"]:
            return -10.0
        if cfg_dict["c_final"] <= cfg_dict["c_init"]:
            return -10.0

        metrics = evaluate_config(
            instances, cfg_dict,
            use_enhancements=args.use_enhancements,
            polish_max_flips=args.polish_max_flips,
            best_of_k=args.best_of_k,
            use_pt=args.use_pt,
            timeout_per_inst=args.timeout_per_inst,
        )
        # primary objective: weighted sat-score - λ_rt * runtime
        score = metrics["mean_sat_score"]
        score -= args.runtime_penalty * metrics["mean_runtime_s"]
        # bonus: fraction of fully-solved instances (rewards getting to corner)
        score += 0.30 * metrics["fraction_full_solved"]
        # for MaxSAT: subtract cost (lower is better)
        if metrics.get("mean_cost") is not None:
            score -= 0.01 * metrics["mean_cost"]

        # log to trial user attrs for inspection
        for k, v in metrics.items():
            trial.set_user_attr(k, v)
        return float(score)

    return objective


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-folder", required=True,
                   help="folder of .cnf / .wcnf instances to tune on")
    p.add_argument("--problem", default="SAT", choices=["SAT", "MaxSAT"])
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--time-budget", type=float, default=1800.0,
                   help="overall wall-clock budget in seconds")
    p.add_argument("--timeout-per-inst", type=float, default=20.0)
    p.add_argument("--runtime-penalty", type=float, default=0.05,
                   help="how heavily to penalise runtime in the objective")
    p.add_argument("--use-enhancements", action="store_true",
                   help="evaluate with polish/best-of-K (recommended)")
    p.add_argument("--polish-max-flips", type=int, default=2000)
    p.add_argument("--best-of-k", type=int, default=8)
    p.add_argument("--use-pt", action="store_true")
    p.add_argument("--out", default="tuned_config.json")
    p.add_argument("--storage", default=None,
                   help="optional optuna SQLite URL for resumable studies")
    args = p.parse_args()

    optuna = _require_optuna()
    instances = load_validation_instances(args.train_folder, args.problem)
    if not instances:
        raise FileNotFoundError(f"no {args.problem} instances found in {args.train_folder}")
    print(f"loaded {len(instances)} validation instance(s) from {args.train_folder}")

    sampler = optuna.samplers.TPESampler(seed=0, multivariate=True, group=True)
    study = optuna.create_study(
        direction="maximize", sampler=sampler,
        storage=args.storage,
        study_name=f"cnlm_{args.problem}",
        load_if_exists=True,
    )
    objective = make_objective(instances, args, args.problem)
    t0 = time.perf_counter()
    study.optimize(
        objective, n_trials=args.n_trials,
        timeout=args.time_budget,
        show_progress_bar=True,
    )
    elapsed = time.perf_counter() - t0

    best = study.best_trial
    print("\n" + "=" * 70)
    print(f"best trial #{best.number}  (score={best.value:.4f})")
    print("-" * 70)
    for k, v in best.params.items():
        print(f"  {k:<24s} = {v}")
    print("-" * 70)
    print("metrics:")
    for k, v in best.user_attrs.items():
        print(f"  {k:<24s} = {v}")
    print(f"\nelapsed: {elapsed:.1f}s over {len(study.trials)} trial(s)")

    # save best config
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "problem_type": args.problem,
        "best_params": best.params,
        "best_score": best.value,
        "best_metrics": best.user_attrs,
        "n_trials": len(study.trials),
        "validation_set_size": len(instances),
        "use_enhancements": args.use_enhancements,
        "polish_max_flips": args.polish_max_flips,
        "best_of_k": args.best_of_k,
        "use_pt": args.use_pt,
    }, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()