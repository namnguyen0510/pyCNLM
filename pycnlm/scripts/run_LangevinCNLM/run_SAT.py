#!/usr/bin/env python3
"""
run_SAT.py — drive the CNLM-Langevin (fast-slow) solver across a folder
of DIMACS .cnf files.

v1.7: adds optional enhancements
  --polish        : WalkSAT post-processing
  --best-of-k     : multi-rounding decoding
  --pt            : parallel tempering across rungs
  --tuned-config  : load Optuna-tuned hyperparameters from JSON

Usage
-----
    python run_SAT.py  <input_folder>  <output_folder>  [options]

Examples
--------
    python run_SAT.py  ./benchmarks/uf20  ./out_uf20
    python run_SAT.py  ./cnf_dir  ./out  --workers 8 --steps 2000 --chains 32
    python run_SAT.py  ./cnf_dir  ./out_v17  --polish --best-of-k 16 --pt
    python run_SAT.py  ./cnf_dir  ./out_v17  --tuned-config tools/tuned_config.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent

# --- import bootstrap -------------------------------------------------------
# Works whether ``pycnlm`` is pip-installed or this script is run straight
# from a source checkout.  Adding the repo root makes ``import pycnlm``
# succeed, which in turn installs the ``cnlm_langevin`` compatibility alias.
_PKG_ROOT = Path(__file__).resolve().parents[3]  # .../<repo>/  (contains pycnlm/)
for _p in (str(_PKG_ROOT), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import pycnlm  # noqa: F401  (activates the cnlm_langevin alias)
except ModuleNotFoundError:
    # Legacy flat layout: cnlm_langevin lives next to this script's tree.
    _LCNLM = Path(__file__).resolve().parents[2] / "core" / "LangevinCNLM"
    sys.path.insert(0, str(_LCNLM))

from cnlm_langevin import SolverConfig, solve_folder


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CNLM-Langevin (fast-slow) SAT solver — folder driver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_folder", help="folder containing .cnf files (recursively)")
    p.add_argument("output_folder", help="where to write per-instance results")

    g = p.add_argument_group("parallelism")
    g.add_argument("--workers", type=int, default=0,
                   help="cross-instance worker processes (0=cpu_count)")
    g.add_argument("--chains", type=int, default=16,
                   help="parallel Langevin walkers per instance (vectorised)")

    g = p.add_argument_group("CNLM-Langevin SDE")
    g.add_argument("--steps", type=int, default=1500, help="Euler-Maruyama steps")
    g.add_argument("--dt", type=float, default=0.05, help="time step Δt")
    g.add_argument("--eps", type=float, default=0.5, help="ε in (0,1) controlling SDNF margin")
    g.add_argument("--lam", type=float, default=1e-3, help="λ regulariser (½λ‖z‖²)")

    g = p.add_argument_group("annealing schedules")
    g.add_argument("--beta-init", type=float, default=1.0)
    g.add_argument("--beta-final", type=float, default=80.0)
    g.add_argument("--beta-schedule", choices=["log", "lin", "poly", "const"], default="log")
    g.add_argument("--c-init", type=float, default=1.0)
    g.add_argument("--c-final", type=float, default=60.0)
    g.add_argument("--c-schedule", choices=["log", "lin", "poly", "const"], default="lin")
    g.add_argument("--c-poly-p", type=float, default=1.5)

    g = p.add_argument_group("slow-mode SDE on ρ = log c")
    g.add_argument("--slow-sde", action="store_true",
                   help="enable noisy SDE on confidence ρ (else deterministic c(t))")
    g.add_argument("--eta", type=float, default=0.05)
    g.add_argument("--beta-c", type=float, default=50.0)

    g = p.add_argument_group("v1.7 enhancements")
    g.add_argument("--polish", action="store_true",
                   help="run a WalkSAT polish after the SDE finishes")
    g.add_argument("--polish-flips", type=int, default=5000,
                   help="max flips for the polish stage")
    g.add_argument("--best-of-k", type=int, default=0,
                   help="K independent stochastic roundings of σ(z_T+ξ); 0 disables")
    g.add_argument("--best-of-k-sigma", type=float, default=0.20)
    g.add_argument("--pt", action="store_true",
                   help="use parallel tempering across temperature rungs")
    g.add_argument("--pt-rungs", type=int, default=6)
    g.add_argument("--swap-every", type=int, default=100)
    g.add_argument("--tuned-config", default=None,
                   help="path to Optuna-tuned JSON; overrides individual flags")

    g = p.add_argument_group("misc")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--no-plots", action="store_true", help="skip per-instance PDF plots")
    g.add_argument("--save-trajectory", action="store_true",
                   help="record full σ(z(t)) trajectory (memory-heavy)")
    g.add_argument("--no-restart", action="store_true", help="disable stuck-chain restarts")
    g.add_argument("--no-early-stop", action="store_true",
                   help="run all steps even after a chain finds full SAT")
    g.add_argument("--verbose", action="store_true")
    return p


def args_to_config(args, tuned_overrides: dict = None) -> SolverConfig:
    base = dict(
        n_steps=args.steps, dt=args.dt, n_chains=args.chains, seed=args.seed,
        eps=args.eps, lam=args.lam,
        beta_init=args.beta_init, beta_final=args.beta_final,
        beta_schedule=args.beta_schedule,
        c_init=args.c_init, c_final=args.c_final,
        c_schedule=args.c_schedule, c_poly_p=args.c_poly_p,
        use_slow_sde=args.slow_sde, eta=args.eta, beta_c=args.beta_c,
        restart_on_stuck=not args.no_restart,
        early_stop_when_sat=not args.no_early_stop,
        record_assignment_every=10 if args.save_trajectory else 0,
        verbose=args.verbose,
    )
    if tuned_overrides:
        # only update fields that exist on SolverConfig
        for k, v in tuned_overrides.items():
            if k in SolverConfig.__dataclass_fields__:
                base[k] = v
    return SolverConfig(**base)


def _enhanced_folder_solve(in_dir: Path, out_dir: Path, cfg: SolverConfig,
                           args, tuned_overrides: dict):
    """Replicates the public solve_folder pipeline for SAT but routes each
    instance through cnlm_langevin.core.enhancements.solve_with_enhancements."""
    from cnlm_langevin import parse_dimacs_cnf, SATInstance
    from cnlm_langevin.core.enhancements import solve_with_enhancements

    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.rglob("*.cnf"))
    if not files:
        raise FileNotFoundError(f"no .cnf files under {in_dir}")
    print(f"[v1.7-enhanced] {len(files)} instance(s) to solve")

    summary_rows = []
    for i, fpath in enumerate(files, start=1):
        rel = fpath.relative_to(in_dir)
        print(f"\n[{i}/{len(files)}] {rel}")
        try:
            inst = SATInstance.from_parsed(parse_dimacs_cnf(fpath), name=str(rel))
        except Exception as exc:
            print(f"   parse error: {exc}")
            continue
        t0 = time.perf_counter()
        try:
            res = solve_with_enhancements(
                inst, cfg,
                polish=args.polish or bool(tuned_overrides),
                polish_max_flips=args.polish_flips,
                best_of_k=max(args.best_of_k, 0),
                best_of_k_sigma=args.best_of_k_sigma,
                use_pt=args.pt,
                pt_n_rungs=args.pt_rungs,
                pt_swap_every=args.swap_every,
                timeout_s=3600.0,
            )
        except Exception as exc:
            print(f"   solver error: {exc}")
            continue
        runtime = time.perf_counter() - t0

        # write per-instance dir
        inst_out = out_dir / rel.with_suffix("")
        inst_out.mkdir(parents=True, exist_ok=True)
        summary = {
            "instance": str(rel),
            "n_vars": inst.n_vars, "n_clauses": inst.n_clauses,
            "is_SAT": bool(res.is_SAT),
            "n_satisfied": int(res.n_satisfied),
            "sat_score": float(res.sat_score),
            "runtime_s": float(runtime),
            "best_chain": int(getattr(res, "best_chain", 0) or 0),
            "n_chains": int(getattr(res, "n_chains", cfg.n_chains)),
            "n_steps": int(getattr(res, "n_steps", cfg.n_steps)),
            "config": {
                "polish": args.polish, "polish_flips": args.polish_flips,
                "best_of_k": args.best_of_k,
                "pt": args.pt, "pt_rungs": args.pt_rungs,
                "tuned_config": args.tuned_config,
            },
        }
        (inst_out / "summary.json").write_text(json.dumps(summary, indent=2))

        # DIMACS-style solution line
        sol = "v " + " ".join(
            (str(i + 1) if b else str(-(i + 1)))
            for i, b in enumerate(res.assignment.astype(bool))
        ) + " 0\n"
        (inst_out / "solution.txt").write_text(
            f"s {'SATISFIABLE' if res.is_SAT else 'UNKNOWN'}\n" + sol
        )
        summary_rows.append(summary)
        print(f"   sat={res.is_SAT}  {res.n_satisfied}/{res.n_clauses}  "
              f"runtime={runtime:.2f}s")

    # top-level summary CSV
    if summary_rows:
        import csv
        keys = sorted({k for r in summary_rows for k in r.keys() if k != "config"})
        with (out_dir / "summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in summary_rows:
                w.writerow({k: r.get(k) for k in keys})
        (out_dir / "all_results.json").write_text(json.dumps(summary_rows, indent=2))
        print(f"\n[v1.7-enhanced] wrote {out_dir/'summary.csv'} "
              f"({len(summary_rows)} instances)")


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    tuned_overrides = None
    if args.tuned_config:
        path = Path(args.tuned_config)
        if path.exists():
            data = json.loads(path.read_text())
            tuned_overrides = data.get("best_params", {})
            print(f"[tuned-config] loaded {len(tuned_overrides)} hyperparameters from {path}")
        else:
            print(f"WARNING: --tuned-config {path} does not exist; ignoring")

    cfg = args_to_config(args, tuned_overrides=tuned_overrides)

    in_dir = Path(args.input_folder).resolve()
    out_dir = Path(args.output_folder).resolve()

    enhanced = (args.polish or args.pt or args.best_of_k > 0
                or tuned_overrides is not None)

    print("CNLM-Langevin (fast-slow) — SAT")
    print(f"  input        : {in_dir}")
    print(f"  output       : {out_dir}")
    print(f"  workers      : {args.workers or os.cpu_count()}")
    print(f"  chains       : {cfg.n_chains}  steps: {cfg.n_steps}  dt: {cfg.dt}")
    print(f"  schedules    : β {cfg.beta_schedule} [{cfg.beta_init}→{cfg.beta_final}]   "
          f"c {cfg.c_schedule} [{cfg.c_init}→{cfg.c_final}]")
    print(f"  slow SDE     : {cfg.use_slow_sde}  (η={cfg.eta}, β_c={cfg.beta_c})")
    if enhanced:
        print(f"  v1.7 enhanced: polish={args.polish} (flips={args.polish_flips}), "
              f"best_of_k={args.best_of_k}, pt={args.pt} "
              f"(rungs={args.pt_rungs}, swap_every={args.swap_every}), "
              f"tuned_config={args.tuned_config}")
    print()

    if enhanced:
        _enhanced_folder_solve(in_dir, out_dir, cfg, args, tuned_overrides)
    else:
        solve_folder(
            in_dir, out_dir,
            problem_type="SAT",
            config=cfg,
            n_workers=args.workers,
            save_plots=not args.no_plots,
            save_history_x=args.save_trajectory,
            progress=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())