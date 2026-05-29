#!/usr/bin/env python3
"""
Benchmark CNLM-Langevin (and its v1.7 variants) against neural and
classical MaxSAT solvers (incl. python-sat's RC2).

Run on a folder of DIMACS .wcnf files (old or MSE-2022 'h' format);
produces per-instance CSV, a per-solver summary JSON, comparison PDF
plots (sat-score, runtime, MaxSAT cost), and the paper-style table.

Usage
-----
    python benchmark_MaxSAT.py  /path/to/wcnf_folder  /path/to/output_dir
        [--timeout 60] [--solvers cnlm_langevin walksat ...]
        [--gms-ckpt PATH] [--sgat-ckpt DIR]
        [--cnlm-tuned-config tuned_config.json]
        [--polish-flips 5000] [--best-of-k 16]
        [--pt-rungs 6]  [--full-registry]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# --- import bootstrap -------------------------------------------------------
# Works whether ``pycnlm`` is pip-installed or this script is run straight
# from a source checkout.  We add the repository root (the directory that
# contains the ``pycnlm/`` package) to sys.path, then import via the full
# package path.  Importing the benchmark package installs the
# ``cnlm_langevin`` compatibility alias used by the driver and adapters.
_PKG_ROOT = Path(__file__).resolve().parents[3]  # .../<repo>/  (contains pycnlm/)
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from pycnlm.core.LangevinCNLM.benchmark.driver import (
    run_benchmark,
    instantiate_adapters,
)



def main():
    p = argparse.ArgumentParser(
        description="MaxSAT solver benchmark (CNLM-Langevin + variants + baselines).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_folder", type=str, help="folder containing .wcnf files")
    p.add_argument("output_folder", type=str,
                   help="where to write CSV / JSON / plots / paper-table")

    p.add_argument("--timeout", type=float, default=60.0,
                   help="per-call timeout in seconds (default 60)")
    p.add_argument("--solvers", nargs="*", default=None,
                   help="whitelist of adapter names; default = DEFAULT_ADAPTERS")
    p.add_argument("--full-registry", action="store_true",
                   help="include adapters that need missing checkpoints")

    g = p.add_argument_group("neural baseline checkpoints (opt-in)")
    g.add_argument("--gms-ckpt", default=None,
                   help="path to GMS pretrained checkpoint")
    g.add_argument("--sgat-ckpt", default=None,
                   help="path to SGAT-MS model directory")
    g.add_argument("--sgat-model-id", default="1",
                   help="SGAT-MS model id (default 1)")
    g.add_argument("--sgat-mode", default="sgat",
                   choices=["sgat", "mixing", "lm"],
                   help="SGAT-MS run mode; mixing/lm don't need a checkpoint")

    g = p.add_argument_group("CNLM-Langevin core hyperparameters")
    g.add_argument("--cnlm-steps", type=int, default=2000)
    g.add_argument("--cnlm-chains", type=int, default=32)
    g.add_argument("--cnlm-hard-scale", type=float, default=1e3,
                   help="multiplier on hard clause weights")

    g = p.add_argument_group("v1.7 enhancement knobs")
    g.add_argument("--polish-flips", type=int, default=5000,
                   help="WalkSAT polish budget for cnlm_ws / cnlm_hyperopt")
    g.add_argument("--best-of-k", type=int, default=16)
    g.add_argument("--best-of-k-sigma", type=float, default=0.20)
    g.add_argument("--pt-rungs", type=int, default=6)
    g.add_argument("--swap-every", type=int, default=100)
    g.add_argument("--cnlm-tuned-config", default=None,
                   help="path to Optuna-tuned config JSON for cnlm_hyperopt")

    args = p.parse_args()

    if args.gms_ckpt:   os.environ["CNLM_GMS_CKPT"] = args.gms_ckpt
    if args.sgat_ckpt:  os.environ["CNLM_SGAT_CKPT"] = args.sgat_ckpt

    cnlm_core = {
        "n_steps": args.cnlm_steps,
        "n_chains": args.cnlm_chains,
        "hard_scale": args.cnlm_hard_scale,
    }
    adapter_kwargs = {
        "cnlm_langevin": dict(cnlm_core),
        "cnlm_ws": dict(
            cnlm_core,
            polish_max_flips=args.polish_flips,
            best_of_k=args.best_of_k,
            best_of_k_sigma=args.best_of_k_sigma,
        ),
        "cnlm_pt": dict(
            cnlm_core,
            polish_max_flips=args.polish_flips,
            best_of_k=args.best_of_k,
            n_rungs=args.pt_rungs,
            swap_every=args.swap_every,
        ),
        "cnlm_hyperopt": dict(
            polish_max_flips=args.polish_flips,
            best_of_k=args.best_of_k,
            tuned_config=args.cnlm_tuned_config,
        ),
        "sgat_ms": {
            "model_id": args.sgat_model_id, "mode": args.sgat_mode,
        },
    }
    adapters = instantiate_adapters(
        problem_type="MaxSAT",
        adapter_names=args.solvers,
        adapter_kwargs=adapter_kwargs,
        full_registry=args.full_registry,
    )

    run_benchmark(
        folder=args.input_folder,
        out_dir=args.output_folder,
        problem_type="MaxSAT",
        adapters=adapters,
        timeout_s=args.timeout,
        verbose=True,
    )


if __name__ == "__main__":
    main()