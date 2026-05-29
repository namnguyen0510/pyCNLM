#!/usr/bin/env python3
"""
Benchmark CNLM-Langevin (and its v1.7 variants) against neural and
classical SAT solvers.

Run on a folder of DIMACS .cnf files; produces per-instance CSV, a
per-solver summary JSON, comparison PDF plots, and a results table (paper_table.pdf + paper_table.tex).

Usage
-----
    python benchmark_SAT.py  small-sat-problems-test  result_sat
        [--timeout 60] [--solvers cnlm_langevin walksat ...]
        [--neurosat-ckpt PATH] [--nsnet-ckpt PATH] [--g4satbench-ckpt PATH]
        [--querysat-ckpt PATH]
        [--cnlm-tuned-config tuned_config.json]
        [--polish-flips 5000] [--best-of-k 16]
        [--pt-rungs 6]  [--full-registry]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# allow running from the repo root
# --- import bootstrap -------------------------------------------------------
# Works whether ``pycnlm`` is pip-installed or this script is run straight
# from a source checkout.  We add the repository root (the directory that
# contains the ``pycnlm/`` package) to sys.path, then import via the full
# package path.  Importing the benchmark package installs the
# ``cnlm_langevin`` compatibility alias used by the driver and adapters.
_PKG_ROOT = Path(__file__).resolve().parents[3]  # .../<repo>/  (contains pycnlm/)
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

try:
    from pycnlm.core.LangevinCNLM.benchmark.driver import (
        run_benchmark,
        instantiate_adapters,
    )
except ModuleNotFoundError:
    # Legacy flat layout: add the LangevinCNLM dir (parent of both
    # ``benchmark/`` and ``cnlm_langevin/``) and import by short name.
    _LCNLM = Path(__file__).resolve().parents[2] / "core" / "LangevinCNLM"
    sys.path.insert(0, str(_LCNLM))
    from pycnlm.core.LangevinCNLM.benchmark.driver import run_benchmark, instantiate_adapters


def main():
    p = argparse.ArgumentParser(
        description="SAT solver benchmark (CNLM-Langevin + variants + baselines).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_folder", type=str, help="folder containing .cnf files")
    p.add_argument("output_folder", type=str,
                   help="where to write CSV / JSON / plots / paper-table")

    p.add_argument("--timeout", type=float, default=60.0,
                   help="per-call timeout in seconds (default 60)")
    p.add_argument("--solvers", nargs="*", default=None,
                   help="whitelist of adapter names; default = DEFAULT_ADAPTERS")
    p.add_argument("--full-registry", action="store_true",
                   help="include adapters that need missing checkpoints "
                        "(NeuroSAT, NSNet, ...) — they will appear as empty rows")

    g = p.add_argument_group("neural baseline checkpoints (opt-in)")
    g.add_argument("--neurosat-ckpt", default=None,
                   help="path to NeuroSAT TF1 checkpoint")
    g.add_argument("--nsnet-ckpt", default=None,
                   help="path to NSNet PyTorch checkpoint")
    g.add_argument("--querysat-ckpt", default=None,
                   help="path to QuerySAT model directory")
    g.add_argument("--g4satbench-ckpt", default=None,
                   help="path to G4SATBench model_best.pt")
    g.add_argument("--g4satbench-model", default="neurosat",
                   choices=["neurosat", "ggnn", "gcn", "gat", "gin"])
    g.add_argument("--g4satbench-graph", default="lcg",
                   choices=["lcg", "vcg"])

    g = p.add_argument_group("CNLM-Langevin core hyperparameters")
    g.add_argument("--cnlm-steps", type=int, default=1500)
    g.add_argument("--cnlm-chains", type=int, default=24)

    g = p.add_argument_group("v1.7 enhancement knobs")
    g.add_argument("--polish-flips", type=int, default=5000,
                   help="WalkSAT polish budget for cnlm_ws / cnlm_hyperopt")
    g.add_argument("--best-of-k", type=int, default=16,
                   help="best-of-K rounding count")
    g.add_argument("--best-of-k-sigma", type=float, default=0.20,
                   help="z-noise std-dev used in best-of-K rounding")
    g.add_argument("--pt-rungs", type=int, default=6,
                   help="number of temperature rungs for cnlm_pt")
    g.add_argument("--swap-every", type=int, default=100,
                   help="parallel-tempering swap interval (steps)")
    g.add_argument("--cnlm-tuned-config", default=None,
                   help="path to Optuna-tuned config JSON for cnlm_hyperopt "
                        "(default: cnlm_langevin/tools/tuned_config.json)")

    args = p.parse_args()

    # propagate ckpt paths to env vars
    if args.neurosat_ckpt:    os.environ["CNLM_NEUROSAT_CKPT"] = args.neurosat_ckpt
    if args.nsnet_ckpt:       os.environ["CNLM_NSNET_CKPT"] = args.nsnet_ckpt
    if args.querysat_ckpt:    os.environ["CNLM_QUERYSAT_CKPT"] = args.querysat_ckpt
    if args.g4satbench_ckpt:  os.environ["CNLM_G4SATBENCH_CKPT"] = args.g4satbench_ckpt

    cnlm_core = {"n_steps": args.cnlm_steps, "n_chains": args.cnlm_chains}
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
        "g4satbench": {
            "model": args.g4satbench_model, "graph": args.g4satbench_graph,
        },
    }

    adapters = instantiate_adapters(
        problem_type="SAT",
        adapter_names=args.solvers,
        adapter_kwargs=adapter_kwargs,
        full_registry=args.full_registry,
    )

    run_benchmark(
        folder=args.input_folder,
        out_dir=args.output_folder,
        problem_type="SAT",
        adapters=adapters,
        timeout_s=args.timeout,
        verbose=True,
    )


if __name__ == "__main__":
    main()