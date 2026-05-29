"""
High-level entry points for solving SAT and MaxSAT files / folders.

Two layers of parallelism are exposed:

  * within-instance:  ``SolverConfig.n_chains`` parallel Langevin walkers
                      vectorised in NumPy (single process, BLAS-parallel).
  * across-instance:  multiple worker *processes* via ProcessPoolExecutor
                      (`n_workers` argument in `solve_folder`).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from .dynamics import CNLMLangevinSolver, SolverConfig, SolveResult
from .instance import SATInstance, MaxSATInstance
from .parser import parse_dimacs_cnf, parse_dimacs_wcnf, parse_dimacs_auto


# ============================================================================
# single-instance entry points
# ============================================================================
def solve_sat_file(
    path: Union[str, Path],
    config: Optional[SolverConfig] = None,
    name: Optional[str] = None,
) -> SolveResult:
    """Parse a CNF file and run the CNLM-Langevin solver on it."""
    p = Path(path)
    parsed = parse_dimacs_cnf(p)
    inst = SATInstance.from_parsed(parsed, name=name or p.name)
    solver = CNLMLangevinSolver(inst, config or SolverConfig())
    return solver.solve()


def solve_maxsat_file(
    path: Union[str, Path],
    config: Optional[SolverConfig] = None,
    name: Optional[str] = None,
) -> SolveResult:
    """Parse a WCNF file and run the CNLM-Langevin solver on it."""
    p = Path(path)
    parsed = parse_dimacs_wcnf(p)
    inst = MaxSATInstance.from_parsed(parsed, name=name or p.name)
    solver = CNLMLangevinSolver(inst, config or SolverConfig())
    return solver.solve()


def solve_file_auto(
    path: Union[str, Path],
    config: Optional[SolverConfig] = None,
    name: Optional[str] = None,
) -> SolveResult:
    """Dispatch on file extension."""
    p = Path(path)
    if p.suffix.lower() in (".wcnf",):
        return solve_maxsat_file(p, config, name)
    return solve_sat_file(p, config, name)


# ============================================================================
# serialization helpers
# ============================================================================
def _json_safe(obj):
    """Convert numpy scalars/arrays/dataclass leftovers to JSON-friendly types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.size > 50_000:
            return {"_truncated": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if not np.isfinite(v):
            return None
        return v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    return str(obj)


def _result_summary(result: SolveResult) -> Dict:
    """Concise human-readable summary of a SolveResult."""
    summary = {
        "instance_name": result.instance_name,
        "problem_type": result.problem_type,
        "n_vars": int(result.n_vars),
        "n_clauses": int(result.n_clauses),
        "is_SAT": bool(result.is_SAT),
        "n_satisfied": int(result.n_satisfied),
        "sat_score": float(result.sat_score),
        "n_chains": int(result.n_chains),
        "n_steps": int(result.n_steps),
        "runtime_s": float(result.runtime_s),
        "converged_step": int(result.converged_step) if result.converged_step is not None else None,
        "best_chain": int(result.best_chain),
    }
    if result.problem_type == "MaxSAT":
        summary.update({
            "cost": float(result.cost) if result.cost is not None else None,
            "soft_weight_satisfied": float(result.soft_weight_satisfied) if result.soft_weight_satisfied is not None else None,
            "n_hard_sat": int(result.n_hard_sat) if result.n_hard_sat is not None else None,
            "n_hard_total": int(result.n_hard_total) if result.n_hard_total is not None else None,
            "n_soft_sat": int(result.n_soft_sat) if result.n_soft_sat is not None else None,
            "n_soft_total": int(result.n_soft_total) if result.n_soft_total is not None else None,
            "is_HARD_SAT": bool(result.is_SAT),
            "soft_satisfaction_rate": (
                float(result.n_soft_sat) / max(1, int(result.n_soft_total))
                if (result.n_soft_total is not None and result.n_soft_total > 0) else None
            ),
        })
    return summary


def write_solution_dimacs(path: Path, result: SolveResult) -> None:
    """
    Write a DIMACS-style solution line:
        s SATISFIABLE  / s UNSATISFIABLE / s UNKNOWN
        v <signed literals> 0
    For MaxSAT we additionally emit "o <cost>".
    """
    lines = [f"c CNLM-Langevin (fast-slow) solver  v1.0"]
    lines.append(f"c instance: {result.instance_name}")
    lines.append(f"c problem : {result.problem_type}")
    lines.append(f"c n_vars  : {result.n_vars}")
    lines.append(f"c n_claus : {result.n_clauses}")
    lines.append(f"c n_sat   : {result.n_satisfied}")
    lines.append(f"c runtime : {result.runtime_s:.3f}s")

    if result.problem_type == "MaxSAT":
        if result.cost is not None:
            lines.append(f"o {int(result.cost) if float(result.cost).is_integer() else result.cost}")
        if result.is_SAT:
            lines.append("s OPTIMUM FOUND" if result.n_soft_sat == result.n_soft_total else "s SATISFIABLE")
        else:
            lines.append("s UNKNOWN")
    else:
        lines.append("s SATISFIABLE" if result.is_SAT else "s UNKNOWN")

    lits = []
    for i, xi in enumerate(result.assignment, start=1):
        lits.append(str(i if int(xi) == 1 else -i))
    # split into multiple v-lines for readability
    chunk = 20
    for k in range(0, len(lits), chunk):
        lines.append("v " + " ".join(lits[k:k+chunk]) + (" 0" if k + chunk >= len(lits) else ""))

    path.write_text("\n".join(lines) + "\n")


def save_result_full(out_dir: Path, result: SolveResult, instance) -> Path:
    """
    Save a complete record of one solve to ``out_dir``:
        - summary.json
        - result_full.npz   (all histories and arrays)
        - solution.txt      (DIMACS solution line)
        - instance_meta.json
    Returns the directory path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _result_summary(result)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # full numerical record (compressed npz)
    npz_payload = {
        "assignment": np.asarray(result.assignment, dtype=np.int8),
        "sat_mask": np.asarray(result.sat_mask, dtype=bool) if result.sat_mask is not None else np.array([]),
        "history_steps": result.history_steps,
        "history_beta": result.history_beta,
        "history_c_mean": result.history_c_mean,
        "history_c_min": result.history_c_min,
        "history_c_max": result.history_c_max,
        "history_free_energy": result.history_free_energy,
        "history_n_sat": result.history_n_sat,
        "history_best_n_sat": result.history_best_n_sat,
        "final_x_all": result.final_x_all,
        "final_n_sat_all": result.final_n_sat_all,
    }
    if result.history_x is not None:
        npz_payload["history_x"] = result.history_x

    np.savez_compressed(out_dir / "result_full.npz", **npz_payload)

    # DIMACS-style solution
    write_solution_dimacs(out_dir / "solution.txt", result)

    # instance metadata
    inst_meta: Dict = {
        "name": getattr(instance, "name", ""),
        "n_vars": int(instance.n_vars),
        "n_clauses": int(instance.n_clauses),
        "clause_widths": {
            "min": int(instance.width.min()) if instance.n_clauses else 0,
            "max": int(instance.width.max()) if instance.n_clauses else 0,
            "mean": float(instance.width.mean()) if instance.n_clauses else 0.0,
            "median": float(np.median(instance.width)) if instance.n_clauses else 0.0,
        },
    }
    if isinstance(instance, MaxSATInstance):
        inst_meta.update({
            "is_maxsat": True,
            "n_hard": int(instance.n_hard),
            "n_soft": int(instance.n_soft),
            "total_soft_weight": float(instance.total_soft_weight),
            "top": float(instance.top) if np.isfinite(instance.top) else "inf",
            "new_format": bool(instance.new_format),
        })
    else:
        inst_meta["is_maxsat"] = False

    (out_dir / "instance_meta.json").write_text(json.dumps(_json_safe(inst_meta), indent=2))
    return out_dir


# ============================================================================
# folder-level driver with cross-instance multiprocessing
# ============================================================================
# A worker function must be top-level for pickling.
def _worker_solve(
    file_path: str,
    problem_type: str,
    config_dict: Dict,
    out_dir: str,
    save_plots: bool,
    save_history_x: bool,
) -> Dict:
    """
    Worker entry: parse one .cnf/.wcnf file, solve, save outputs.

    Imports are local so that worker processes start cleanly under
    multiprocessing's spawn method on macOS / Windows.
    """
    import json as _json
    from pathlib import Path as _Path
    import traceback as _tb

    try:
        from .parser import parse_dimacs_cnf as _pcnf, parse_dimacs_wcnf as _pwcnf
        from .instance import SATInstance as _SAT, MaxSATInstance as _MS
        from .dynamics import CNLMLangevinSolver as _Solver, SolverConfig as _Cfg

        cfg = _Cfg(**config_dict)
        # honour save_history_x
        if not save_history_x:
            cfg.record_assignment_every = 0

        fp = _Path(file_path)
        if problem_type == "SAT":
            parsed = _pcnf(fp)
            inst = _SAT.from_parsed(parsed, name=fp.name)
        else:
            parsed = _pwcnf(fp)
            inst = _MS.from_parsed(parsed, name=fp.name)

        result = _Solver(inst, cfg).solve()

        # output directory: out_dir / <instance_stem>
        inst_out = _Path(out_dir) / fp.stem
        save_result_full(inst_out, result, inst)

        if save_plots:
            try:
                from .viz import save_all_plots
                save_all_plots(inst_out, result, inst)
            except Exception as e:  # plots must never break the run
                _tb_str = _tb.format_exc()
                (inst_out / "plot_error.log").write_text(f"{e}\n\n{_tb_str}")

        return {
            "ok": True,
            "file": str(fp),
            "out_dir": str(inst_out),
            "summary": _result_summary(result),
        }
    except Exception as e:  # capture any error for the report
        return {
            "ok": False,
            "file": str(file_path),
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def _discover_files(folder: Path, problem_type: str) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"folder not found: {folder}")
    if problem_type == "SAT":
        exts = (".cnf", ".dimacs", ".sat")
    else:
        exts = (".wcnf",)
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    return files


def solve_folder(
    folder: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    problem_type: str = "SAT",
    config: Optional[SolverConfig] = None,
    n_workers: int = 0,
    save_plots: bool = True,
    save_history_x: bool = False,
    progress: bool = True,
) -> Dict:
    """
    Solve every .cnf (or .wcnf) under ``folder`` in parallel and write
    per-instance outputs under ``out_dir``.

    Parameters
    ----------
    n_workers : int
        Number of worker *processes*.  0 → use ``os.cpu_count()``.
        1 → run sequentially in the calling process (handy for debugging).
    save_history_x : bool
        Record the full sigmoid trajectory for every instance.  Memory-
        hungry; off by default.
    """
    folder = Path(folder)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if problem_type not in ("SAT", "MaxSAT"):
        raise ValueError("problem_type must be 'SAT' or 'MaxSAT'")

    files = _discover_files(folder, problem_type)
    if not files:
        raise FileNotFoundError(f"no {'.cnf' if problem_type=='SAT' else '.wcnf'} files under {folder}")

    cfg = config or SolverConfig()
    cfg_dict = asdict(cfg)
    if not save_history_x:
        cfg_dict["record_assignment_every"] = 0

    if n_workers == 0:
        n_workers = max(1, os.cpu_count() or 1)
    n_workers = min(n_workers, len(files))

    t0 = time.perf_counter()
    rows: List[Dict] = []

    if n_workers == 1:
        # sequential for debug / very small batches
        for i, fp in enumerate(files):
            if progress:
                print(f"[{i+1}/{len(files)}] {fp.name}", flush=True)
            row = _worker_solve(
                str(fp), problem_type, cfg_dict, str(out_dir),
                save_plots, save_history_x,
            )
            rows.append(row)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {
                ex.submit(
                    _worker_solve,
                    str(fp), problem_type, cfg_dict, str(out_dir),
                    save_plots, save_history_x,
                ): fp
                for fp in files
            }
            done = 0
            for fut in as_completed(futures):
                fp = futures[fut]
                try:
                    row = fut.result()
                except Exception as e:
                    row = {
                        "ok": False, "file": str(fp), "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                rows.append(row)
                done += 1
                if progress:
                    summary = row.get("summary", {})
                    if row.get("ok"):
                        n_sat = summary.get("n_satisfied", "?")
                        nc = summary.get("n_clauses", "?")
                        rt = summary.get("runtime_s", 0.0)
                        is_sat = "SAT" if summary.get("is_SAT") else "UNK"
                        score = summary.get("sat_score", 0.0)
                        msg = f"[{done}/{len(files)}] {fp.name}: {is_sat} {n_sat}/{nc} ({score:.3f}) in {rt:.2f}s"
                    else:
                        msg = f"[{done}/{len(files)}] {fp.name}: ERROR {row.get('error','')[:80]}"
                    print(msg, flush=True)

    runtime = time.perf_counter() - t0

    # aggregate
    ok_rows = [r for r in rows if r.get("ok")]
    err_rows = [r for r in rows if not r.get("ok")]
    summaries = [r["summary"] for r in ok_rows]
    aggregate = _aggregate_summaries(summaries, problem_type)

    aggregate["n_files_total"] = len(files)
    aggregate["n_files_ok"] = len(ok_rows)
    aggregate["n_files_failed"] = len(err_rows)
    aggregate["total_runtime_s"] = runtime
    aggregate["n_workers"] = n_workers
    aggregate["problem_type"] = problem_type
    aggregate["folder_in"] = str(folder)
    aggregate["folder_out"] = str(out_dir)

    # write top-level reports
    (out_dir / "all_results.json").write_text(
        json.dumps(_json_safe({"aggregate": aggregate, "rows": rows}), indent=2)
    )
    _write_csv(out_dir / "summary.csv", summaries, problem_type)
    if err_rows:
        (out_dir / "errors.json").write_text(json.dumps(_json_safe(err_rows), indent=2))

    if progress:
        _print_aggregate(aggregate)

    return {"aggregate": aggregate, "rows": rows}


# ============================================================================
# aggregation
# ============================================================================
def _aggregate_summaries(summaries: List[Dict], problem_type: str) -> Dict:
    if not summaries:
        return {"empty": True}

    sat_scores = [s["sat_score"] for s in summaries]
    runtimes = [s["runtime_s"] for s in summaries]
    n_sat_solved = sum(1 for s in summaries if s.get("is_SAT"))
    agg = {
        "n_solved_completely": int(n_sat_solved),
        "fraction_solved": n_sat_solved / max(1, len(summaries)),
        "sat_score_mean": float(np.mean(sat_scores)),
        "sat_score_median": float(np.median(sat_scores)),
        "sat_score_min": float(np.min(sat_scores)),
        "runtime_mean_s": float(np.mean(runtimes)),
        "runtime_median_s": float(np.median(runtimes)),
        "runtime_total_s": float(np.sum(runtimes)),
    }

    if problem_type == "MaxSAT":
        costs = [s["cost"] for s in summaries if s.get("cost") is not None]
        n_hard_ok = sum(1 for s in summaries if s.get("is_HARD_SAT"))
        soft_rates = [s["soft_satisfaction_rate"] for s in summaries
                      if s.get("soft_satisfaction_rate") is not None]
        agg["maxsat"] = {
            "n_hard_satisfied": int(n_hard_ok),
            "fraction_hard_satisfied": n_hard_ok / max(1, len(summaries)),
            "cost_mean": float(np.mean(costs)) if costs else None,
            "cost_median": float(np.median(costs)) if costs else None,
            "cost_min": float(np.min(costs)) if costs else None,
            "cost_max": float(np.max(costs)) if costs else None,
            "soft_sat_rate_mean": float(np.mean(soft_rates)) if soft_rates else None,
        }

    return agg


def _write_csv(path: Path, summaries: List[Dict], problem_type: str) -> None:
    import csv
    if not summaries:
        path.write_text("")
        return
    keys = list(summaries[0].keys())
    # ensure consistent ordering: preferred columns first
    preferred = ["instance_name", "problem_type", "n_vars", "n_clauses",
                 "is_SAT", "n_satisfied", "sat_score", "runtime_s",
                 "n_steps", "n_chains", "best_chain", "converged_step"]
    if problem_type == "MaxSAT":
        preferred += ["cost", "soft_weight_satisfied", "n_hard_sat",
                      "n_hard_total", "n_soft_sat", "n_soft_total",
                      "soft_satisfaction_rate", "is_HARD_SAT"]
    final_keys = [k for k in preferred if k in keys] + [k for k in keys if k not in preferred]

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=final_keys)
        w.writeheader()
        for s in summaries:
            row = {k: s.get(k, "") for k in final_keys}
            for k, v in row.items():
                if isinstance(v, (np.floating, np.integer)):
                    row[k] = v.item()
            w.writerow(row)


def _print_aggregate(agg: Dict) -> None:
    print()
    print("=" * 70)
    print("CNLM-Langevin (fast-slow) — folder summary")
    print("=" * 70)
    print(f"  files in       : {agg.get('folder_in')}")
    print(f"  files out      : {agg.get('folder_out')}")
    print(f"  problem type   : {agg.get('problem_type')}")
    print(f"  files solved   : {agg.get('n_files_ok')}/{agg.get('n_files_total')}")
    print(f"  fully SAT/HARD : {agg.get('n_solved_completely')} "
          f"({agg.get('fraction_solved', 0):.1%})")
    print(f"  sat score mean : {agg.get('sat_score_mean', 0):.4f}  "
          f"(median {agg.get('sat_score_median', 0):.4f}, min {agg.get('sat_score_min', 0):.4f})")
    print(f"  runtime totals : sum={agg.get('runtime_total_s', 0):.2f}s  "
          f"mean={agg.get('runtime_mean_s', 0):.2f}s  workers={agg.get('n_workers')}")
    if "maxsat" in agg and agg["maxsat"]:
        ms = agg["maxsat"]
        print("  --- MaxSAT")
        print(f"  hard satisfied : {ms.get('n_hard_satisfied')} "
              f"({ms.get('fraction_hard_satisfied', 0):.1%})")
        if ms.get("cost_mean") is not None:
            print(f"  cost           : mean={ms.get('cost_mean'):.2f}  "
                  f"median={ms.get('cost_median'):.2f}  "
                  f"min={ms.get('cost_min'):.2f}")
        if ms.get("soft_sat_rate_mean") is not None:
            print(f"  soft sat rate  : {ms.get('soft_sat_rate_mean'):.4f}")
    print("=" * 70)
