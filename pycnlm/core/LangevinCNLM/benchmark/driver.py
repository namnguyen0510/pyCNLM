"""
Benchmark driver shared by `benchmark_SAT.py` and `benchmark_MaxSAT.py`.

Discovers DIMACS files in a folder, instantiates the requested adapters,
runs each on each instance with a per-call timeout, writes per-run CSVs,
JSON, and comparison plots.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from cnlm_langevin import (
    parse_dimacs_cnf, parse_dimacs_wcnf,
    SATInstance, MaxSATInstance,
)

from .adapters import get_adapters_for, BaseAdapter, SolveOutcome


def discover_files(folder: Path, problem_type: str) -> List[Path]:
    folder = Path(folder)
    if problem_type == "SAT":
        files = sorted(folder.rglob("*.cnf"))
    else:
        files = sorted(folder.rglob("*.wcnf"))
    return files


def load_instance(path: Path, problem_type: str, root: Path = None):
    # if root is provided, use the path *relative* to root as the instance
    # name — so subfolder structure ("easy/inst_0.cnf") is preserved and
    # downstream grouping (paper_table.py) can detect it.
    if root is not None:
        try:
            rel = path.relative_to(root)
            inst_name = str(rel)
        except ValueError:
            inst_name = path.name
    else:
        inst_name = path.name
    if problem_type == "SAT":
        parsed = parse_dimacs_cnf(path)
        return SATInstance.from_parsed(parsed, name=inst_name)
    parsed = parse_dimacs_wcnf(path)
    return MaxSATInstance.from_parsed(parsed, name=inst_name)


def instantiate_adapters(
    problem_type: str,
    adapter_names: Optional[List[str]] = None,
    adapter_kwargs: Optional[dict] = None,
    full_registry: bool = False,
) -> List[BaseAdapter]:
    """Instantiate all matching adapters (whether or not they're available).

    Parameters
    ----------
    problem_type : "SAT" | "MaxSAT"
    adapter_names : optional whitelist of adapter `.name` strings
    adapter_kwargs : optional {adapter_name: kwargs_dict} per-adapter config
    full_registry : if True, use ALL_ADAPTERS (incl. checkpoint-less neural
        baselines that will appear as empty rows). Default False uses
        DEFAULT_ADAPTERS — the 21 always-runnable solvers.
    """
    adapter_kwargs = adapter_kwargs or {}
    classes = get_adapters_for(
        problem_type,
        names=adapter_names,
        full_registry=full_registry,
    )
    out = []
    for cls in classes:
        kw = adapter_kwargs.get(cls.name, {})
        out.append(cls(**kw))
    return out


def run_benchmark(
    folder: Path,
    out_dir: Path,
    problem_type: str,
    adapters: Optional[List[BaseAdapter]] = None,
    timeout_s: float = 60.0,
    verbose: bool = True,
) -> dict:
    folder = Path(folder)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files(folder, problem_type)
    if not files:
        raise FileNotFoundError(
            f"No {'cnf' if problem_type == 'SAT' else 'wcnf'} files found "
            f"under {folder}"
        )
    if adapters is None:
        adapters = instantiate_adapters(problem_type)

    # availability check up-front
    avail, unavail = [], []
    for a in adapters:
        if a.available():
            avail.append(a)
        else:
            unavail.append(a)

    print(f"\n{'='*72}")
    print(f"Benchmark — {problem_type} — {len(files)} instance(s)")
    print(f"{'='*72}")
    print("Adapters available:")
    for a in avail:
        print(f"  ✓  {a.name:<26s} ({a.kind})")
    if unavail:
        print("Adapters skipped:")
        for a in unavail:
            print(f"  ✗  {a.name:<26s} — {a.unavailable_reason}")
    print()

    rows: List[dict] = []
    n_total = len(files) * max(len(avail), 1)
    counter = 0
    for fpath in files:
        try:
            inst = load_instance(fpath, problem_type, root=folder)
        except Exception as exc:
            if verbose:
                print(f"  ! parse error on {fpath.name}: {exc}")
            continue

        if verbose:
            print(f"\n--- {fpath.name}  (n={inst.n_vars}, m={inst.n_clauses})")

        # also record the unavailable ones once per instance
        for a in unavail:
            counter += 1
            rows.append(SolveOutcome(
                solver=a.name, instance=inst.name,
                problem_type=problem_type, available=False,
                unavailable_reason=a.unavailable_reason,
            ).to_row())

        for a in avail:
            counter += 1
            t0 = time.perf_counter()
            try:
                outcome = a.solve(inst, timeout_s=timeout_s)
            except Exception as exc:
                outcome = SolveOutcome(
                    solver=a.name, instance=inst.name,
                    problem_type=problem_type, available=True,
                    runtime_s=time.perf_counter() - t0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            rows.append(outcome.to_row())
            if verbose:
                if outcome.error:
                    msg = f"ERROR ({outcome.error.splitlines()[0][:60]})"
                elif outcome.timed_out:
                    msg = "TIMEOUT"
                else:
                    msg = (
                        f"sat={outcome.is_SAT}  "
                        f"{outcome.n_satisfied}/{outcome.n_clauses}  "
                        f"score={outcome.sat_score:.4f}  "
                        f"runtime={outcome.runtime_s:.2f}s"
                    )
                    if outcome.cost is not None:
                        msg += f"  cost={outcome.cost:.3f}"
                print(f"   [{counter}/{n_total}]  {a.name:<26s}  {msg}")

    # ----- write per-run CSV
    csv_path = out_dir / "results_per_instance.csv"
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n=> per-instance results -> {csv_path}")

    # ----- aggregate per solver
    summary = aggregate(rows, problem_type)
    summary_path = out_dir / "summary_per_solver.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"=> per-solver summary  -> {summary_path}")

    # ----- plots
    try:
        plot_summary(summary, rows, problem_type, out_dir)
        print(f"=> plots               -> {out_dir}")
    except Exception as exc:
        print(f"  ! plotting failed: {exc}")

    # ----- paper-style results table
    try:
        from .paper_table import write_paper_table
        # detect optional grouping by sub-folder name
        instance_groups = None
        if any(("/" in r["instance"] or "\\" in r["instance"]) for r in rows):
            # rows have path-like names — group by parent dir
            groups: dict = {}
            for r in rows:
                inst = r["instance"]
                parts = inst.replace("\\", "/").split("/")
                key = parts[-2] if len(parts) > 1 else "all"
                groups.setdefault(key, set()).add(inst)
            if len(groups) > 1:
                instance_groups = {k: sorted(v) for k, v in groups.items()}

        # short hyperparameter strings per adapter
        hp_strings = {}
        for a in (adapters or []):
            if a.name == "cnlm_langevin":
                hp_strings[a.name] = (
                    f"{a.config.get('n_steps','?')} steps · "
                    f"{a.config.get('n_chains','?')} chains"
                )
            elif a.name == "walksat":
                hp_strings[a.name] = (
                    f"max-flips {a.config.get('max_flips','?')} · "
                    f"p={a.config.get('p','?')}"
                )
            elif a.name == "random_restart_greedy":
                hp_strings[a.name] = (
                    f"{a.config.get('n_restarts','?')} restarts"
                )
            elif a.name in ("satnet_sdp_numpy", "satnet_official"):
                hp_strings[a.name] = f"k={a.config.get('k', 32)}"

        pdf_path, tex_path = write_paper_table(
            raw_rows=rows, problem_type=problem_type,
            out_dir=out_dir,
            instance_groups=instance_groups,
            hyperparam_strings=hp_strings,
        )
        print(f"=> paper-style table   -> {pdf_path}")
        print(f"                          {tex_path}")
    except Exception as exc:
        import traceback
        print(f"  ! paper-table render failed: {exc}")
        traceback.print_exc()

    print_leaderboard(summary, problem_type)
    return {"rows": rows, "summary": summary, "out_dir": str(out_dir)}


def aggregate(rows, problem_type):
    out = {}
    by_solver = {}
    for r in rows:
        by_solver.setdefault(r["solver"], []).append(r)

    for solver, items in by_solver.items():
        avail = [r for r in items if r["available"]]
        ok = [r for r in avail if not r.get("error") and not r.get("timed_out")]
        scores = [r["sat_score"] for r in ok if r["sat_score"] is not None]
        runtimes = [r["runtime_s"] for r in ok]
        timeouts = sum(1 for r in avail if r.get("timed_out"))
        errors = sum(1 for r in avail if r.get("error"))
        full_solved = sum(1 for r in ok if r.get("is_SAT"))

        d = {
            "n_instances": len(items),
            "available_runs": len(avail),
            "successful_runs": len(ok),
            "errors": errors,
            "timeouts": timeouts,
            "full_solved": full_solved,
            "sat_score_mean": float(np.mean(scores)) if scores else None,
            "sat_score_median": float(np.median(scores)) if scores else None,
            "sat_score_min": float(np.min(scores)) if scores else None,
            "runtime_mean_s": float(np.mean(runtimes)) if runtimes else None,
            "runtime_median_s": float(np.median(runtimes)) if runtimes else None,
        }
        if problem_type == "MaxSAT":
            costs = [r["cost"] for r in ok if r["cost"] is not None]
            hard_sat_rates = [
                (r["n_hard_sat"] or 0) / max(r["n_hard_total"] or 1, 1)
                for r in ok if r["n_hard_total"] is not None and r["n_hard_total"] > 0
            ]
            d["cost_mean"] = float(np.mean(costs)) if costs else None
            d["cost_median"] = float(np.median(costs)) if costs else None
            d["cost_min"] = float(np.min(costs)) if costs else None
            d["hard_sat_rate_mean"] = (
                float(np.mean(hard_sat_rates)) if hard_sat_rates else None
            )
        out[solver] = d
    return out


def print_leaderboard(summary, problem_type):
    print(f"\n{'='*72}")
    print(f"LEADERBOARD — {problem_type}")
    print(f"{'='*72}")
    rows = []
    for solver, d in summary.items():
        if d["successful_runs"] == 0:
            rows.append((solver, None, None, None, d))
            continue
        rows.append((solver, d["sat_score_mean"], d["full_solved"],
                     d["runtime_mean_s"], d))

    # sort: by full_solved desc, then sat_score_mean desc, then runtime asc
    def key(r):
        _, score, full, rt, _ = r
        return (
            -(full or 0),
            -(score or 0.0),
            (rt if rt is not None else 1e9),
        )
    rows.sort(key=key)

    print(f"{'solver':<28s} {'full_SAT':>10s} {'score':>10s} "
          f"{'runtime_s':>12s} {'extras':>16s}")
    print("-" * 78)
    for solver, score, full, rt, d in rows:
        if d["successful_runs"] == 0:
            extra = f"({d['errors']} err, {d['timeouts']} TO)" \
                    if d.get("available_runs") else "(skipped)"
            print(f"{solver:<28s} {'-':>10s} {'-':>10s} {'-':>12s} {extra:>16s}")
            continue
        if problem_type == "MaxSAT":
            cost_mean = d.get("cost_mean")
            extra = f"cost={cost_mean:.3f}" if cost_mean is not None else "-"
        else:
            extra = ""
        print(f"{solver:<28s} {full:>10d} {score:>10.4f} "
              f"{rt:>12.3f} {extra:>16s}")
    print("=" * 72)


def plot_summary(summary, rows, problem_type, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    avail_solvers = [s for s, d in summary.items() if d["successful_runs"] > 0]
    if not avail_solvers:
        return

    # 1) bar chart: mean sat_score per solver
    fig, ax = plt.subplots(figsize=(8, 4))
    means = [summary[s]["sat_score_mean"] or 0.0 for s in avail_solvers]
    fulls = [summary[s]["full_solved"] for s in avail_solvers]
    n_inst = max(1, summary[avail_solvers[0]]["n_instances"])
    full_frac = [f / n_inst for f in fulls]
    x = np.arange(len(avail_solvers))
    ax.bar(x - 0.2, means, width=0.4, color="#2E5EAA", label="mean sat-score")
    ax.bar(x + 0.2, full_frac, width=0.4, color="#1E8449",
           label="fully-solved fraction")
    ax.set_xticks(x); ax.set_xticklabels(avail_solvers, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction")
    ax.set_title(f"{problem_type} benchmark — sat-score and full-solved rate "
                 "across solvers", weight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "01_score_and_solved.pdf")
    plt.close(fig)

    # 2) box plot: runtime distribution per solver
    runs_per_solver = {s: [] for s in avail_solvers}
    for r in rows:
        if r["available"] and not r.get("error") and not r.get("timed_out") \
                and r["solver"] in runs_per_solver:
            runs_per_solver[r["solver"]].append(float(r["runtime_s"]))
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [runs_per_solver[s] for s in avail_solvers]
    bp = ax.boxplot(data, labels=avail_solvers, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#2E5EAA"); patch.set_alpha(0.5)
    ax.set_yscale("log")
    ax.set_ylabel("runtime (s, log)")
    ax.set_title(f"{problem_type} benchmark — per-instance runtime", weight="bold")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "02_runtime_box.pdf")
    plt.close(fig)

    # 3) MaxSAT-only: cost comparison
    if problem_type == "MaxSAT":
        cost_per_solver = {s: [] for s in avail_solvers}
        for r in rows:
            if r["available"] and not r.get("error") and not r.get("timed_out") \
                    and r["solver"] in cost_per_solver \
                    and r.get("cost") is not None:
                cost_per_solver[r["solver"]].append(float(r["cost"]))
        if any(cost_per_solver.values()):
            fig, ax = plt.subplots(figsize=(8, 4))
            data = [cost_per_solver[s] for s in avail_solvers]
            bp = ax.boxplot(data, labels=avail_solvers, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("#C0392B"); patch.set_alpha(0.5)
            ax.set_ylabel("MaxSAT cost  (Σ unsat soft weights)")
            ax.set_title("MaxSAT benchmark — cost (lower is better)",
                         weight="bold")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            fig.tight_layout()
            fig.savefig(out_dir / "03_cost_box.pdf")
            plt.close(fig)
