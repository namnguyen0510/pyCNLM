"""
Command-line interface for ``pycnlm``.

Run ``pycnlm --help`` for the available sub-commands::

    pycnlm version           Show installed version and exit.
    pycnlm solve-sat   PATH  Solve a CNF instance / folder with CNLM-Langevin.
    pycnlm solve-maxsat PATH Solve a WCNF instance / folder with CNLM-Langevin.
    pycnlm info              Print package introspection (modules, optional deps).

The CLI is a thin shim over the public Python API: every command can be
replicated programmatically with one or two lines of code.  See the
docstrings of :mod:`pycnlm.core.LangevinCNLM.cnlm_langevin` for details.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

from pycnlm._version import __version__

# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s :: %(message)s"
_LOG_DATEFMT = "%H:%M:%S"
log = logging.getLogger("pycnlm.cli")


def _configure_logging(verbosity: int) -> None:
    """Map -v / -vv / -q to a stdlib logging level."""
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt=_LOG_DATEFMT)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _dump_json(obj: Any, path: Path) -> None:
    """Serialise the SolveResult dataclass (or any nested object) to JSON."""
    def _convert(o: Any) -> Any:
        if is_dataclass(o) and not isinstance(o, type):
            return _convert(asdict(o))
        if isinstance(o, dict):
            return {k: _convert(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_convert(v) for v in o]
        # NumPy scalars / arrays (avoid hard numpy import at module top).
        np = sys.modules.get("numpy")
        if np is not None:
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.generic):
                return o.item()
        if isinstance(o, Path):
            return str(o)
        return o

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_convert(obj), f, indent=2, default=str)


def _resolve_paths(target: str, suffixes: tuple[str, ...]) -> list[Path]:
    """Expand a file-or-folder argument into a list of instance paths."""
    p = Path(target)
    if not p.exists():
        raise SystemExit(f"error: path does not exist: {p}")
    if p.is_file():
        return [p]
    matches: list[Path] = []
    for suf in suffixes:
        matches.extend(sorted(p.rglob(f"*{suf}")))
    if not matches:
        raise SystemExit(
            f"error: no files matching {suffixes} found under {p}"
        )
    return matches


def _build_solver_config(args: argparse.Namespace):
    """Translate CLI args into a SolverConfig dataclass instance.

    Done lazily so ``pycnlm --help`` doesn't drag in numpy.
    """
    from pycnlm.core.LangevinCNLM.cnlm_langevin import SolverConfig

    overrides: dict[str, Any] = {}
    if args.chains is not None:
        overrides["n_chains"] = args.chains
    if args.steps is not None:
        overrides["n_steps"] = args.steps
    if args.seed is not None:
        overrides["seed"] = args.seed
    if getattr(args, "slow_sde", False):
        overrides["use_slow_sde"] = True

    # SolverConfig is a dataclass; only pass fields it actually has.
    valid = {f for f in SolverConfig.__dataclass_fields__}
    overrides = {k: v for k, v in overrides.items() if k in valid}
    return SolverConfig(**overrides)


# ---------------------------------------------------------------------------
#  Sub-commands
# ---------------------------------------------------------------------------
def _cmd_version(_args: argparse.Namespace) -> int:
    print(f"pycnlm {__version__}")
    return 0


def _cmd_info(_args: argparse.Namespace) -> int:
    """Diagnostic dump — which optional deps are installed, where the package
    lives, Python interpreter, etc."""
    import platform

    import pycnlm

    print(f"pycnlm           {pycnlm.__version__}")
    print(f"package path     {Path(pycnlm.__file__).parent}")
    print(f"python           {platform.python_version()} ({sys.executable})")
    print(f"platform         {platform.platform()}")
    print()
    print("Optional dependency groups:")
    groups = {
        "dwave":     ["dwave_networkx", "minorminer", "dimod", "neal"],
        "neural":    ["torch"],
        "benchmark": ["pysat", "sklearn", "pandas", "cnfgen", "tqdm"],
        "docs":      ["mkdocs", "mkdocs_material", "mkdocstrings"],
    }
    for name, mods in groups.items():
        installed = []
        for m in mods:
            try:
                importlib.import_module(m)
                installed.append(m)
            except ImportError:
                pass
        status = (
            "OK"   if len(installed) == len(mods)
            else "partial" if installed
            else "missing"
        )
        print(f"  [{name:<9}] {status:<7}  ({len(installed)}/{len(mods)} importable)")
    return 0


def _cmd_solve_sat(args: argparse.Namespace) -> int:
    """Solve one or more CNF files."""
    from pycnlm.core.LangevinCNLM.cnlm_langevin import solve_folder, solve_sat_file

    cfg = _build_solver_config(args)
    paths = _resolve_paths(args.path, suffixes=(".cnf",))
    out_dir = Path(args.out) if args.out else None

    if len(paths) == 1:
        log.info("Solving %s", paths[0])
        t0 = time.perf_counter()
        result = solve_sat_file(paths[0], config=cfg)
        dt = time.perf_counter() - t0
        log.info("Done in %.2fs — SAT=%s, best_energy=%.4g", dt,
                 getattr(result, "is_sat", "?"),
                 getattr(result, "best_energy", float("nan")))
        if out_dir is not None:
            _dump_json(result, out_dir / f"{paths[0].stem}.json")
            log.info("Wrote %s", out_dir / f"{paths[0].stem}.json")
        return 0

    # Folder mode — delegate to the parallel driver.
    if out_dir is None:
        raise SystemExit("error: --out is required when solving a folder")
    log.info("Solving %d files with %d workers", len(paths), args.workers)
    solve_folder(
        Path(args.path),
        out_dir,
        problem_type="SAT",
        config=cfg,
        n_workers=args.workers,
        save_plots=args.save_plots,
        progress=not args.quiet,
    )
    return 0


def _cmd_solve_maxsat(args: argparse.Namespace) -> int:
    """Solve one or more WCNF files."""
    from pycnlm.core.LangevinCNLM.cnlm_langevin import (
        solve_folder,
        solve_maxsat_file,
    )

    cfg = _build_solver_config(args)
    paths = _resolve_paths(args.path, suffixes=(".wcnf",))
    out_dir = Path(args.out) if args.out else None

    if len(paths) == 1:
        log.info("Solving %s", paths[0])
        t0 = time.perf_counter()
        result = solve_maxsat_file(paths[0], config=cfg)
        dt = time.perf_counter() - t0
        log.info("Done in %.2fs", dt)
        if out_dir is not None:
            _dump_json(result, out_dir / f"{paths[0].stem}.json")
            log.info("Wrote %s", out_dir / f"{paths[0].stem}.json")
        return 0

    if out_dir is None:
        raise SystemExit("error: --out is required when solving a folder")
    log.info("Solving %d files with %d workers", len(paths), args.workers)
    solve_folder(
        Path(args.path),
        out_dir,
        problem_type="MaxSAT",
        config=cfg,
        n_workers=args.workers,
        save_plots=args.save_plots,
        progress=not args.quiet,
    )
    return 0


# ---------------------------------------------------------------------------
#  Argument parser
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pycnlm",
        description="Continuous Non-Linear Manifold solver toolkit.",
    )
    p.add_argument(
        "-V", "--version", action="version",
        version=f"pycnlm {__version__}",
    )
    p.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="-v for INFO, -vv for DEBUG (default: WARNING).",
    )

    sub = p.add_subparsers(dest="command", metavar="<command>", required=True)

    # version
    sp = sub.add_parser("version", help="Show installed version and exit.")
    sp.set_defaults(func=_cmd_version)

    # info
    sp = sub.add_parser("info", help="Print package & optional-dep diagnostics.")
    sp.set_defaults(func=_cmd_info)

    # ---- solver options shared between solve-sat and solve-maxsat ---------
    def _add_solver_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "path",
            help="CNF/WCNF file or directory tree containing them.",
        )
        sp.add_argument(
            "--out", "-o", default=None,
            help="Output directory for JSON results. Required for folder mode.",
        )
        sp.add_argument(
            "--chains", type=int, default=None,
            help="Number of parallel Langevin chains per instance.",
        )
        sp.add_argument(
            "--steps", type=int, default=None,
            help="Number of integration steps.",
        )
        sp.add_argument(
            "--workers", "-j", type=int, default=max(1, (os.cpu_count() or 2) - 1),
            help="Number of worker processes when solving a folder.",
        )
        sp.add_argument(
            "--seed", type=int, default=None,
            help="Master RNG seed.",
        )
        sp.add_argument(
            "--slow-sde", action="store_true",
            help="Enable the slow ρ-process (coupled fast–slow SDE).",
        )
        sp.add_argument(
            "--save-plots", action="store_true",
            help="(Folder mode) Write per-instance diagnostic plots.",
        )
        sp.add_argument(
            "--quiet", "-q", action="store_true",
            help="(Folder mode) Suppress the per-file progress bar.",
        )

    # solve-sat
    sp = sub.add_parser(
        "solve-sat",
        help="Solve a CNF instance (or folder) with CNLM-Langevin.",
        description=(
            "Solve a CNF instance (or folder) with the CNLM-Langevin solver."
        ),
    )
    _add_solver_args(sp)
    sp.set_defaults(func=_cmd_solve_sat)

    # solve-maxsat
    sp = sub.add_parser(
        "solve-maxsat",
        help="Solve a WCNF instance (or folder) with CNLM-Langevin.",
        description=(
            "Solve a WCNF instance (or folder) with the CNLM-Langevin solver."
        ),
    )
    _add_solver_args(sp)
    sp.set_defaults(func=_cmd_solve_maxsat)

    return p


# ---------------------------------------------------------------------------
#  Entry-point
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        return 130
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        log.error("Unhandled error: %s", exc, exc_info=args.verbose >= 2)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
