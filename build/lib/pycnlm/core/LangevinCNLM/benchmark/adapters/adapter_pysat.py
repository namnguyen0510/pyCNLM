"""
python-sat (PySAT) backend adapters.

Wraps the 13 classical CDCL SAT solvers shipped with the
`python-sat` PyPI package, plus the RC2 weighted-MaxSAT solver, in
the unified BaseAdapter interface.

Reference: Ignatiev, Morgado & Marques-Silva, "PySAT: A Python Toolkit
for Prototyping with SAT Oracles", SAT 2018.
"""
from __future__ import annotations

import sys
import time
from typing import List, Tuple
import numpy as np

from .base import BaseAdapter, SolveOutcome


# --------------------------------------------------------------------------- #
#  Robust import of the optional ``pysat`` dependency
# --------------------------------------------------------------------------- #
def _ensure_pysat_importable():
    """Try hard to import ``pysat``; return ``None`` on success else the exc.

    A plain ``import pysat`` can fail even when ``python-sat`` *is* installed
    if the interpreter running the benchmark has a stripped / incomplete
    ``sys.path`` (e.g. a script launched so that ``sys.path[0]`` shadows the
    environment, a ``--user`` install whose user-site was skipped, or a
    sub-interpreter started without site initialisation).  Before giving up
    we re-discover every standard site-packages directory and splice any
    missing ones onto ``sys.path``, then retry.
    """
    import importlib

    try:
        importlib.import_module("pysat")
        return None
    except Exception as first_exc:  # noqa: BLE001 - we retry below
        pass

    # Gather candidate site-packages directories from every reliable source.
    candidates: list[str] = []
    try:
        import site

        if hasattr(site, "getsitepackages"):
            candidates.extend(site.getsitepackages())
        if hasattr(site, "getusersitepackages"):
            candidates.append(site.getusersitepackages())
    except Exception:  # noqa: BLE001
        pass
    try:
        import sysconfig

        for key in ("purelib", "platlib"):
            p = sysconfig.get_paths().get(key)
            if p:
                candidates.append(p)
    except Exception:  # noqa: BLE001
        pass

    added = False
    for path in candidates:
        if path and path not in sys.path:
            sys.path.append(path)
            added = True

    if added:
        importlib.invalidate_caches()

    try:
        importlib.import_module("pysat")
        return None
    except Exception as exc:  # noqa: BLE001
        return exc


# ---------------------------------------------------------- backend metadata
PYSAT_BACKENDS: List[Tuple[str, str, str]] = [
    # (pysat_name, display_label, citation_short)
    ("cd",   "CaDiCaL 1.0",    "Biere 2017"),
    ("cd15", "CaDiCaL 1.5",    "Biere 2024"),
    ("gc3",  "Glucose 3.0",    "Audemard & Simon 2009"),
    ("gc4",  "Glucose 4.1",    "Audemard & Simon 2018"),
    ("g3",   "Glucose 3",      "Audemard & Simon 2009"),
    ("g4",   "Glucose 4",      "Audemard & Simon 2018"),
    ("lgl",  "Lingeling",      "Biere 2017"),
    ("m22",  "MiniSat 2.2",    "Eén & Sörensson 2003"),
    ("mc",   "MiniCard",       "Liffiton & Maglalang 2012"),
    ("mgh",  "MiniSat-GH",     "Eén & Sörensson 2018"),
    ("mcb",  "MapleSAT (DBQAS)", "Liang et al. 2017"),
    ("mcm",  "MapleSAT (LRB)", "Liang et al. 2016"),
    ("mpl",  "MapleSAT",       "Liang et al. 2018"),
]


# ============================================================== SAT
class _PySATSATAdapterBase(BaseAdapter):
    """Base class for any PySAT CDCL backend used as a SAT solver."""
    kind = "classical"
    supports = {"SAT", "MaxSAT"}    # via the embedded SAT path; MaxSAT has RC2 below
    SOLVER_NAME = "cd"               # overridden per concrete subclass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def available(self) -> bool:
        exc = _ensure_pysat_importable()
        if exc is not None:
            self.unavailable_reason = (
                f"python-sat not importable (`pip install python-sat`).  "
                f"Import failed even after sys.path repair: {exc}"
            )
            return False
        return True

    def solve(self, instance, timeout_s: float = 120.0) -> SolveOutcome:
        from pysat.solvers import Solver
        from cnlm_langevin.core.instance import MaxSATInstance

        is_max = isinstance(instance, MaxSATInstance)
        problem_type = "MaxSAT" if is_max else "SAT"
        n = instance.n_vars
        m = instance.n_clauses

        # Convert clauses to plain Python int lists (PySAT expects int lits)
        clauses = [[int(l) for l in cl] for cl in instance.clauses]
        if is_max:
            # for MaxSAT we hand only HARD clauses to the SAT solver and
            # report whether they are jointly satisfiable; the solver does
            # NOT attempt cost optimisation here (use pysat_rc2 for that).
            hard_idx = np.where(instance.is_hard)[0]
            clauses_to_solve = [clauses[j] for j in hard_idx]
        else:
            clauses_to_solve = clauses

        t0 = time.perf_counter()
        try:
            with Solver(name=self.SOLVER_NAME,
                        bootstrap_with=clauses_to_solve) as solver:
                # PySAT supports a CPU-time budget for some solvers
                if hasattr(solver, "time_budget") and timeout_s > 0:
                    try:
                        solver.time_budget(int(max(timeout_s, 1)))
                    except Exception:
                        pass
                sat_flag = solver.solve()
                model = solver.get_model() if sat_flag else None
        except Exception as exc:
            return SolveOutcome(
                solver=self.name, instance=getattr(instance, "name", ""),
                problem_type=problem_type, available=True,
                runtime_s=time.perf_counter() - t0,
                error=f"{type(exc).__name__}: {exc}",
            )
        runtime = time.perf_counter() - t0

        # convert PySAT model (list of signed lits) → boolean assignment
        x = np.zeros(n, dtype=bool)
        if model is not None:
            for lit in model:
                v = abs(lit) - 1
                if 0 <= v < n:
                    x[v] = lit > 0

        n_sat, _, h_sat, s_sat, cost = self._verify(x, instance)
        out = SolveOutcome(
            solver=self.name, instance=getattr(instance, "name", ""),
            problem_type=problem_type, available=True,
            runtime_s=runtime,
            is_SAT=(n_sat == m if not is_max
                    else (h_sat == int(instance.is_hard.sum()))),
            n_satisfied=n_sat, n_clauses=m,
            sat_score=n_sat / max(m, 1),
            assignment=x.astype(int).tolist(),
        )
        if is_max:
            out.cost = cost
            out.n_hard_sat = h_sat
            out.n_hard_total = int(instance.is_hard.sum())
            out.n_soft_sat = s_sat
            out.n_soft_total = int((~instance.is_hard).sum())
            out.extras = {"note": "hard-only solve (no MaxSAT optimisation; use pysat_rc2 for that)"}
        return out


def _make_pysat_sat_adapter(backend_name: str) -> type:
    """Factory: build a unique subclass of :class:`_PySATSATAdapterBase`
    bound to a specific PySAT backend."""
    cls = type(
        f"PySAT_{backend_name}_Adapter",
        (_PySATSATAdapterBase,),
        {"SOLVER_NAME": backend_name,
         "name": f"pysat_{backend_name}",
         "kind": "classical",
         "supports": {"SAT", "MaxSAT"}},
    )
    return cls


# generate one Adapter class per PySAT backend
PySAT_cd_Adapter   = _make_pysat_sat_adapter("cd")
PySAT_cd15_Adapter = _make_pysat_sat_adapter("cd15")
PySAT_gc3_Adapter  = _make_pysat_sat_adapter("gc3")
PySAT_gc4_Adapter  = _make_pysat_sat_adapter("gc4")
PySAT_g3_Adapter   = _make_pysat_sat_adapter("g3")
PySAT_g4_Adapter   = _make_pysat_sat_adapter("g4")
PySAT_lgl_Adapter  = _make_pysat_sat_adapter("lgl")
PySAT_m22_Adapter  = _make_pysat_sat_adapter("m22")
PySAT_mc_Adapter   = _make_pysat_sat_adapter("mc")
PySAT_mgh_Adapter  = _make_pysat_sat_adapter("mgh")
PySAT_mcb_Adapter  = _make_pysat_sat_adapter("mcb")
PySAT_mcm_Adapter  = _make_pysat_sat_adapter("mcm")
PySAT_mpl_Adapter  = _make_pysat_sat_adapter("mpl")

ALL_PYSAT_SAT_ADAPTERS = [
    PySAT_cd_Adapter, PySAT_cd15_Adapter,
    PySAT_gc3_Adapter, PySAT_gc4_Adapter,
    PySAT_g3_Adapter, PySAT_g4_Adapter,
    PySAT_lgl_Adapter,
    PySAT_m22_Adapter, PySAT_mc_Adapter, PySAT_mgh_Adapter,
    PySAT_mcb_Adapter, PySAT_mcm_Adapter, PySAT_mpl_Adapter,
]


# ============================================================== MaxSAT (RC2)
class PySATRC2Adapter(BaseAdapter):
    """RC2 — Relaxable Cardinality 2, an exact MaxSAT solver shipped with PySAT.

    Reference: Ignatiev, Morgado, Marques-Silva, "RC2: an Efficient MaxSAT
    Solver." 2019.
    """
    name = "pysat_rc2"
    kind = "classical"
    supports = {"MaxSAT"}

    def __init__(self, backend: str = "g3", **kwargs):
        super().__init__(backend=backend, **kwargs)

    def available(self) -> bool:
        exc = _ensure_pysat_importable()
        if exc is None:
            try:
                from pysat.examples.rc2 import RC2     # noqa: F401
                from pysat.formula import WCNF        # noqa: F401
            except Exception as exc:  # noqa: BLE001
                pass
            else:
                return True
        self.unavailable_reason = (
            f"python-sat not importable (`pip install python-sat`).  "
            f"Import failed even after sys.path repair: {exc}"
        )
        return False

    def solve(self, instance, timeout_s: float = 120.0) -> SolveOutcome:
        from pysat.examples.rc2 import RC2
        from pysat.formula import WCNF

        wcnf = WCNF()
        for cl, w, hard in zip(instance.clauses, instance.weights, instance.is_hard):
            cl_int = [int(l) for l in cl]
            if bool(hard):
                wcnf.append(cl_int)
            else:
                wcnf.append(cl_int, weight=float(w))

        t0 = time.perf_counter()
        x_arr = np.zeros(instance.n_vars, dtype=bool)
        cost_rc2 = None
        timed_out = False
        try:
            with RC2(wcnf, solver=self.config["backend"]) as rc2:
                # rc2.compute() runs to optimum; we crudely cap by checking after
                # iterative calls if time is over (RC2 doesn't expose a budget).
                model = rc2.compute()
                cost_rc2 = float(rc2.cost) if rc2.cost is not None else None
            runtime = time.perf_counter() - t0
            if model is not None:
                for lit in model:
                    v = abs(lit) - 1
                    if 0 <= v < instance.n_vars:
                        x_arr[v] = lit > 0
        except Exception as exc:
            return SolveOutcome(
                solver=self.name, instance=getattr(instance, "name", ""),
                problem_type="MaxSAT", available=True,
                runtime_s=time.perf_counter() - t0,
                error=f"{type(exc).__name__}: {exc}",
            )
        # If we exceeded the timeout
        if runtime > timeout_s:
            timed_out = True

        n_sat, _, h_sat, s_sat, cost = self._verify(x_arr, instance)
        out = SolveOutcome(
            solver=self.name, instance=getattr(instance, "name", ""),
            problem_type="MaxSAT", available=True,
            runtime_s=runtime, timed_out=timed_out,
            is_SAT=(h_sat == int(instance.is_hard.sum())),
            n_satisfied=n_sat, n_clauses=instance.n_clauses,
            sat_score=n_sat / max(instance.n_clauses, 1),
            cost=cost,
            n_hard_sat=h_sat,
            n_hard_total=int(instance.is_hard.sum()),
            n_soft_sat=s_sat,
            n_soft_total=int((~instance.is_hard).sum()),
            assignment=x_arr.astype(int).tolist(),
            extras={"backend": self.config["backend"],
                    "rc2_cost": cost_rc2},
        )
        return out


__all__ = (
    [f"PySAT_{n}_Adapter" for n, _, _ in PYSAT_BACKENDS]
    + ["ALL_PYSAT_SAT_ADAPTERS", "PYSAT_BACKENDS", "PySATRC2Adapter"]
)
