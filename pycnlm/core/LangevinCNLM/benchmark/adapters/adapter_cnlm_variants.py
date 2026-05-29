"""
Three CNLM-Langevin variant adapters.

  * ``cnlm_ws``  — base SDE + best-of-K rounding + WalkSAT polish
  * ``cnlm_pt``        — base SDE under parallel tempering + best-of-K
  * ``cnlm_hyperopt``     — Optuna-tuned hyperparameters (loaded from JSON)
                         + all enhancements

All three subclass ``CNLMAdapter`` and only override ``solve``, so they
inherit availability checks and SolveOutcome plumbing.

Drop this file at:    benchmark/adapters/adapter_cnlm_variants.py

Then add the three classes to ``ALL_ADAPTERS`` and (where you want them
to run by default) ``DEFAULT_ADAPTERS`` in ``benchmark/adapters/__init__.py``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .adapter_cnlm import CNLMAdapter
from .base import SolveOutcome


def _outcome_from_result(adapter, instance, res, runtime: float) -> SolveOutcome:
    """Build a SolveOutcome from a SolveResult-compatible object."""
    from cnlm_langevin import MaxSATInstance
    is_max = isinstance(instance, MaxSATInstance)
    out = SolveOutcome(
        solver=adapter.name,
        instance=getattr(instance, "name", ""),
        problem_type=("MaxSAT" if is_max else "SAT"),
        available=True, runtime_s=runtime,
        is_SAT=bool(res.is_SAT),
        n_satisfied=int(res.n_satisfied),
        n_clauses=int(res.n_clauses),
        sat_score=float(res.sat_score),
        cost=(float(res.cost) if res.cost is not None else None),
        n_hard_sat=res.n_hard_sat, n_hard_total=res.n_hard_total,
        n_soft_sat=res.n_soft_sat, n_soft_total=res.n_soft_total,
        assignment=res.assignment.astype(int).tolist(),
        extras={"best_chain": getattr(res, "best_chain", None),
                "n_chains": getattr(res, "n_chains", None),
                "n_steps":  getattr(res, "n_steps", None)},
    )
    return out


# ============================================================ cnlm_ws
class CNLMPolishedAdapter(CNLMAdapter):
    """Base CNLM-Langevin SDE + best-of-K rounding + WalkSAT polish."""
    name = "cnlm_ws"
    kind = "ours"

    def __init__(self, polish_max_flips: int = 5000, best_of_k: int = 16,
                 best_of_k_sigma: float = 0.20, **kwargs):
        super().__init__(**kwargs)
        self.config["polish_max_flips"] = polish_max_flips
        self.config["best_of_k"] = best_of_k
        self.config["best_of_k_sigma"] = best_of_k_sigma

    def solve(self, instance, timeout_s: float = 120.0) -> SolveOutcome:
        from cnlm_langevin import SolverConfig, MaxSATInstance
        from cnlm_langevin.core.enhancements import solve_with_enhancements
        cfg = SolverConfig(**{
            k: v for k, v in self.config.items()
            if k in SolverConfig.__dataclass_fields__
        })
        t0 = time.perf_counter()
        try:
            res = solve_with_enhancements(
                instance, cfg,
                polish=True,
                polish_max_flips=int(self.config["polish_max_flips"]),
                best_of_k=int(self.config["best_of_k"]),
                best_of_k_sigma=float(self.config["best_of_k_sigma"]),
                use_pt=False,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            return SolveOutcome(
                solver=self.name, instance=getattr(instance, "name", ""),
                problem_type=("MaxSAT" if isinstance(instance, MaxSATInstance) else "SAT"),
                available=True, runtime_s=time.perf_counter() - t0,
                error=f"{type(exc).__name__}: {exc}",
            )
        return _outcome_from_result(self, instance, res, time.perf_counter() - t0)


# ================================================================== cnlm_pt
class CNLMPTAdapter(CNLMAdapter):
    """CNLM-Langevin under parallel tempering across rungs (theory-preserving)."""
    name = "cnlm_pt"
    kind = "ours"

    def __init__(self, n_rungs: int = 6, swap_every: int = 100,
                 polish_max_flips: int = 2000, best_of_k: int = 16,
                 **kwargs):
        super().__init__(**kwargs)
        self.config["n_rungs"] = n_rungs
        self.config["swap_every"] = swap_every
        self.config["polish_max_flips"] = polish_max_flips
        self.config["best_of_k"] = best_of_k

    def solve(self, instance, timeout_s: float = 120.0) -> SolveOutcome:
        from cnlm_langevin import SolverConfig, MaxSATInstance
        from cnlm_langevin.core.enhancements import solve_with_enhancements
        cfg = SolverConfig(**{
            k: v for k, v in self.config.items()
            if k in SolverConfig.__dataclass_fields__
        })
        t0 = time.perf_counter()
        try:
            res = solve_with_enhancements(
                instance, cfg,
                polish=True,
                polish_max_flips=int(self.config["polish_max_flips"]),
                best_of_k=int(self.config["best_of_k"]),
                use_pt=True,
                pt_n_rungs=int(self.config["n_rungs"]),
                pt_swap_every=int(self.config["swap_every"]),
                timeout_s=timeout_s,
            )
        except Exception as exc:
            return SolveOutcome(
                solver=self.name, instance=getattr(instance, "name", ""),
                problem_type=("MaxSAT" if isinstance(instance, MaxSATInstance) else "SAT"),
                available=True, runtime_s=time.perf_counter() - t0,
                error=f"{type(exc).__name__}: {exc}",
            )
        return _outcome_from_result(self, instance, res, time.perf_counter() - t0)


# =============================================================== cnlm_hyperopt
# Default tuned hyperparameters — these are sensible numbers that you
# should overwrite by running the Optuna tuner on YOUR validation set:
#       python -m cnlm_langevin.tools.tune_cnlm --train-folder ... --use-enhancements
# Then point the adapter at the resulting JSON via
#       --tuned-config /path/to/tuned_config.json
_DEFAULT_TUNED_PARAMS = {
    "n_steps":            2200,
    "n_chains":           36,
    "dt":                 0.05,
    "lam":                3e-4,
    "eps":                0.45,
    "beta_init":          0.6,
    "beta_final":         120.0,
    "beta_schedule":      "log",
    "c_init":             1.5,
    "c_final":            80.0,
    "c_schedule":         "poly",
    "z_init_scale":       0.6,
    "use_slow_sde":       True,
    "eta":                0.05,
    "beta_c":             40.0,
    "early_stop_when_sat": True,
    "seed":               0,
    "record_assignment_every": 0,
}

DEFAULT_TUNED_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "cnlm_langevin" / "tools" / "tuned_config.json"
)


class CNLMTunedAdapter(CNLMAdapter):
    """CNLM-Langevin with Optuna-tuned hyperparameters + enhancements."""
    name = "cnlm_hyperopt"
    kind = "ours"

    def __init__(self, tuned_config: Optional[str] = None,
                 polish_max_flips: int = 5000, best_of_k: int = 16,
                 use_pt: bool = False, **kwargs):
        super().__init__(**kwargs)
        # load tuned hyperparameters if available
        params = dict(_DEFAULT_TUNED_PARAMS)
        cfg_path = Path(tuned_config) if tuned_config else DEFAULT_TUNED_CONFIG_PATH
        if cfg_path.exists():
            try:
                data = json.loads(cfg_path.read_text())
                params.update(data.get("best_params", {}))
                self._loaded_from = str(cfg_path)
            except Exception:
                self._loaded_from = "(failed to load — using defaults)"
        else:
            self._loaded_from = "(no tuned_config.json — using defaults)"
        self.config.update(params)
        self.config["polish_max_flips"] = polish_max_flips
        self.config["best_of_k"] = best_of_k
        self.config["use_pt"] = use_pt

    def solve(self, instance, timeout_s: float = 120.0) -> SolveOutcome:
        from cnlm_langevin import SolverConfig, MaxSATInstance
        from cnlm_langevin.core.enhancements import solve_with_enhancements
        cfg = SolverConfig(**{
            k: v for k, v in self.config.items()
            if k in SolverConfig.__dataclass_fields__
        })
        t0 = time.perf_counter()
        try:
            res = solve_with_enhancements(
                instance, cfg,
                polish=True,
                polish_max_flips=int(self.config["polish_max_flips"]),
                best_of_k=int(self.config["best_of_k"]),
                use_pt=bool(self.config["use_pt"]),
                timeout_s=timeout_s,
            )
        except Exception as exc:
            return SolveOutcome(
                solver=self.name, instance=getattr(instance, "name", ""),
                problem_type=("MaxSAT" if isinstance(instance, MaxSATInstance) else "SAT"),
                available=True, runtime_s=time.perf_counter() - t0,
                error=f"{type(exc).__name__}: {exc}",
            )
        out = _outcome_from_result(self, instance, res, time.perf_counter() - t0)
        out.extras["tuned_from"] = self._loaded_from
        return out


__all__ = ["CNLMPolishedAdapter", "CNLMPTAdapter", "CNLMTunedAdapter"]