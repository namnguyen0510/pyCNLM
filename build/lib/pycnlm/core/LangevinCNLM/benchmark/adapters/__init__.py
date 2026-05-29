"""
Adapter registry.

Two registry views are exported:

* ``ALL_ADAPTERS``        — every adapter class (including legacy / unavailable
                            neural baselines that need their own checkpoints)
* ``DEFAULT_ADAPTERS``    — what the benchmark scripts run by default; this
                            *excludes* the neural rows that have no shipped
                            checkpoints.  Users can opt them back in via
                            ``--solvers <name1> <name2> …`` or ``--full-registry``.
"""
from .base import BaseAdapter, SolveOutcome
from .adapter_cnlm import CNLMAdapter
from .adapter_cnlm_variants import (         # ← v1.7 variants
    CNLMPolishedAdapter, CNLMPTAdapter, CNLMTunedAdapter,
)
from .adapter_walksat import WalkSATAdapter
from .adapter_random_restart import RandomRestartGreedyAdapter
from .adapter_classical_extra import (
    SurveyPropagationAdapter, SimulatedAnnealingAdapter,
)
from .adapter_satnet import SATNetSDPAdapter, SATNetOfficialAdapter
from .adapter_neurosat_mini import NeuroSATMiniAdapter
from .adapter_neurosat import NeuroSATAdapter
from .adapter_neural_others import (
    PDPAdapter,
    NSNetAdapter,
    QuerySATAdapter,
    GMSAdapter,
    SGATMSAdapter,
    G4SATBenchAdapter,
)
from .adapter_pysat import (
    ALL_PYSAT_SAT_ADAPTERS,
    PySATRC2Adapter,
)


# ----- the full list (includes adapters that need missing checkpoints)
ALL_ADAPTERS = [
    # ours: baseline + 3 v1.7 variants
    CNLMAdapter,
    CNLMPolishedAdapter,         # ← polish + best-of-K
    CNLMPTAdapter,               # ← parallel tempering + polish + best-of-K
    CNLMTunedAdapter,            # ← Optuna-tuned hyperparameters + all enhancements
    # classical SLS / message-passing
    WalkSATAdapter,
    RandomRestartGreedyAdapter,
    SurveyPropagationAdapter,
    SimulatedAnnealingAdapter,
    # neural-flavoured (always-runnable)
    SATNetSDPAdapter,
    NeuroSATMiniAdapter,
    # PySAT CDCL backends
    #*ALL_PYSAT_SAT_ADAPTERS,
    PySATRC2Adapter,
    # ↓ no shipped checkpoints / not installable in vanilla setups; opt-in only
    SATNetOfficialAdapter,
    NeuroSATAdapter,
    PDPAdapter,
    NSNetAdapter,
    QuerySATAdapter,
    GMSAdapter,
    SGATMSAdapter,
    G4SATBenchAdapter,
]

# ----- the default set used by benchmark_SAT.py / benchmark_MaxSAT.py
# (everything that can actually run out of the box)
DEFAULT_ADAPTERS = [
    CNLMAdapter,
    CNLMPolishedAdapter,         # ← v1.7
    CNLMPTAdapter,               # ← v1.7
    CNLMTunedAdapter,            # ← v1.7
    WalkSATAdapter,
    RandomRestartGreedyAdapter,
    SurveyPropagationAdapter,
    SimulatedAnnealingAdapter,
    SATNetSDPAdapter,
    NeuroSATMiniAdapter,
    #*ALL_PYSAT_SAT_ADAPTERS,
    PySATRC2Adapter,
]


def get_adapters_for(problem_type: str, names=None,
                     full_registry: bool = False):
    """Return adapter classes that support `problem_type`.

    Parameters
    ----------
    problem_type : "SAT" | "MaxSAT"
    names : optional whitelist of adapter `.name` strings.
    full_registry : if True use ALL_ADAPTERS (incl. always-unavailable rows);
        otherwise (default) use DEFAULT_ADAPTERS.
    """
    pool = ALL_ADAPTERS if (full_registry or names is not None) else DEFAULT_ADAPTERS
    out = []
    for cls in pool:
        if problem_type not in cls.supports:
            continue
        if names is not None and cls.name not in names:
            continue
        out.append(cls)
    return out


__all__ = [
    "BaseAdapter", "SolveOutcome",
    "ALL_ADAPTERS", "DEFAULT_ADAPTERS", "get_adapters_for",
    "CNLMAdapter",
    "CNLMPolishedAdapter", "CNLMPTAdapter", "CNLMTunedAdapter",
    "WalkSATAdapter",
    "RandomRestartGreedyAdapter",
    "SurveyPropagationAdapter", "SimulatedAnnealingAdapter",
    "SATNetSDPAdapter", "SATNetOfficialAdapter",
    "NeuroSATMiniAdapter",
    "NeuroSATAdapter",
    "PDPAdapter", "NSNetAdapter", "QuerySATAdapter",
    "GMSAdapter", "SGATMSAdapter", "G4SATBenchAdapter",
    "ALL_PYSAT_SAT_ADAPTERS", "PySATRC2Adapter",
]