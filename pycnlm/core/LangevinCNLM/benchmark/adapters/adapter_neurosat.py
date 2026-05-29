"""
NeuroSAT adapter (wraps third_party/neurosat — Selsam et al., ICLR 2019).

NeuroSAT is a message-passing GNN that classifies SAT/UNSAT and recovers
the satisfying assignment by clustering literal embeddings.  The official
repo (https://github.com/dselsam/neurosat) is TensorFlow 1.x and ships
*no pretrained weights*.

To make this adapter actually run you need to:

  1. Train a NeuroSAT checkpoint on your problem distribution
     (see ``scripts/toy_train.sh`` in the cloned repo), or download
     someone else's checkpoint;

  2. Point the adapter at the checkpoint via the ``checkpoint`` kwarg
     (or via the ``CNLM_NEUROSAT_CKPT`` environment variable).

If neither weights nor a working TF1 install is found, the adapter
reports unavailable with the reason — the benchmark just skips it.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

from .base import BaseAdapter, SolveOutcome

THIRD_PARTY_ROOT = Path(__file__).resolve().parent.parent.parent / "third_party"
NEUROSAT_DIR = THIRD_PARTY_ROOT / "neurosat"


class NeuroSATAdapter(BaseAdapter):
    name = "neurosat"
    kind = "neural"
    supports = {"SAT"}    # Selsam's NeuroSAT is SAT-classification only

    def __init__(self, checkpoint: str = None, n_rounds: int = 26, **kwargs):
        super().__init__(checkpoint=checkpoint, n_rounds=n_rounds, **kwargs)

    def available(self) -> bool:
        if not NEUROSAT_DIR.exists():
            self.unavailable_reason = (
                f"third_party/neurosat not present (expected {NEUROSAT_DIR})"
            )
            return False
        try:
            import tensorflow as tf  # noqa: F401
        except Exception as exc:
            self.unavailable_reason = (
                f"NeuroSAT requires TensorFlow 1.x (`pip install tensorflow==1.15`). "
                f"Import failed: {exc}"
            )
            return False
        # check version
        import tensorflow as tf
        if not tf.__version__.startswith("1."):
            self.unavailable_reason = (
                f"NeuroSAT requires TensorFlow 1.x but found {tf.__version__}. "
                f"Use a venv with `tensorflow==1.15`."
            )
            return False

        # locate checkpoint
        ckpt = self.config.get("checkpoint") or os.environ.get("CNLM_NEUROSAT_CKPT")
        if ckpt is None or not Path(ckpt).exists():
            self.unavailable_reason = (
                "NeuroSAT checkpoint not found.  Train one via "
                "`third_party/neurosat/scripts/toy_train.sh` and pass via "
                "`--neurosat-ckpt /path/to/ckpt` or env CNLM_NEUROSAT_CKPT."
            )
            return False
        self._ckpt = ckpt
        return True

    def solve(self, instance, timeout_s: float = 120.0) -> SolveOutcome:
        # Lazy import — we only get here if available() succeeded
        sys.path.insert(0, str(NEUROSAT_DIR / "python"))
        try:
            from neurosat import NeuroSAT
            from mk_problem import mk_batch_problem
            from solver import Solver
        except Exception as exc:
            return SolveOutcome(
                solver=self.name, instance=getattr(instance, "name", ""),
                problem_type="SAT", available=True,
                error=f"NeuroSAT import failed: {exc}",
            )

        t0 = time.perf_counter()
        try:
            # Build a single-instance batch in NeuroSAT's expected format
            # (NeuroSAT's mk_problem expects a list of (is_sat, iclauses))
            iclauses = [list(c) for c in instance.clauses]
            problem = mk_batch_problem(
                problems=[(True, iclauses)],
                n_vars=instance.n_vars,
            )
            # Restore the model
            opts = self._make_opts()
            net = NeuroSAT(opts)
            net.restore(opts.run_id, opts.restore_id, opts.restore_epoch)
            # Run inference; recover the assignment by clustering literal embeddings
            soln = net.solve(problem)
            # `soln` is a vector of bools length n
            x = np.asarray(soln, dtype=bool)[: instance.n_vars]
        except Exception as exc:
            return SolveOutcome(
                solver=self.name, instance=getattr(instance, "name", ""),
                problem_type="SAT", available=True,
                runtime_s=time.perf_counter() - t0,
                error=f"NeuroSAT runtime: {exc}",
            )

        runtime = time.perf_counter() - t0
        n_sat, _, _, _, _ = self._verify(x, instance)
        n_clauses = instance.n_clauses
        return SolveOutcome(
            solver=self.name, instance=getattr(instance, "name", ""),
            problem_type="SAT", available=True,
            runtime_s=runtime,
            is_SAT=(n_sat == n_clauses),
            n_satisfied=n_sat, n_clauses=n_clauses,
            sat_score=n_sat / max(n_clauses, 1),
            assignment=x.astype(int).tolist(),
            extras={"checkpoint": self._ckpt, "n_rounds": self.config["n_rounds"]},
        )

    def _make_opts(self):
        # Build the argparse-style namespace expected by NeuroSAT
        from argparse import Namespace
        return Namespace(
            run_id=None,
            restore_id=0,
            restore_epoch=0,
            n_rounds=int(self.config["n_rounds"]),
            n_saves_to_keep=1,
            d=128,
            commit_freq=1000,
            lr_start=2e-5, lr_end=1e-6, lr_decay_zero_by=2e5,
            l2_weight=1e-10,
            n_msg_layers=3, n_vote_layers=3,
            **{},
        )
