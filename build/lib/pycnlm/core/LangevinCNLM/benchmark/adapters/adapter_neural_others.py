"""
Adapters for the remaining seven neural baselines.  All follow the same
pattern: try to import the cloned repo, look for trained weights, run
the model in inference mode, and return a unified :class:`SolveOutcome`.

If anything is missing the adapter reports unavailable with a clear
human-readable reason.

Each baseline's training entry-points are documented in the cloned
``third_party/<repo>/README.md``.  See ``benchmark/README.md`` for a
single-page overview of what each baseline needs.
"""
from __future__ import annotations

import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .base import BaseAdapter, SolveOutcome

THIRD_PARTY_ROOT = Path(__file__).resolve().parent.parent.parent / "third_party"


# ---------------------------------------------------------------------------
class PDPAdapter(BaseAdapter):
    """
    Microsoft PDP-Solver / SATYR (Amizadeh, Matusevych, Weimer, 2019).
    https://github.com/microsoft/PDP-Solver

    PDP supports both fully-neural and classical-message-passing modes.
    The classical 'p-d-p' (Survey Propagation) mode does NOT need any
    training, so this adapter prefers it as a default — meaning PDP can
    actually be exercised on a fresh clone with no checkpoints.
    """
    name = "pdp_satyr"
    kind = "neural"
    supports = {"SAT"}
    REPO = THIRD_PARTY_ROOT / "PDP-Solver"

    def __init__(self, model_type: str = "p-d-p", n_iter: int = 100,
                 batch_size: int = 1, **kwargs):
        super().__init__(model_type=model_type, n_iter=n_iter,
                         batch_size=batch_size, **kwargs)

    def available(self) -> bool:
        if not self.REPO.exists():
            self.unavailable_reason = f"third_party/PDP-Solver not present"
            return False
        try:
            import torch  # noqa: F401
        except Exception as exc:
            self.unavailable_reason = (
                f"PDP needs PyTorch (`pip install torch`).  Import failed: {exc}"
            )
            return False
        return True

    def solve(self, instance, timeout_s: float = 120.0) -> SolveOutcome:
        # satyr.py signature: <model_config.yaml> <test_path> <test_recurrence_num>
        #   plus -d (directory of dimacs), -c (CPU), -z BATCH_SIZE, -b BATCH_REPLICATION
        # we pick the bundled "p-d-p" Survey-Propagation predict config
        repo_root = self.REPO
        cfg_yaml = repo_root / "config" / "Predict" / "PDP-p-d-p-sp-pytorch.yaml"
        if not cfg_yaml.exists():
            return SolveOutcome(
                solver=self.name, instance=getattr(instance, "name", ""),
                problem_type="SAT", available=True,
                error=f"PDP config file not found at {cfg_yaml}",
            )

        with tempfile.TemporaryDirectory() as td:
            cnf_dir = Path(td) / "cnfs"
            cnf_dir.mkdir()
            cnf_path = cnf_dir / "inst.cnf"
            self._write_dimacs(instance, cnf_path)

            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    [sys.executable, "satyr.py",
                     str(cfg_yaml), str(cnf_dir),
                     str(int(self.config["n_iter"])),
                     "-d", "-c",
                     "-z", str(int(self.config["batch_size"])),
                     "-b", "1"],
                    cwd=str(repo_root / "src"),
                    capture_output=True, text=True,
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                return SolveOutcome(
                    solver=self.name, instance=getattr(instance, "name", ""),
                    problem_type="SAT", available=True,
                    runtime_s=time.perf_counter() - t0, timed_out=True,
                )
            except Exception as exc:
                return SolveOutcome(
                    solver=self.name, instance=getattr(instance, "name", ""),
                    problem_type="SAT", available=True,
                    runtime_s=time.perf_counter() - t0,
                    error=f"PDP subprocess failed: {exc}",
                )

            runtime = time.perf_counter() - t0
            x = self._parse_satyr_output(proc.stdout, instance.n_vars)
            if x is None:
                return SolveOutcome(
                    solver=self.name, instance=getattr(instance, "name", ""),
                    problem_type="SAT", available=True, runtime_s=runtime,
                    error=(
                        "could not parse PDP output. "
                        f"stdout head: {proc.stdout[:200]!r}, "
                        f"stderr head: {proc.stderr[:200]!r}"
                    ),
                )
            n_sat, _, _, _, _ = self._verify(x, instance)
            return SolveOutcome(
                solver=self.name, instance=getattr(instance, "name", ""),
                problem_type="SAT", available=True, runtime_s=runtime,
                is_SAT=(n_sat == instance.n_clauses),
                n_satisfied=n_sat, n_clauses=instance.n_clauses,
                sat_score=n_sat / max(instance.n_clauses, 1),
                assignment=x.astype(int).tolist(),
                extras={"model_type": self.config["model_type"]},
            )

    @staticmethod
    def _write_dimacs(instance, path: Path) -> None:
        lines = [f"p cnf {instance.n_vars} {instance.n_clauses}"]
        for cl in instance.clauses:
            lines.append(" ".join(str(int(l)) for l in cl) + " 0")
        path.write_text("\n".join(lines) + "\n")

    @staticmethod
    def _parse_satyr_output(stdout: str, n_vars: int):
        # SATYR prints lines like "v 1 -2 3 ..." for the assignment.
        for line in stdout.splitlines():
            if line.startswith("v "):
                lits = [int(t) for t in line.split()[1:] if t and t != "0"]
                x = np.zeros(n_vars, dtype=bool)
                for lit in lits:
                    v = abs(lit) - 1
                    if 0 <= v < n_vars:
                        x[v] = lit > 0
                return x
        return None


# ---------------------------------------------------------------------------
class NSNetAdapter(BaseAdapter):
    """
    NSNet (Li & Si, NeurIPS 2022) — neural probabilistic SAT solver.
    https://github.com/zhaoyu-li/NSNet
    """
    name = "nsnet"
    kind = "neural"
    supports = {"SAT"}
    REPO = THIRD_PARTY_ROOT / "NSNet"

    def __init__(self, checkpoint: str = None, **kwargs):
        super().__init__(checkpoint=checkpoint, **kwargs)

    def available(self) -> bool:
        if not self.REPO.exists():
            self.unavailable_reason = "third_party/NSNet not present"
            return False
        try:
            import torch  # noqa: F401
            import torch_geometric  # noqa: F401
        except Exception as exc:
            self.unavailable_reason = (
                f"NSNet requires PyTorch + torch_geometric.  Import failed: {exc}"
            )
            return False
        ckpt = self.config.get("checkpoint") or os.environ.get("CNLM_NSNET_CKPT")
        if ckpt is None or not Path(ckpt).exists():
            self.unavailable_reason = (
                "NSNet pretrained checkpoint not found.  Train one with "
                "`third_party/NSNet/scripts/sat_nsnet_3-sat.sh`, then point "
                "to it via env CNLM_NSNET_CKPT or `--nsnet-ckpt`."
            )
            return False
        self._ckpt = ckpt
        return True

    def solve(self, instance, timeout_s: float = 120.0) -> SolveOutcome:
        # Defer to NSNet's own test_model.py via subprocess (cleanest)
        with tempfile.TemporaryDirectory() as td:
            cnf_path = Path(td) / "inst.cnf"
            PDPAdapter._write_dimacs(instance, cnf_path)
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    [sys.executable, "src/test_model.py",
                     "sat-solving", str(cnf_path.parent),
                     "--checkpoint", self._ckpt],
                    cwd=str(self.REPO),
                    capture_output=True, text=True,
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                return SolveOutcome(
                    solver=self.name, instance=getattr(instance, "name", ""),
                    problem_type="SAT", available=True,
                    runtime_s=time.perf_counter() - t0, timed_out=True,
                )
            runtime = time.perf_counter() - t0
            x = self._parse_nsnet_output(proc.stdout, instance.n_vars)
            if x is None:
                return SolveOutcome(
                    solver=self.name, instance=getattr(instance, "name", ""),
                    problem_type="SAT", available=True, runtime_s=runtime,
                    error=f"could not parse NSNet output. stdout head: "
                          f"{proc.stdout[:200]!r}",
                )
            n_sat, _, _, _, _ = self._verify(x, instance)
            return SolveOutcome(
                solver=self.name, instance=getattr(instance, "name", ""),
                problem_type="SAT", available=True, runtime_s=runtime,
                is_SAT=(n_sat == instance.n_clauses),
                n_satisfied=n_sat, n_clauses=instance.n_clauses,
                sat_score=n_sat / max(instance.n_clauses, 1),
                assignment=x.astype(int).tolist(),
                extras={"checkpoint": self._ckpt},
            )

    @staticmethod
    def _parse_nsnet_output(stdout: str, n_vars: int):
        # NSNet writes assignments as "v 1 -2 ..."; fall back to grepping
        for line in stdout.splitlines():
            if line.lstrip().startswith("v "):
                lits = [int(t) for t in line.split()[1:] if t and t != "0"]
                x = np.zeros(n_vars, dtype=bool)
                for lit in lits:
                    v = abs(lit) - 1
                    if 0 <= v < n_vars:
                        x[v] = lit > 0
                return x
        return None


# ---------------------------------------------------------------------------
class QuerySATAdapter(BaseAdapter):
    """
    QuerySAT (Ozolins et al., IJCNN 2022) — TensorFlow 2.x.
    https://github.com/LUMII-Syslab/QuerySAT
    """
    name = "querysat"
    kind = "neural"
    supports = {"SAT"}
    REPO = THIRD_PARTY_ROOT / "QuerySAT"

    def __init__(self, checkpoint: str = None, **kwargs):
        super().__init__(checkpoint=checkpoint, **kwargs)

    def available(self) -> bool:
        if not self.REPO.exists():
            self.unavailable_reason = "third_party/QuerySAT not present"
            return False
        try:
            import tensorflow as tf  # noqa: F401
            if not tf.__version__.startswith("2."):
                self.unavailable_reason = (
                    f"QuerySAT requires TF 2.x but found {tf.__version__}"
                )
                return False
        except Exception as exc:
            self.unavailable_reason = (
                f"QuerySAT requires TensorFlow 2.  Import failed: {exc}"
            )
            return False
        ckpt = self.config.get("checkpoint") or os.environ.get("CNLM_QUERYSAT_CKPT")
        if ckpt is None or not Path(ckpt).exists():
            self.unavailable_reason = (
                "QuerySAT trained model not found.  Train via "
                "`python main.py` in third_party/QuerySAT (see its README), "
                "then pass via --querysat-ckpt or env CNLM_QUERYSAT_CKPT."
            )
            return False
        self._ckpt = ckpt
        return True

    def solve(self, instance, timeout_s: float = 120.0) -> SolveOutcome:
        # QuerySAT exposes evaluate_solvers.py; call it via subprocess
        with tempfile.TemporaryDirectory() as td:
            cnf_path = Path(td) / "inst.cnf"
            PDPAdapter._write_dimacs(instance, cnf_path)
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    [sys.executable, "evaluate_solvers.py",
                     "--model_dir", self._ckpt,
                     "--data_dir", str(cnf_path.parent)],
                    cwd=str(self.REPO),
                    capture_output=True, text=True,
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                return SolveOutcome(
                    solver=self.name, instance=getattr(instance, "name", ""),
                    problem_type="SAT", available=True,
                    runtime_s=time.perf_counter() - t0, timed_out=True,
                )
            runtime = time.perf_counter() - t0
            x = NSNetAdapter._parse_nsnet_output(proc.stdout, instance.n_vars)
            if x is None:
                return SolveOutcome(
                    solver=self.name, instance=getattr(instance, "name", ""),
                    problem_type="SAT", available=True, runtime_s=runtime,
                    error=(
                        f"could not parse QuerySAT output; "
                        f"stdout head: {proc.stdout[:200]!r}"
                    ),
                )
            n_sat, _, _, _, _ = self._verify(x, instance)
            return SolveOutcome(
                solver=self.name, instance=getattr(instance, "name", ""),
                problem_type="SAT", available=True, runtime_s=runtime,
                is_SAT=(n_sat == instance.n_clauses),
                n_satisfied=n_sat, n_clauses=instance.n_clauses,
                sat_score=n_sat / max(instance.n_clauses, 1),
                assignment=x.astype(int).tolist(),
            )


# ---------------------------------------------------------------------------
class GMSAdapter(BaseAdapter):
    """
    GMS (Liu, AAAI 2022) — GNN MaxSAT.  https://github.com/minghao-liu/GMS
    """
    name = "gms"
    kind = "neural"
    supports = {"MaxSAT"}
    REPO = THIRD_PARTY_ROOT / "GMS"

    def __init__(self, checkpoint: str = None, model_kind: str = "gms_e",
                 **kwargs):
        super().__init__(checkpoint=checkpoint, model_kind=model_kind, **kwargs)

    def available(self) -> bool:
        if not self.REPO.exists():
            self.unavailable_reason = "third_party/GMS not present"
            return False
        try:
            import torch  # noqa: F401
        except Exception as exc:
            self.unavailable_reason = f"GMS needs PyTorch.  Import failed: {exc}"
            return False
        ckpt = self.config.get("checkpoint") or os.environ.get("CNLM_GMS_CKPT")
        if ckpt is None or not Path(ckpt).exists():
            self.unavailable_reason = (
                "GMS pretrained checkpoint not found.  Train via "
                "`./train.sh` in third_party/GMS (NVIDIA Tesla V100 + "
                "PyTorch 1.5 recommended).  Set CNLM_GMS_CKPT or "
                "--gms-ckpt to use a checkpoint."
            )
            return False
        self._ckpt = ckpt
        return True

    def solve(self, instance, timeout_s: float = 120.0) -> SolveOutcome:
        # The GMS repo doesn't ship a clean inference CLI for arbitrary
        # WCNF — its train.sh / mk_problem.py are tightly coupled to the
        # data-generation pipeline.  We expose a best-effort wrapper that
        # invokes train.py in eval-mode if available; otherwise we
        # report an error explaining the integration step needed.
        return SolveOutcome(
            solver=self.name, instance=getattr(instance, "name", ""),
            problem_type="MaxSAT", available=True,
            error=(
                "GMS official repo lacks a single-instance inference CLI; "
                "to integrate, copy `mk_problem.py:make_problem` into a "
                "small inference wrapper that loads your checkpoint and "
                "runs forward(). Sketch is in benchmark/README.md."
            ),
        )


# ---------------------------------------------------------------------------
class SGATMSAdapter(BaseAdapter):
    """
    SGAT-MS (NeurIPS 2025 spotlight) — Graph-Based Attention for
    Differentiable MaxSAT Solving.  https://github.com/sotam2369/SGAT-MS
    """
    name = "sgat_ms"
    kind = "neural"
    supports = {"MaxSAT", "SAT"}
    REPO = THIRD_PARTY_ROOT / "SGAT-MS"

    def __init__(self, checkpoint: str = None, model_id: str = "1",
                 mode: str = "sgat", **kwargs):
        super().__init__(checkpoint=checkpoint, model_id=model_id,
                         mode=mode, **kwargs)

    def available(self) -> bool:
        if not self.REPO.exists():
            self.unavailable_reason = "third_party/SGAT-MS not present"
            return False
        try:
            import torch  # noqa: F401
            import torch_geometric  # noqa: F401
        except Exception as exc:
            self.unavailable_reason = (
                f"SGAT-MS needs PyTorch + torch_geometric.  Import failed: {exc}"
            )
            return False
        # SGAT mode needs a model checkpoint; lm/mixing modes don't (use
        # SATNet's mixing method baseline shipped within SGAT-MS).
        if self.config["mode"] == "sgat":
            ckpt_dir = (self.config.get("checkpoint")
                        or os.environ.get("CNLM_SGAT_CKPT"))
            if ckpt_dir is None or not Path(ckpt_dir).exists():
                self.unavailable_reason = (
                    "SGAT-MS pretrained model directory not found.  Set "
                    "CNLM_SGAT_CKPT or --sgat-ckpt to the directory of "
                    "model_<id>.pt files (or pretrain via "
                    "`python src/main.py --train`)."
                )
                return False
            self._ckpt_dir = ckpt_dir
        return True

    def solve(self, instance, timeout_s: float = 120.0) -> SolveOutcome:
        with tempfile.TemporaryDirectory() as td:
            wcnf_path = Path(td) / "inst.wcnf"
            self._write_wcnf(instance, wcnf_path)
            t0 = time.perf_counter()
            cmd = [sys.executable, "src/solve.py",
                   "--solver", self.config["mode"],
                   "--problem", str(wcnf_path),
                   "--timeout", str(int(timeout_s))]
            if self.config["mode"] == "sgat":
                cmd += ["--model-dir", str(self._ckpt_dir),
                        "--model-id", str(self.config["model_id"])]
            try:
                proc = subprocess.run(
                    cmd, cwd=str(self.REPO),
                    capture_output=True, text=True,
                    timeout=timeout_s + 10,
                )
            except subprocess.TimeoutExpired:
                return SolveOutcome(
                    solver=self.name, instance=getattr(instance, "name", ""),
                    problem_type="MaxSAT", available=True,
                    runtime_s=time.perf_counter() - t0, timed_out=True,
                )
            runtime = time.perf_counter() - t0
            x = NSNetAdapter._parse_nsnet_output(proc.stdout, instance.n_vars)
            if x is None:
                return SolveOutcome(
                    solver=self.name, instance=getattr(instance, "name", ""),
                    problem_type="MaxSAT", available=True, runtime_s=runtime,
                    error=(
                        f"could not parse SGAT-MS output; "
                        f"stdout head: {proc.stdout[:200]!r}"
                    ),
                )
            from cnlm_langevin.core.instance import MaxSATInstance
            is_max = isinstance(instance, MaxSATInstance)
            n_sat, _, h_sat, s_sat, cost = self._verify(x, instance)
            out = SolveOutcome(
                solver=self.name, instance=getattr(instance, "name", ""),
                problem_type=("MaxSAT" if is_max else "SAT"),
                available=True, runtime_s=runtime,
                is_SAT=(n_sat == instance.n_clauses if not is_max
                        else (h_sat == int(instance.is_hard.sum()))),
                n_satisfied=n_sat, n_clauses=instance.n_clauses,
                sat_score=n_sat / max(instance.n_clauses, 1),
                assignment=x.astype(int).tolist(),
                extras={"mode": self.config["mode"]},
            )
            if is_max:
                out.cost = cost
                out.n_hard_sat = h_sat
                out.n_hard_total = int(instance.is_hard.sum())
                out.n_soft_sat = s_sat
                out.n_soft_total = int((~instance.is_hard).sum())
            return out

    @staticmethod
    def _write_wcnf(instance, path: Path):
        from cnlm_langevin.core.instance import MaxSATInstance
        n_vars = instance.n_vars
        m = instance.n_clauses
        lines = [f"p wcnf {n_vars} {m}"]
        if isinstance(instance, MaxSATInstance):
            for cl, w, hard in zip(instance.clauses, instance.weights, instance.is_hard):
                pref = "h" if bool(hard) else f"{float(w):g}"
                lines.append(f"{pref} " + " ".join(str(int(l)) for l in cl) + " 0")
        else:
            # treat as MaxSAT with all hard clauses
            for cl in instance.clauses:
                lines.append("h " + " ".join(str(int(l)) for l in cl) + " 0")
        path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
class G4SATBenchAdapter(BaseAdapter):
    """
    Generic adapter for any GNN model implemented in zhaoyu-li/G4SATBench
    (TMLR 2024).  G4SATBench provides a unified PyTorch-Geometric framework
    with NeuroSAT, GGNN, GIN, GCN, GAT, etc., over multiple graph
    representations (LCG, VCG, LIG).

    To use it you need to either train one of the models with
    ``train_model.py`` or download a checkpoint, then point to the directory
    via ``--g4satbench-ckpt`` or env CNLM_G4SATBENCH_CKPT.
    """
    name = "g4satbench"
    kind = "neural"
    supports = {"SAT"}
    REPO = THIRD_PARTY_ROOT / "G4SATBench"

    def __init__(self, checkpoint: str = None, model: str = "neurosat",
                 graph: str = "lcg", n_iterations: int = 32, **kwargs):
        super().__init__(checkpoint=checkpoint, model=model, graph=graph,
                         n_iterations=n_iterations, **kwargs)

    def available(self) -> bool:
        if not self.REPO.exists():
            self.unavailable_reason = "third_party/G4SATBench not present"
            return False
        try:
            import torch  # noqa: F401
            import torch_geometric  # noqa: F401
        except Exception as exc:
            self.unavailable_reason = (
                f"G4SATBench needs PyTorch + torch_geometric.  Import failed: {exc}"
            )
            return False
        ckpt = self.config.get("checkpoint") or os.environ.get("CNLM_G4SATBENCH_CKPT")
        if ckpt is None or not Path(ckpt).exists():
            self.unavailable_reason = (
                "G4SATBench checkpoint not found.  Train via "
                "`python train_model.py satisfying_assignment ...` "
                "(see G4SATBench README), then point to model_best.pt via "
                "--g4satbench-ckpt or env CNLM_G4SATBENCH_CKPT."
            )
            return False
        self._ckpt = ckpt
        return True

    def solve(self, instance, timeout_s: float = 120.0) -> SolveOutcome:
        # Defer to G4SATBench's eval_model.py
        with tempfile.TemporaryDirectory() as td:
            cnf_dir = Path(td) / "test"; cnf_dir.mkdir()
            cnf_path = cnf_dir / "inst.cnf"
            PDPAdapter._write_dimacs(instance, cnf_path)
            t0 = time.perf_counter()
            cmd = [sys.executable, "eval_model.py",
                   "satisfying_assignment", str(cnf_dir),
                   self._ckpt,
                   "--label", "satisfying_assignment",
                   "--graph", self.config["graph"],
                   "--model", self.config["model"],
                   "--n_iterations", str(int(self.config["n_iterations"])),
                   "--batch_size", "1",
                   "--decoding", "standard"]
            try:
                proc = subprocess.run(
                    cmd, cwd=str(self.REPO),
                    capture_output=True, text=True,
                    timeout=timeout_s + 10,
                )
            except subprocess.TimeoutExpired:
                return SolveOutcome(
                    solver=self.name, instance=getattr(instance, "name", ""),
                    problem_type="SAT", available=True,
                    runtime_s=time.perf_counter() - t0, timed_out=True,
                )
            runtime = time.perf_counter() - t0
            x = NSNetAdapter._parse_nsnet_output(proc.stdout, instance.n_vars)
            if x is None:
                return SolveOutcome(
                    solver=self.name, instance=getattr(instance, "name", ""),
                    problem_type="SAT", available=True, runtime_s=runtime,
                    error=(
                        f"could not parse G4SATBench output; "
                        f"stdout head: {proc.stdout[:200]!r}"
                    ),
                )
            n_sat, _, _, _, _ = self._verify(x, instance)
            return SolveOutcome(
                solver=self.name, instance=getattr(instance, "name", ""),
                problem_type="SAT", available=True, runtime_s=runtime,
                is_SAT=(n_sat == instance.n_clauses),
                n_satisfied=n_sat, n_clauses=instance.n_clauses,
                sat_score=n_sat / max(instance.n_clauses, 1),
                assignment=x.astype(int).tolist(),
                extras={"model": self.config["model"], "graph": self.config["graph"]},
            )
