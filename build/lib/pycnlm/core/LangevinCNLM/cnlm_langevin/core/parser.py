"""
DIMACS CNF / WCNF parser.

Supports:
* Standard CNF:               header  "p cnf <nvars> <nclauses>", clauses end with 0.
* Old WCNF (pre-2022):        header  "p wcnf <nvars> <nclauses> <top>", each line
                              "<weight> <lit1> <lit2> ... 0".  Hard clauses have weight == top.
* New WCNF (MSE 2022+):       no header, lines start with 'h' (hard) or a positive
                              weight (soft) followed by literals and 0.
Comment lines starting with 'c' and blank lines are ignored.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Union


class DimacsParseError(ValueError):
    """Raised when a DIMACS file cannot be parsed."""


@dataclass
class ParsedCNF:
    n_vars: int
    n_clauses: int
    clauses: List[List[int]]   # signed integer literals, each clause without trailing 0
    raw_header: Optional[str] = None


@dataclass
class ParsedWCNF:
    n_vars: int
    n_clauses: int
    top: float                          # weight that marks hard clauses (np.inf for new format)
    clauses: List[List[int]]            # signed-int literal lists
    weights: List[float]                # weight per clause (== top for hard)
    is_hard: List[bool]                 # convenience flag
    raw_header: Optional[str] = None
    new_format: bool = False


# ----------------------------------------------------------------------------- helpers
def _strip_iter(path: Union[str, Path]):
    """Yield non-empty, non-comment lines (with line numbers)."""
    p = Path(path)
    if not p.exists():
        raise DimacsParseError(f"File not found: {p}")
    with p.open("r", errors="replace") as f:
        for ln, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("c") or line.startswith("%"):
                continue
            yield ln, line


def _parse_clause_tokens(tokens, lineno: int) -> List[int]:
    """Convert a list of integer tokens (terminated by 0) to a clause."""
    if not tokens:
        raise DimacsParseError(f"line {lineno}: empty clause")
    if tokens[-1] != 0:
        raise DimacsParseError(f"line {lineno}: clause must end with 0")
    if 0 in tokens[:-1]:
        raise DimacsParseError(f"line {lineno}: stray 0 in middle of clause")
    return tokens[:-1]


# ----------------------------------------------------------------------------- CNF
def parse_dimacs_cnf(path: Union[str, Path]) -> ParsedCNF:
    """Parse a standard DIMACS CNF file."""
    header = None
    clauses: List[List[int]] = []
    declared_vars = declared_clauses = None
    pending_buf: List[int] = []  # to handle clauses spanning multiple lines

    for lineno, line in _strip_iter(path):
        if line.startswith("p"):
            parts = line.split()
            if len(parts) < 4 or parts[1].lower() != "cnf":
                raise DimacsParseError(f"line {lineno}: expected 'p cnf <nv> <nc>', got '{line}'")
            try:
                declared_vars = int(parts[2])
                declared_clauses = int(parts[3])
            except ValueError as e:
                raise DimacsParseError(f"line {lineno}: bad header: {e}")
            header = line
            continue

        try:
            toks = [int(t) for t in line.split()]
        except ValueError as e:
            raise DimacsParseError(f"line {lineno}: non-integer token: {e}")

        pending_buf.extend(toks)
        # consume any complete clauses (terminated by 0)
        while 0 in pending_buf:
            idx = pending_buf.index(0)
            clause = pending_buf[:idx]
            pending_buf = pending_buf[idx + 1 :]
            if not clause:
                # empty clause => unsatisfiable instance, but record it
                clauses.append([])
            else:
                clauses.append(clause)

    if pending_buf:
        raise DimacsParseError("dangling literals not terminated by 0")

    if declared_vars is None:
        # tolerate header-less files; infer
        declared_vars = max((abs(l) for cl in clauses for l in cl), default=0)
        declared_clauses = len(clauses)

    # consistency check (warn-only via slight tolerance, as some files mis-state)
    if declared_clauses is not None and len(clauses) != declared_clauses:
        # Don't raise — many real files are inconsistent; trust the body.
        pass

    return ParsedCNF(
        n_vars=declared_vars,
        n_clauses=len(clauses),
        clauses=clauses,
        raw_header=header,
    )


# ----------------------------------------------------------------------------- WCNF
def parse_dimacs_wcnf(path: Union[str, Path]) -> ParsedWCNF:
    """Parse a Weighted CNF file (old or new MaxSAT format)."""
    header = None
    declared_vars = declared_clauses = None
    declared_top: Optional[float] = None
    new_format = False

    clauses: List[List[int]] = []
    weights: List[float] = []
    is_hard: List[bool] = []

    for lineno, line in _strip_iter(path):
        # ----- header (old format only)
        if line.startswith("p"):
            parts = line.split()
            if len(parts) >= 4 and parts[1].lower() == "wcnf":
                try:
                    declared_vars = int(parts[2])
                    declared_clauses = int(parts[3])
                    if len(parts) >= 5:
                        declared_top = float(parts[4])
                    else:
                        declared_top = float("inf")
                    header = line
                    continue
                except ValueError as e:
                    raise DimacsParseError(f"line {lineno}: bad WCNF header: {e}")
            else:
                raise DimacsParseError(f"line {lineno}: expected 'p wcnf', got '{line}'")

        toks = line.split()
        if not toks:
            continue

        # ----- new-format hard clause: 'h ... 0'
        if toks[0].lower() == "h":
            new_format = True
            try:
                lits = [int(t) for t in toks[1:]]
            except ValueError as e:
                raise DimacsParseError(f"line {lineno}: non-integer literal: {e}")
            clause = _parse_clause_tokens(lits, lineno)
            clauses.append(clause)
            weights.append(float("inf"))
            is_hard.append(True)
            continue

        # ----- otherwise, first token is the weight
        try:
            w = float(toks[0])
        except ValueError as e:
            raise DimacsParseError(f"line {lineno}: cannot parse weight: {e}")

        try:
            lits = [int(t) for t in toks[1:]]
        except ValueError as e:
            raise DimacsParseError(f"line {lineno}: non-integer literal: {e}")

        clause = _parse_clause_tokens(lits, lineno)

        # mark hard if matches declared top (old format only)
        hard = False
        if declared_top is not None and not new_format:
            hard = (w >= declared_top - 1e-12)
        clauses.append(clause)
        weights.append(w)
        is_hard.append(hard)

    if declared_vars is None:
        declared_vars = max((abs(l) for cl in clauses for l in cl), default=0)
        declared_clauses = len(clauses)

    if declared_top is None:
        declared_top = float("inf")
        new_format = True

    return ParsedWCNF(
        n_vars=declared_vars,
        n_clauses=len(clauses),
        top=declared_top,
        clauses=clauses,
        weights=weights,
        is_hard=is_hard,
        raw_header=header,
        new_format=new_format,
    )


# ----------------------------------------------------------------------------- auto
def parse_dimacs_auto(path: Union[str, Path]) -> Union[ParsedCNF, ParsedWCNF]:
    """Dispatch on file extension."""
    p = Path(path)
    suff = p.suffix.lower()
    if suff in (".wcnf",):
        return parse_dimacs_wcnf(p)
    if suff in (".cnf", ".dimacs", ".sat"):
        return parse_dimacs_cnf(p)
    # try to sniff
    try:
        with p.open("r", errors="replace") as f:
            head = f.read(2048).lower()
        if "p wcnf" in head or "\nh " in head or head.startswith("h "):
            return parse_dimacs_wcnf(p)
        return parse_dimacs_cnf(p)
    except OSError as e:
        raise DimacsParseError(f"cannot read {p}: {e}")
