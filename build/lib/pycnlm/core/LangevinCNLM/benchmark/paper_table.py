r"""
Paper-style results table for benchmark output.

Renders a publication-quality summary table mimicking the visual style of
modern ML papers (e.g. VL-JEPA's video-benchmarks table):

* booktabs-style top / mid / bottom rules
* grouped column headers with a horizontal span line ("Video Classification ..."-style)
* vertical column labels with tiny grey citation strings underneath
* row groups labelled by family (Classical / Neural / Ours)
* bold + underlined best entry per column (excluding the grey "Previous SoTA" row)
* blue-tinted "ours" row, grey-italic "previous SoTA" row

Two outputs are produced from a single call:

* ``<out>.pdf``  — matplotlib-rendered preview, no LaTeX needed
* ``<out>.tex``  — booktabs LaTeX source, drop straight into a paper

The tex file is self-contained: paste it inside any ``article`` document with
``\usepackage{booktabs,multirow,colortbl,xcolor}`` to compile.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


# ============================================================== adapter meta
# Per-method metadata that doesn't fit naturally on the Adapter class itself.
# Used to populate the "Trained?", "Generalist?", and "Hyperparams" columns.
_ADAPTER_META: Dict[str, Dict[str, Any]] = {
    "cnlm_langevin": {
        "family": "Ours",
        "trained": False,
        "generalist": True,
        "cite": "this work",
    },
    "neurosat_mini": {
        "family": "Neural (trained)",
        "trained": True,
        "generalist": False,
        "cite": "in-package, trained on 3-SAT (n=8–30)",
    },
    "survey_propagation": {
        "family": "Classical (SLS/MP)",
        "trained": False,
        "generalist": True,
        "cite": "Mézard et al. 2002",
    },
    "simulated_annealing": {
        "family": "Classical (SLS/MP)",
        "trained": False,
        "generalist": True,
        "cite": "Kirkpatrick et al. 1983",
    },
    "walksat": {
        "family": "Classical (SLS/MP)",
        "trained": False,
        "generalist": True,
        "cite": "Selman et al. 1994",
    },
    "random_restart_greedy": {
        "family": "Classical (SLS/MP)",
        "trained": False,
        "generalist": True,
        "cite": "—",
    },
    # ----- PySAT CDCL backends (all "Classical (CDCL)" family)
    "pysat_cd":   {"family": "Classical (CDCL)", "trained": False,
                   "generalist": True, "cite": "Biere 2017 (CaDiCaL)"},
    "pysat_cd15": {"family": "Classical (CDCL)", "trained": False,
                   "generalist": True, "cite": "Biere 2024 (CaDiCaL 1.5)"},
    "pysat_gc3":  {"family": "Classical (CDCL)", "trained": False,
                   "generalist": True, "cite": "Audemard & Simon 2009 (Glucose3)"},
    "pysat_gc4":  {"family": "Classical (CDCL)", "trained": False,
                   "generalist": True, "cite": "Audemard & Simon 2018 (Glucose4)"},
    "pysat_g3":   {"family": "Classical (CDCL)", "trained": False,
                   "generalist": True, "cite": "Audemard & Simon 2009"},
    "pysat_g4":   {"family": "Classical (CDCL)", "trained": False,
                   "generalist": True, "cite": "Audemard & Simon 2018"},
    "pysat_lgl":  {"family": "Classical (CDCL)", "trained": False,
                   "generalist": True, "cite": "Biere 2017 (Lingeling)"},
    "pysat_m22":  {"family": "Classical (CDCL)", "trained": False,
                   "generalist": True, "cite": "Eén & Sörensson 2003 (MiniSat)"},
    "pysat_mc":   {"family": "Classical (CDCL)", "trained": False,
                   "generalist": True, "cite": "Liffiton & Maglalang 2012 (MiniCard)"},
    "pysat_mgh":  {"family": "Classical (CDCL)", "trained": False,
                   "generalist": True, "cite": "Eén & Sörensson 2018 (MiniSat-GH)"},
    "pysat_mcb":  {"family": "Classical (CDCL)", "trained": False,
                   "generalist": True, "cite": "Liang et al. 2017 (MapleSAT-DBQAS)"},
    "pysat_mcm":  {"family": "Classical (CDCL)", "trained": False,
                   "generalist": True, "cite": "Liang et al. 2016 (MapleSAT-LRB)"},
    "pysat_mpl":  {"family": "Classical (CDCL)", "trained": False,
                   "generalist": True, "cite": "Liang et al. 2018 (MapleSAT)"},
    # ----- PySAT MaxSAT solver (RC2)
    "pysat_rc2":  {"family": "Classical (MaxSAT)", "trained": False,
                   "generalist": True, "cite": "Ignatiev et al. 2019 (RC2)"},
    "satnet_sdp_numpy": {
        "family": "Neural (untrained)",
        "trained": False,
        "generalist": True,
        "cite": "Wang et al. 2019",
    },
    "satnet_official": {
        "family": "Neural (untrained)",
        "trained": False,
        "generalist": True,
        "cite": "Wang et al. 2019",
    },
    "neurosat": {
        "family": "Neural (trained)",
        "trained": True,
        "generalist": False,
        "cite": "Selsam et al. 2019",
    },
    "pdp_satyr": {
        "family": "Neural (untrained)",
        "trained": False,
        "generalist": True,
        "cite": "Amizadeh et al. 2019",
    },
    "nsnet": {
        "family": "Neural (trained)",
        "trained": True,
        "generalist": False,
        "cite": "Li & Si 2022",
    },
    "querysat": {
        "family": "Neural (trained)",
        "trained": True,
        "generalist": False,
        "cite": "Ozolins et al. 2022",
    },
    "gms": {
        "family": "Neural (trained)",
        "trained": True,
        "generalist": False,
        "cite": "Liu 2022",
    },
    "sgat_ms": {
        "family": "Neural (trained)",
        "trained": True,
        "generalist": False,
        "cite": "NeurIPS 2025",
    },
    "g4satbench": {
        "family": "Neural (trained)",
        "trained": True,
        "generalist": False,
        "cite": "Li et al. 2024",
    },
}

# Order in which families appear top-to-bottom in the table
_FAMILY_ORDER = [
    "Classical (CDCL)",
    "Classical (MaxSAT)",
    "Classical (SLS/MP)",
    "Neural (untrained)",
    "Neural (trained)",
    "Ours",
]


# ===================================================================== spec
@dataclass
class Column:
    name: str                     # e.g. "Average", "easy", "uf50-218"
    cite: str = ""                # tiny grey citation under the header
    is_average: bool = False


@dataclass
class ColumnGroup:
    label: str                    # e.g. "SAT solving (% fully solved)"
    higher_is_better: bool        # True for sat-score, False for cost / runtime
    cols: List[Column] = field(default_factory=list)


@dataclass
class TableRow:
    method: str                   # display name
    family: str                   # row-group key
    available: bool               # if False, all values are "—"
    trained: Optional[bool]
    generalist: Optional[bool]
    hyperparams: str              # short string, e.g. "1500 steps · 24 chains"
    values: Dict[Tuple[str, str], Optional[float]]   # (group_label, col_name) → value
    style: str = "normal"         # "normal" | "ours" | "gray"
    unavailable_reason: Optional[str] = None   # concise explanation for empty rows


@dataclass
class TableSpec:
    title: str
    subtitle: str
    col_groups: List[ColumnGroup]
    rows: List[TableRow]


# =================================================== build a TableSpec
def build_paper_table_spec(
    raw_rows: List[dict],
    problem_type: str,
    instance_groups: Optional[Dict[str, List[str]]] = None,
    hyperparam_strings: Optional[Dict[str, str]] = None,
) -> TableSpec:
    """
    Convert benchmark driver rows into a :class:`TableSpec`.

    Parameters
    ----------
    raw_rows
        List of dicts as written into ``results_per_instance.csv``.
    problem_type
        ``"SAT"`` or ``"MaxSAT"``.
    instance_groups
        Optional mapping ``{group_name: [instance_filename, ...]}``.  If
        provided, one column per group (plus an Average) is rendered.
        If ``None`` we just render an Average column.
    hyperparam_strings
        Optional ``{adapter_name: short_str}`` to fill the Hyperparams column.
    """
    hyperparam_strings = hyperparam_strings or {}

    # -------- determine columns ------------------------------------------------
    if instance_groups:
        group_names = list(instance_groups.keys())
        instance_to_group = {}
        for gname, files in instance_groups.items():
            for f in files:
                instance_to_group[f] = gname
    else:
        group_names = []
        instance_to_group = {}

    def _make_cols(label: str, higher: bool) -> ColumnGroup:
        cols = [Column(name="Avg", is_average=True)]
        for g in group_names:
            cols.append(Column(name=g))
        return ColumnGroup(label=label, higher_is_better=higher, cols=cols)

    if problem_type == "SAT":
        col_groups = [
            _make_cols("Solved (%)",   higher=True),
            _make_cols("Sat-score",    higher=True),
            _make_cols("Runtime (s)",  higher=False),
        ]
    else:
        col_groups = [
            _make_cols("Hard-SAT (%)", higher=True),
            _make_cols("Sat-score",    higher=True),
            _make_cols("Cost",         higher=False),
            _make_cols("Runtime (s)",  higher=False),
        ]

    # -------- aggregate per (solver, group) -----------------------------------
    by_solver: Dict[str, List[dict]] = {}
    for r in raw_rows:
        by_solver.setdefault(r["solver"], []).append(r)

    rows: List[TableRow] = []
    for solver, items in by_solver.items():
        meta = _ADAPTER_META.get(solver, {
            "family": "Other", "trained": None, "generalist": None, "cite": "",
        })
        avail = [r for r in items if r.get("available")]
        ok = [r for r in avail if not r.get("error") and not r.get("timed_out")]

        # Pre-bucket by group_name (or "Avg" if no grouping)
        def _bucket_of(r):
            inst = r.get("instance", "")
            return instance_to_group.get(inst)

        # Per-cell aggregations
        values: Dict[Tuple[str, str], Optional[float]] = {}

        def _agg_metric(extract, agg="mean"):
            """Return {group_name: value} including 'Avg'."""
            out = {}
            xs_all = [extract(r) for r in ok if extract(r) is not None]
            if xs_all:
                out["Avg"] = float(np.mean(xs_all)) if agg == "mean" \
                    else float(np.sum(xs_all) / max(len(items), 1)) * 100.0
            else:
                out["Avg"] = None
            for g in group_names:
                xs = [extract(r) for r in ok
                      if _bucket_of(r) == g and extract(r) is not None]
                # for "Solved (%)" / "Hard-SAT (%)" we want the fraction in this
                # group, not just over successful runs
                if agg == "fraction_solved":
                    n_in_group = sum(1 for r in items if _bucket_of(r) == g)
                    if n_in_group:
                        out[g] = 100.0 * sum(1 for r in ok
                                             if _bucket_of(r) == g
                                             and extract(r) == 1) / n_in_group
                    else:
                        out[g] = None
                else:
                    out[g] = float(np.mean(xs)) if xs else None
            return out

        if problem_type == "SAT":
            # Solved (%): full-SAT fraction
            solved_pct_avg = (
                100.0 * sum(1 for r in ok if r.get("is_SAT")) / max(len(items), 1)
            ) if items else None
            # per-group: fraction in that group
            v_solved = {"Avg": solved_pct_avg}
            for g in group_names:
                n_in_group = sum(1 for r in items if _bucket_of(r) == g)
                if n_in_group:
                    v_solved[g] = (
                        100.0 * sum(1 for r in ok
                                    if _bucket_of(r) == g and r.get("is_SAT"))
                        / n_in_group
                    )
                else:
                    v_solved[g] = None
            for cn, val in v_solved.items():
                values[("Solved (%)", cn)] = val

            v_score = _agg_metric(lambda r: r.get("sat_score"))
            for cn, val in v_score.items():
                values[("Sat-score", cn)] = val

            v_rt = _agg_metric(lambda r: r.get("runtime_s"))
            for cn, val in v_rt.items():
                values[("Runtime (s)", cn)] = val
        else:
            # MaxSAT: full-hard-SAT fraction
            def _is_hard_sat(r):
                ht = r.get("n_hard_total")
                hs = r.get("n_hard_sat")
                if ht is None or hs is None:
                    return None
                return 1 if (ht == 0 or hs == ht) else 0

            v_hard = {"Avg": (
                100.0 * sum(1 for r in ok
                            if _is_hard_sat(r) == 1) / max(len(items), 1)
            ) if items else None}
            for g in group_names:
                n_in_group = sum(1 for r in items if _bucket_of(r) == g)
                if n_in_group:
                    v_hard[g] = (
                        100.0 * sum(1 for r in ok
                                    if _bucket_of(r) == g
                                    and _is_hard_sat(r) == 1)
                        / n_in_group
                    )
                else:
                    v_hard[g] = None
            for cn, val in v_hard.items():
                values[("Hard-SAT (%)", cn)] = val

            v_score = _agg_metric(lambda r: r.get("sat_score"))
            for cn, val in v_score.items():
                values[("Sat-score", cn)] = val

            v_cost = _agg_metric(lambda r: r.get("cost"))
            for cn, val in v_cost.items():
                values[("Cost", cn)] = val

            v_rt = _agg_metric(lambda r: r.get("runtime_s"))
            for cn, val in v_rt.items():
                values[("Runtime (s)", cn)] = val

        is_avail = (len(avail) > 0)

        # collect the unavailable reason if every run for this solver is unavailable
        unavail_reason = None
        if not is_avail and items:
            reasons = [r.get("unavailable_reason") for r in items
                       if r.get("unavailable_reason")]
            unavail_reason = reasons[0] if reasons else None
            # condense long messages
            if unavail_reason:
                unavail_reason = _condense_reason(unavail_reason)
        # also: if available but every run errored
        elif is_avail and not ok and avail:
            errs = [r.get("error") for r in avail if r.get("error")]
            if errs:
                unavail_reason = "(runtime error) " + _condense_reason(errs[0])

        rows.append(TableRow(
            method=solver,
            family=meta["family"],
            available=is_avail,
            trained=meta.get("trained"),
            generalist=meta.get("generalist"),
            hyperparams=hyperparam_strings.get(solver, ""),
            values=values,
            style=("ours" if meta["family"] == "Ours"
                   else ("gray" if not is_avail else "normal")),
            unavailable_reason=unavail_reason,
        ))

    # -------- sort rows by family order, then by method name within
    rows.sort(key=lambda r: (
        _FAMILY_ORDER.index(r.family) if r.family in _FAMILY_ORDER else 99,
        not r.available,    # available first within a family
        r.method,
    ))

    title_map = {
        "SAT": "CNLM-Langevin SAT benchmark — comparison with neural and classical solvers",
        "MaxSAT": "CNLM-Langevin MaxSAT benchmark — comparison with neural and classical solvers",
    }
    n_inst = len({r["instance"] for r in raw_rows})
    sub = (f"{n_inst} instance(s)"
           + (f", {len(group_names)} group(s)" if group_names else "")
           + f"; bold + underlined = best per column among available solvers.")

    return TableSpec(
        title=title_map.get(problem_type, problem_type),
        subtitle=sub,
        col_groups=col_groups,
        rows=rows,
    )


# =========================================================== best detection
def _best_per_column(spec: TableSpec) -> Dict[Tuple[str, str], Optional[str]]:
    """
    For each (group_label, col_name) cell, return the method whose value is
    best.  Ties are broken arbitrarily but consistently (first encountered).
    Cells with no value or no available rows return None.
    Grey rows (style='gray') are excluded from the comparison.
    """
    best = {}
    for g in spec.col_groups:
        for c in g.cols:
            key = (g.label, c.name)
            best_method = None
            best_val = -np.inf if g.higher_is_better else np.inf
            for r in spec.rows:
                if not r.available or r.style == "gray":
                    continue
                v = r.values.get(key)
                if v is None:
                    continue
                if g.higher_is_better and v > best_val:
                    best_val, best_method = v, r.method
                elif (not g.higher_is_better) and v < best_val:
                    best_val, best_method = v, r.method
            best[key] = best_method
    return best


def _fmt_value(v: Optional[float], col_label: str) -> str:
    if v is None:
        return "—"
    if "%" in col_label:
        return f"{v:.1f}"
    if col_label == "Sat-score":
        return f"{v:.3f}"
    if col_label == "Runtime (s)":
        return f"{v:.3f}" if v < 10 else f"{v:.1f}"
    if col_label == "Cost":
        return f"{v:.2f}"
    return f"{v:.3f}"


def _condense_reason(raw: str, max_len: int = 110) -> str:
    """Short, human-friendly summary of an unavailability message."""
    s = (raw or "").splitlines()[0].strip()
    # drop verbose error prefixes
    s = s.replace("Import failed:", "→")
    s = s.replace("ModuleNotFoundError: ", "")
    s = s.replace("RuntimeError: ", "")
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"
    return s


# ============================================================ matplotlib PDF
def render_paper_table_pdf(spec: TableSpec, out_path: Path) -> None:
    """Render the table as a PDF using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    out_path = Path(out_path)
    best = _best_per_column(spec)

    # ----- column layout
    # leftmost: family group label
    # then: method, hyperparams, trained?, generalist?
    # then: data columns (each col-group's columns side-by-side)
    meta_col_widths = [1.6, 2.2, 2.2, 0.7, 0.7]  # family, method, hp, T?, G?
    data_col_width = 1.05
    n_data = sum(len(g.cols) for g in spec.col_groups)
    col_widths = meta_col_widths + [data_col_width] * n_data
    n_cols = len(col_widths)

    # x positions = cumulative sum
    xs = [0.0]
    for w in col_widths:
        xs.append(xs[-1] + w)
    total_w = xs[-1]

    # ----- row layout
    # header rows: col-group label / col name / (citations)
    # body rows: one per spec.row, with extra spacing between families
    row_h = 0.32
    header_h_group = 0.40     # the "Video Classification..."-style line
    header_h_name = 0.70      # vertical column names
    has_any_cite = any(c.cite for g in spec.col_groups for c in g.cols)
    header_h_cite = 0.35 if has_any_cite else 0.0

    # find family transitions
    family_separators = []
    cur_family = None
    body_rows = []
    cur_y = 0.0  # we'll fill from bottom up below
    for i, r in enumerate(spec.rows):
        if r.family != cur_family:
            family_separators.append(i)
            cur_family = r.family
    n_body = len(spec.rows)
    body_h = n_body * row_h + len(family_separators) * 0.05

    title_h = 0.6
    subtitle_h = 0.30
    fig_h = title_h + subtitle_h + header_h_group + header_h_name + header_h_cite \
            + body_h + 0.4
    fig_w = total_w + 0.5  # padding

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("auto")
    ax.set_axis_off()
    plt.rcParams["font.family"] = "serif"

    # title at top
    y_cursor = fig_h - 0.2
    ax.text(total_w / 2, y_cursor, spec.title, ha="center", va="top",
            fontsize=12, fontweight="bold")
    y_cursor -= title_h
    ax.text(total_w / 2, y_cursor, spec.subtitle, ha="center", va="top",
            fontsize=8, style="italic", color="#555")
    y_cursor -= subtitle_h

    # top rule
    top_rule_y = y_cursor
    ax.plot([0, total_w], [top_rule_y, top_rule_y], color="black", lw=1.2)
    y_cursor -= 0.05

    # ---- column-group span line
    y_group_label = y_cursor - header_h_group / 2
    # write group labels and span lines for the data columns only
    data_col_x_start = xs[len(meta_col_widths)]
    cur_x = data_col_x_start
    for g in spec.col_groups:
        n_g = len(g.cols)
        x_left = cur_x
        x_right = cur_x + n_g * data_col_width
        x_mid = (x_left + x_right) / 2
        ax.text(x_mid, y_group_label, g.label, ha="center", va="center",
                fontsize=9, fontweight="bold")
        # short rule under the group label
        rule_y = y_group_label - header_h_group / 2 + 0.04
        ax.plot([x_left + 0.1, x_right - 0.1],
                [rule_y, rule_y],
                color="black", lw=0.6)
        cur_x = x_right
    y_cursor -= header_h_group

    # mid rule below group labels (only spanning meta+data columns)
    mid_rule_y = y_cursor
    ax.plot([0, total_w], [mid_rule_y, mid_rule_y], color="black", lw=0.5)
    y_cursor -= 0.03

    # ---- vertical column-name row
    y_name_top = y_cursor
    y_name_bot = y_cursor - header_h_name
    y_name_center = (y_name_top + y_name_bot) / 2

    meta_col_names = ["", "Method", "Hyperparams", "Trained", "Generalist"]
    for ci, name in enumerate(meta_col_names):
        x_mid = (xs[ci] + xs[ci + 1]) / 2
        ax.text(x_mid, y_name_center, name, ha="center", va="center",
                fontsize=9, fontweight="bold")

    ci = len(meta_col_widths)
    for g in spec.col_groups:
        for c in g.cols:
            x_mid = (xs[ci] + xs[ci + 1]) / 2
            txt = c.name
            ax.text(x_mid, y_name_top - 0.05, txt, ha="center", va="top",
                    fontsize=8, fontweight="bold", rotation=90)
            ci += 1
    y_cursor -= header_h_name

    # ---- citations / subtext row (only for data cols)
    if has_any_cite:
        y_cite_top = y_cursor
        y_cite_center = y_cursor - header_h_cite / 2
        ci = len(meta_col_widths)
        for g in spec.col_groups:
            for c in g.cols:
                x_mid = (xs[ci] + xs[ci + 1]) / 2
                if c.cite:
                    ax.text(x_mid, y_cite_center, f"[{c.cite}]",
                            ha="center", va="center",
                            fontsize=6, color="#777")
                ci += 1
        y_cursor -= header_h_cite

    # mid rule below names
    mid2_y = y_cursor
    ax.plot([0, total_w], [mid2_y, mid2_y], color="black", lw=0.5)
    y_cursor -= 0.05

    # ---- body rows
    family_first_row_idx = {}
    family_last_row_idx = {}
    cur_family = None
    for i, r in enumerate(spec.rows):
        if r.family != cur_family:
            family_first_row_idx[r.family] = i
            cur_family = r.family
        family_last_row_idx[r.family] = i

    # iterate top-to-bottom
    cur_family = None
    for i, r in enumerate(spec.rows):
        # tiny separator gap when family changes
        if r.family != cur_family and i > 0:
            y_cursor -= 0.05
            cur_family = r.family
        elif cur_family is None:
            cur_family = r.family

        y_top = y_cursor
        y_bot = y_cursor - row_h
        y_mid = (y_top + y_bot) / 2

        # row tint
        if r.style == "ours":
            ax.add_patch(Rectangle((0, y_bot), total_w, row_h,
                                   facecolor="#DCE8F4", edgecolor="none",
                                   zorder=-2))
        elif r.style == "gray":
            ax.add_patch(Rectangle((0, y_bot), total_w, row_h,
                                   facecolor="#F2F2F2", edgecolor="none",
                                   zorder=-2))

        # family label: only in first row of family, and visually centred
        if i == family_first_row_idx[r.family]:
            first = family_first_row_idx[r.family]
            last = family_last_row_idx[r.family]
            n_fam_rows = last - first + 1
            # y_top is the top edge of the first row;
            # centre = y_top - n_fam_rows * row_h / 2
            y_fam_label = y_top - (n_fam_rows * row_h) / 2
            ax.text(xs[0] + meta_col_widths[0] / 2, y_fam_label, r.family,
                    ha="center", va="center", fontsize=9,
                    fontweight="bold", style="italic")

        # method name (italic if gray)
        gray_style = (r.style == "gray")
        method_label = r.method
        ax.text(xs[1] + 0.1, y_mid, method_label,
                ha="left", va="center", fontsize=8.5,
                color="#777" if gray_style else "black",
                style="italic" if gray_style else "normal",
                fontweight=("bold" if r.style == "ours" else "normal"))

        # hyperparams
        ax.text(xs[2] + 0.1, y_mid, r.hyperparams or "—",
                ha="left", va="center", fontsize=7.5,
                color="#777" if gray_style else "#333")

        # trained / generalist checkmarks
        for ci_meta, val in zip([3, 4], [r.trained, r.generalist]):
            x_mid = (xs[ci_meta] + xs[ci_meta + 1]) / 2
            if val is True:
                ax.text(x_mid, y_mid, r"$\checkmark$", ha="center", va="center",
                        fontsize=11, color="#1E5BBA")
            elif val is False:
                ax.text(x_mid, y_mid, r"$\times$", ha="center", va="center",
                        fontsize=11, color="#A0A0A0")
            else:
                ax.text(x_mid, y_mid, "—", ha="center", va="center",
                        fontsize=9, color="#999")

        # data cells
        ci = len(meta_col_widths)
        if r.unavailable_reason:
            # span the entire data-column area with the reason text
            data_x_left = xs[ci]
            data_x_right = xs[-1]
            data_x_mid = (data_x_left + data_x_right) / 2
            ax.text(data_x_mid, y_mid,
                    f"({r.unavailable_reason})",
                    ha="center", va="center",
                    fontsize=7.0, color="#888",
                    style="italic")
        else:
            for g in spec.col_groups:
                for c in g.cols:
                    x_mid = (xs[ci] + xs[ci + 1]) / 2
                    v = r.values.get((g.label, c.name))
                    txt = _fmt_value(v, g.label) if r.available else "—"
                    is_best = (best.get((g.label, c.name)) == r.method
                               and r.available and r.style != "gray")
                    color = "#777" if gray_style or not r.available else "black"
                    weight = "bold" if is_best else "normal"
                    t = ax.text(x_mid, y_mid, txt, ha="center", va="center",
                                fontsize=8.5, color=color, fontweight=weight,
                                style=("italic" if gray_style else "normal"))
                    if is_best:
                        # underline by drawing a short line beneath the text
                        # (matplotlib without LaTeX has no native \underline)
                        # we approximate with a manual line
                        txt_w_approx = 0.45    # cell-relative half-width estimate
                        ax.plot([x_mid - txt_w_approx, x_mid + txt_w_approx],
                                [y_mid - 0.13, y_mid - 0.13],
                                color="black", lw=0.7)
                    ci += 1

        y_cursor -= row_h

    # bottom rule
    bot_rule_y = y_cursor
    ax.plot([0, total_w], [bot_rule_y, bot_rule_y], color="black", lw=1.2)

    # footer note
    n_unavail = sum(1 for r in spec.rows if not r.available)
    if n_unavail > 0:
        y_cursor -= 0.35
        ax.text(
            0.05, y_cursor,
            (f"Note: {n_unavail} row(s) shown in italic grey could not be run "
             "in this benchmark — each row's reason is given inline. "
             "Trained-model baselines need either pretrained checkpoints "
             "or local training (see benchmark/README.md)."),
            ha="left", va="top", fontsize=7, color="#666", style="italic",
            wrap=True,
        )

    fig.subplots_adjust(left=0.03, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


# ============================================================ LaTeX source
def render_paper_table_latex(spec: TableSpec, out_path: Path) -> None:
    """Write a LaTeX booktabs source file for the same table."""
    out_path = Path(out_path)
    best = _best_per_column(spec)

    n_data = sum(len(g.cols) for g in spec.col_groups)
    n_meta = 5  # family, method, hp, trained, generalist
    n_total = n_meta + n_data

    lines = [
        "% Auto-generated by cnlm_langevin/benchmark/paper_table.py",
        "% Required preamble:",
        "%   \\usepackage{booktabs,multirow,array,colortbl,xcolor,graphicx}",
        "%   \\definecolor{ourblue}{HTML}{DCE8F4}",
        "%   \\definecolor{citegray}{HTML}{777777}",
        "",
        "\\begin{table*}[t]\\centering\\small",
        f"  \\caption{{{spec.title}.  {spec.subtitle}}}",
        "  \\label{tab:cnlm_benchmark}",
        "  \\setlength{\\tabcolsep}{4pt}",
        f"  \\begin{{tabular}}{{l l l c c {' '.join(['r'] * n_data)}}}",
        "    \\toprule",
    ]

    # Column-group header row (multicolumn)
    cmid_lines = []
    parts = ["", "", "", "", ""]   # blanks for the 5 meta cols
    col_idx = n_meta + 1
    for g in spec.col_groups:
        ng = len(g.cols)
        parts.append(f"\\multicolumn{{{ng}}}{{c}}{{\\textbf{{{g.label}}}}}")
        cmid_lines.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + ng - 1}}}")
        col_idx += ng
    lines.append("    " + " & ".join(parts) + " \\\\")
    lines.append("    " + " ".join(cmid_lines))

    # Column name row
    parts = ["", "\\textbf{Method}", "\\textbf{Hyperparams}",
             "\\textbf{Trained}", "\\textbf{Generalist}"]
    for g in spec.col_groups:
        for c in g.cols:
            parts.append(f"\\rotatebox{{90}}{{\\textbf{{{c.name}}}}}")
    lines.append("    " + " & ".join(parts) + " \\\\")

    # Citation sub-row (only data columns)
    parts = ["", "", "", "", ""]
    for g in spec.col_groups:
        for c in g.cols:
            cite = c.cite if c.cite else ""
            parts.append(f"{{\\scriptsize\\color{{citegray}} {cite}}}" if cite else "")
    lines.append("    " + " & ".join(parts) + " \\\\")
    lines.append("    \\midrule")

    # Body rows, grouped by family
    cur_family = None
    rows_in_family = []
    family_blocks = []   # list of (family, [row_indices])
    for i, r in enumerate(spec.rows):
        if r.family != cur_family:
            if rows_in_family:
                family_blocks.append((cur_family, rows_in_family))
            rows_in_family = []
            cur_family = r.family
        rows_in_family.append(i)
    if rows_in_family:
        family_blocks.append((cur_family, rows_in_family))

    def _fmt_check(v):
        if v is True:
            return "\\textcolor{blue}{$\\checkmark$}"
        if v is False:
            return "\\textcolor{gray}{$\\times$}"
        return "—"

    for blk_idx, (family, idxs) in enumerate(family_blocks):
        if blk_idx > 0:
            lines.append("    \\midrule")
        nrows = len(idxs)
        for j, i in enumerate(idxs):
            r = spec.rows[i]
            family_cell = ""
            if j == 0:
                if nrows > 1:
                    family_cell = (
                        f"\\multirow{{{nrows}}}{{*}}{{\\textit{{{family}}}}}"
                    )
                else:
                    family_cell = f"\\textit{{{family}}}"
            method_cell = r.method.replace("_", "\\_")
            if r.style == "ours":
                method_cell = f"\\textbf{{{method_cell}}}"
            elif r.style == "gray":
                method_cell = f"\\textit{{\\color{{gray}}{method_cell}}}"
            hp_cell = r.hyperparams.replace("_", "\\_") if r.hyperparams else "—"
            t_cell = _fmt_check(r.trained)
            g_cell = _fmt_check(r.generalist)

            if r.unavailable_reason:
                # span the data area with a single multicolumn cell
                reason = r.unavailable_reason.replace("_", "\\_")\
                                              .replace("&", "\\&")\
                                              .replace("%", "\\%")\
                                              .replace("#", "\\#")
                cells = [family_cell, method_cell, hp_cell, t_cell, g_cell]
                row_str = ("    " + " & ".join(cells)
                           + f" & \\multicolumn{{{n_data}}}{{c}}{{"
                           + f"\\textit{{\\color{{gray}}({reason})}}}}"
                           + " \\\\")
            else:
                data_cells = []
                for g in spec.col_groups:
                    for c in g.cols:
                        v = r.values.get((g.label, c.name))
                        txt = _fmt_value(v, g.label) if r.available else "—"
                        is_best = (best.get((g.label, c.name)) == r.method
                                   and r.available and r.style != "gray")
                        if is_best:
                            txt = f"\\underline{{\\textbf{{{txt}}}}}"
                        if r.style == "gray":
                            txt = f"\\textit{{\\color{{gray}} {txt}}}"
                        data_cells.append(txt)
                cells = [family_cell, method_cell, hp_cell, t_cell, g_cell] + data_cells
                row_str = "    " + " & ".join(cells) + " \\\\"
            if r.style == "ours":
                row_str = row_str.replace("    " + " & ".join(
                    [family_cell, method_cell, hp_cell, t_cell, g_cell]
                ), "    \\rowcolor{ourblue} " + " & ".join(
                    [family_cell, method_cell, hp_cell, t_cell, g_cell]
                ))
            lines.append(row_str)

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table*}")
    out_path.write_text("\n".join(lines) + "\n")


# ================================================================ entry-point
def write_paper_table(
    raw_rows: List[dict],
    problem_type: str,
    out_dir: Path,
    instance_groups: Optional[Dict[str, List[str]]] = None,
    hyperparam_strings: Optional[Dict[str, str]] = None,
) -> Tuple[Path, Path]:
    """Build the spec, write both the rendered PDF and the LaTeX source.

    Returns the (pdf_path, tex_path) tuple.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = build_paper_table_spec(
        raw_rows, problem_type,
        instance_groups=instance_groups,
        hyperparam_strings=hyperparam_strings,
    )
    pdf_path = out_dir / "paper_table.pdf"
    tex_path = out_dir / "paper_table.tex"
    render_paper_table_pdf(spec, pdf_path)
    render_paper_table_latex(spec, tex_path)
    return pdf_path, tex_path


__all__ = [
    "TableSpec", "TableRow", "Column", "ColumnGroup",
    "build_paper_table_spec",
    "render_paper_table_pdf",
    "render_paper_table_latex",
    "write_paper_table",
]
