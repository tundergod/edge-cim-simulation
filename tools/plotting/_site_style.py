"""Shared style + helpers for the Phase-1 site's editorial figures (tools/plotting/site_*.py).

7 of the 8 site_*.py (cpu/gpu/m1/m2/m5/m7/npu) redefine a byte-identical mpl.rcParams
block (font.size 9, Helvetica Neue editorial sizing), the load()/_grid() helpers, and a
subset of the HERO/WARM/OK/GREY/FAINT/INK/SOFT/PAPER/GRID palette. This module factors
those OUT. site_m8.py's rcParams (font.size 11) and _grid (no `which="major"`) genuinely
differ from the rest, so it keeps its own local copies of those two.

Distinct from _style.py: that module runs its OWN separate mpl.rcParams.update
(font.size 7, Arial) for the Phase 0.2/0.3 figures and must not be touched or merged
with this one.
"""
import json
from pathlib import Path

import matplotlib as mpl

# editorial palette shared across the Phase-1 site figures (union of each file's subset;
# FAINT is site_m1's name for the same grey as GREY elsewhere, #b9b09c)
HERO = "#0072B2"; WARM = "#C45A12"; OK = "#1b7f5a"; GREY = "#b9b09c"; FAINT = "#b9b09c"
INK = "#17150f"; SOFT = "#5b554a"; PAPER = "#fbf6ec"; GRID = "#e8e1d2"


def apply_site_style():
    """The rcParams block shared verbatim by site_cpu/gpu/m1/m2/m5/m7/npu (site_m8 differs)."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 0.9, "axes.edgecolor": "#888",
        "xtick.color": "#555", "ytick.color": "#555",
        "axes.labelcolor": INK, "text.color": INK,
        "legend.frameon": False, "figure.dpi": 150,
    })


def load(p):
    return json.loads(Path(p).read_text())


def _grid(ax):
    """The _grid shared verbatim by site_cpu/gpu/m2/m5/m7/npu (site_m1/site_m8 differ, stay local)."""
    ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
