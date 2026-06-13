"""M4-CPU (ARM A76 support ops) editorial figures for the Phase-1 site (nature-figure style).

Clean, single-purpose, white-background figures sized for the report page. M4-CPU's basic
function: predict the latency of the SIX non-GEMM support ops (rmsnorm, rope_apply, residual,
swiglu, softmax, sampling_argmax) via an INSTRUCTION-COUNT roofline
    latency = max(compute, memory) + overhead_op
where exp() (transcendental ~30 fp ops) in softmax/swiglu is the cost driver.
Regenerable from committed JSON only. Writes PNG (+PDF/SVG) to docs/figures/phase1-site/.

  cpu_roofline   — §1/§2: 6 ops measured median vs instruction-count roofline prediction
                   (parity, +-10/20% bands), overall median annotated.
  cpu_ops_cost   — §1: per-op ops_per_elem with exp()-driven ops (softmax/swiglu) highlighted.

Run: ./.venv/bin/python tools/plotting/site_cpu.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/plotting"))
import _style as S  # noqa: E402
from simulator.models.m4_cpu import CpuModel  # noqa: E402

SPEC = ROOT / "simulator/specs/cpu_rk3588.json"
CPUOPS = ROOT / "measurements/aetina/cpu_ops.json"
INSTR = ROOT / "simulator/models/params/m4_cpu_instrcount.json"
REPORT = ROOT / "validation/reports/phase1.2/m4_cpu.json"
OUT = ROOT / "docs/figures/phase1-site"

HERO = "#0072B2"; WARM = "#C45A12"; OK = "#1b7f5a"; GREY = "#b9b09c"
INK = "#17150f"; SOFT = "#5b554a"; PAPER = "#fbf6ec"; GRID = "#e8e1d2"

# ops grouped: exp()-driven (cost driver, warm) vs the rest (hero blue)
EXP_OPS = {"softmax", "swiglu"}
OP_LABEL = {"residual": "residual", "rmsnorm": "rmsnorm", "rope_apply": "rope_apply",
            "swiglu": "swiglu", "softmax": "softmax", "sampling_argmax": "sampling\nargmax"}

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
    ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def _meas_pred():
    """Every fp32 cpu_ops measurement -> (op_base, measured_us, predicted_us) via the calibrated
    instruction-count roofline. softmax_kv{N} parses the kv sweep; others are per-model points."""
    spec = load(SPEC)
    m = CpuModel(spec)
    ops = load(CPUOPS)["ops"]
    rows = []
    for r in ops.values():
        if r["dtype"] != "fp32":
            continue
        op = r["op"]
        base = "softmax" if op.startswith("softmax") else op
        kv = int(op[len("softmax_kv"):]) if op.startswith("softmax_kv") else None
        pred = m.op_us(base, r["model"], dtype="fp32", kv=kv)
        rows.append((base, r["median_us"], pred))
    return rows


def fig_roofline():
    """§1/§2 — measured vs instruction-count roofline prediction (parity), exp-ops highlighted."""
    rows = _meas_pred()
    rep = load(REPORT)["overall_residual_pct"]
    meas = np.array([r[1] for r in rows])
    pred = np.array([r[2] for r in rows])
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    _grid(ax)
    lim = [0, max(meas.max(), pred.max()) * 1.08]
    x = np.array(lim)
    ax.fill_between(x, x * 0.8, x * 1.2, color=HERO, alpha=0.07, zorder=1, label="+-20%")
    ax.fill_between(x, x * 0.9, x * 1.1, color=HERO, alpha=0.13, zorder=1, label="+-10%")
    ax.plot(lim, lim, "-", color="#aaa", lw=1.0, zorder=2)
    for base in sorted({r[0] for r in rows}):
        sel = [r for r in rows if r[0] == base]
        c = WARM if base in EXP_OPS else HERO
        mk = "s" if base in EXP_OPS else "o"
        ax.scatter([r[1] for r in sel], [r[2] for r in sel], s=42, color=c, marker=mk,
                   edgecolors="white", linewidths=0.6, zorder=4)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("measured median  (us, fp32, 1 A76 core)")
    ax.set_ylabel("roofline prediction  (us)")
    ax.text(0.04, 0.96,
            f"overall median |err| {rep['median']:.2f}%\np95 {rep['p95']:.2f}%\nn = {len(rows)}",
            transform=ax.transAxes, fontsize=8.5, va="top", color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc=PAPER, ec=GRID, lw=0.8))
    # direct legend for the two op classes
    ax.scatter([], [], s=42, color=WARM, marker="s", edgecolors="white", linewidths=0.6,
               label="exp()-driven (softmax/swiglu)")
    ax.scatter([], [], s=42, color=HERO, marker="o", edgecolors="white", linewidths=0.6,
               label="other 4 ops")
    ax.legend(loc="lower right", fontsize=7.8)
    S.save(fig, OUT / "cpu_roofline")


def fig_ops_cost():
    """§1 — per-op ops_per_elem: the structural cost driver. exp()-ops (softmax/swiglu) dwarf the rest."""
    p = load(INSTR)
    ope = p["ops_per_elem"]
    order = ["residual", "sampling_argmax", "rmsnorm", "rope_apply", "softmax", "swiglu"]
    vals = [ope[o] for o in order]
    cols = [WARM if o in EXP_OPS else HERO for o in order]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    _grid(ax)
    y = np.arange(len(order))
    ax.barh(y, vals, color=cols, height=0.62, zorder=3, edgecolor="white", linewidth=0.6)
    for yi, v, o in zip(y, vals, order):
        ax.text(v + 0.6, yi, str(v), va="center", fontsize=8.5,
                color=WARM if o in EXP_OPS else SOFT, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([OP_LABEL[o].replace("\n", " ") for o in order])
    ax.set_xlabel("ops per element  (equivalent fp ops)")
    ax.set_xlim(0, max(vals) * 1.18)
    ax.text(0.97, 0.06,
            "exp() ~= 30 fp ops:\nsoftmax/swiglu are the\ntranscendental cost driver",
            transform=ax.transAxes, fontsize=8, ha="right", va="bottom", color=WARM)
    S.save(fig, OUT / "cpu_ops_cost")


def main():
    fig_roofline()
    fig_ops_cost()
    figs = sorted(p.name for p in OUT.glob("cpu_*.png"))
    print(f"wrote {len(figs)} CPU site figures: {figs}")


if __name__ == "__main__":
    main()
