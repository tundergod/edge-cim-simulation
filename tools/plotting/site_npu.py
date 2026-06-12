"""M4-NPU (RKNPU2) editorial figures for the Phase-1 site (build artifact, nature-figure style).

THE MOST HONESTY-SENSITIVE UNIT: there is NO RKNPU2 silicon (board offline, issue #13). Every
number here is analytic-baseline or simulated -- NONE is calibrated, NONE is a ground truth. These
figures frame the NPU as an UNCERTAINTY SPREAD, not a validation: three sims (analytic roofline,
ONNXim heavy-sim, ScaleSim-pending) with no silicon judge between them.
Regenerable from committed JSON only. Writes PNG (+PDF/SVG) to docs/figures/phase1-site/.

  npu_systolic — §1/§2: analytic systolic-roofline staircase (32x32 align quantum). ANALYTIC
                 BASELINE (no silicon) -- shape borrowed from HeteroInfer Fig3, height from 6 TOPS.
  npu_spread   — §3: analytic vs ONNXim per-shape as an UNCERTAINTY BAND (~318% median /493% max
                 delta), plus an empty dashed "ScaleSim (pending)" slot. No silicon ground truth.

Run: ./.venv/bin/python tools/plotting/site_npu.py
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
from simulator.models.engine import Workload  # noqa: E402
from simulator.models.m4_npu import NpuModel  # noqa: E402

TREND = ROOT / "validation/reports/phase1.2/m4_npu.json"
ONNXIM = ROOT / "validation/reports/phase1.3/m4_npu_onnxim.json"
SPEC = ROOT / "simulator/specs/npu_rknpu2.json"
OUT = ROOT / "docs/figures/phase1-site"

# editorial palette (same as site_m1/site_m2). NO "OK"/green here on purpose: nothing is validated.
HERO = "#0072B2"; WARM = "#C45A12"; GREY = "#b9b09c"
INK = "#17150f"; SOFT = "#5b554a"; PAPER = "#fbf6ec"; GRID = "#e8e1d2"
SCALE = "#9a93a8"  # muted violet-grey for the pending ScaleSim slot (clearly "not here yet")

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


def fig_systolic(trend, spec):
    """§1/§2 -- analytic systolic-roofline staircase: latency vs N (compute-bound, M=512, K=2048),
    stepping every time N crosses a multiple of the BORROWED 32x32 quantum. ANALYTIC BASELINE,
    no silicon: the SHAPE is borrowed from HeteroInfer Fig3, the step HEIGHT comes from 6 TOPS."""
    npu = NpuModel(spec, engine="analytic")
    cond = trend["trend_conditions"]["a_staircase_vs_fig3"]
    sd = cond["borrowed_systolic_dim"]
    knees = cond["knee_positions"]
    M, K = 512, 2048
    Ns = np.arange(1, 257)
    lat = np.array([npu.predict(Workload(op="gemm", M=M, K=K, N=int(n), dtype="int8"))["latency_us"]
                    for n in Ns])
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    _grid(ax)
    ax.step(Ns, lat, where="post", color=HERO, lw=1.6, zorder=3)
    for kn in knees:
        ax.axvline(kn, ls=":", color=GREY, lw=0.8, zorder=1)
    ax.scatter(knees, [npu.predict(Workload(op="gemm", M=M, K=K, N=int(n), dtype="int8"))["latency_us"]
                       for n in knees], s=30, color=WARM, edgecolors="white", linewidths=0.5,
               zorder=5, label=f"{len(knees)} knees @ multiples of {sd}")
    ax.set_xlim(0, 260); ax.set_ylim(bottom=0)
    ax.set_xlabel("output channels  N   (M=512, K=2048, INT8)")
    ax.set_ylabel("analytic latency  (us)")
    ax.text(0.015, 0.97, "ANALYTIC BASELINE -- no silicon (#13)", transform=ax.transAxes,
            fontsize=8.5, va="top", color=WARM, fontweight="bold")
    ax.annotate(f"staircase: pad N up to {sd}x{sd}\n(shape borrowed HeteroInfer Fig3;\nheight from 6 TOPS, not measured)",
                xy=(0.04, 0.62), xycoords="axes fraction", fontsize=7.8, color=SOFT, va="top")
    ax.legend(loc="lower right", fontsize=8)
    S.save(fig, OUT / "npu_systolic")


def fig_spread(onx):
    """§3 -- analytic vs ONNXim per-shape as an UNCERTAINTY BAND (NOT an agreement plot): ONNXim
    runs systematically ~4x above the analytic roofline (median ~318%, max ~493%). A third engine,
    ScaleSim, is PENDING (empty dashed slot). No silicon judges any of the three."""
    rows = sorted(onx["per_shape"], key=lambda r: r["analytic_us"])
    an = np.array([r["analytic_us"] for r in rows])
    on = np.array([r["onnxim_us"] for r in rows])
    x = np.arange(len(rows))
    med = onx["median_abs_delta_pct"]; mx = onx["max_abs_delta_pct"]
    i_max = int(np.argmax([r["delta_pct"] for r in rows]))

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    _grid(ax)
    # the spread itself: a vertical band from analytic (lower) to ONNXim (upper) at each shape.
    ax.vlines(x, an, on, color=GREY, lw=1.4, zorder=2)
    ax.scatter(x, an, s=34, color=HERO, edgecolors="white", linewidths=0.5, zorder=4,
               label="analytic roofline (lower)")
    ax.scatter(x, on, s=34, color=WARM, marker="s", edgecolors="white", linewidths=0.5, zorder=4,
               label="ONNXim heavy-sim (upper)")
    # ScaleSim: a clearly-empty, dashed pending slot -- no data, deliberately blank.
    ax.scatter([], [], marker="D", facecolors="none", edgecolors=SCALE, linewidths=1.1,
               label="ScaleSim (pending -- to be built)")
    ax.plot([x[0], x[-1]], [an.max() * 1.55] * 2, ls="--", color=SCALE, lw=1.1, zorder=1)
    ax.text(x[-1], an.max() * 1.62, "ScaleSim slot (pending)", color=SCALE, fontsize=7.6,
            ha="right", va="bottom", style="italic")
    # call out the spread magnitude (uncertainty, not error).
    ax.annotate(f"max spread +{mx:.0f}%", xy=(x[i_max], on[i_max]),
                xytext=(x[i_max] - 3.4, on[i_max] * 1.9), fontsize=8, color=WARM, fontweight="bold",
                ha="center", arrowprops=dict(arrowstyle="->", color=WARM, lw=0.9))
    ax.set_yscale("log")
    ax.set_xlim(-0.7, len(rows) - 0.3)
    ax.set_xticks([]); ax.set_xlabel("GEMM shapes  (sorted by analytic latency)")
    ax.set_ylabel("latency  (us, log)")
    ax.text(0.015, 0.97,
            f"sim-vs-sim SPREAD, no silicon ground truth (#13)\nmedian spread {med:.0f}% -- neither is a judge",
            transform=ax.transAxes, fontsize=8, va="top", color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc=PAPER, ec=GRID, lw=0.8))
    ax.legend(loc="lower right", fontsize=7.6)
    S.save(fig, OUT / "npu_spread")


def main():
    fig_systolic(load(TREND), load(SPEC))
    fig_spread(load(ONNXIM))
    figs = sorted(p.name for p in OUT.glob("npu_*.png"))
    print(f"wrote {len(figs)} NPU site figures: {figs}")


if __name__ == "__main__":
    main()
