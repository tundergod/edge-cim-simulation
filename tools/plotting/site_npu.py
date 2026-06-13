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
SCALESIM = ROOT / "validation/reports/phase1.6/npu_scalesim.json"
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


def fig_spread(nps):
    """§3 -- THREE sims per-shape as an UNCERTAINTY BAND (NOT an agreement plot): analytic / ONNXim /
    SCALE-Sim diverge by a median ~7x with NO silicon judge between them. ScaleSim is a third
    uncertainty point, not an adjudicator -- divergence is the cost of #13, not a defect."""
    rows = sorted(nps["three_way_subset"], key=lambda r: r["lat_us"]["analytic"])
    an = np.array([r["lat_us"]["analytic"] for r in rows])
    on = np.array([r["lat_us"]["onnxim"] for r in rows])
    sc = np.array([r["lat_us"]["scalesim"] for r in rows])
    x = np.arange(len(rows))
    med, mx = nps["spread_median_x"], nps["spread_max_x"]

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    _grid(ax)
    lo = np.minimum.reduce([an, on, sc]); hi = np.maximum.reduce([an, on, sc])
    ax.vlines(x, lo, hi, color=GREY, lw=1.4, zorder=2)   # the spread = full divergence band
    ax.scatter(x, an, s=34, color=HERO, edgecolors="white", linewidths=0.5, zorder=4, label="analytic roofline")
    ax.scatter(x, on, s=34, color=WARM, marker="s", edgecolors="white", linewidths=0.5, zorder=4, label="ONNXim heavy-sim")
    ax.scatter(x, sc, s=40, color=SCALE, marker="D", edgecolors="white", linewidths=0.5, zorder=4, label="SCALE-Sim (32x32-WS)")
    ax.set_yscale("log")
    ax.set_xlim(-0.7, len(rows) - 0.3)
    ax.set_xticks([]); ax.set_xlabel(f"GEMM shapes ({len(rows)} common; {len(nps['skipped_shapes'])} giant shapes ScaleSim-intractable)")
    ax.set_ylabel("latency  (us, log)")
    ax.text(0.015, 0.97,
            f"three-sim SPREAD, no silicon ground truth (#13)\nmedian {med:.1f}x / max {mx:.1f}x -- none is a judge",
            transform=ax.transAxes, fontsize=8, va="top", color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc=PAPER, ec=GRID, lw=0.8))
    ax.legend(loc="lower right", fontsize=7.6)
    S.save(fig, OUT / "npu_spread")


def fig_sensitivity(nps):
    """§(new) -- native systolic sensitivities the MODEL produces (emergent, not tuned; NOT measured
    RKNPU2). Hero: per-token decode GEMV runs at ~1% array utilisation, climbing as activations batch
    (M). Side: alignment / operand-order / shape best<->worst ratios. Design guideline, not silicon."""
    sens = nps["native_sensitivity"]
    sm = sens["shape_M"]["cases"]
    M = [c["MKN"][0] for c in sm]; util = [c["util_pct"] for c in sm]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.4), gridspec_kw={"width_ratios": [1.25, 1]})
    _grid(a1); _grid(a2)
    # left: utilisation climbs with batched activations M (decode GEMV ~1% -> prefill)
    bars = a1.bar(range(len(M)), util, width=0.6, color=[WARM, HERO, HERO])
    for i, (m, u) in enumerate(zip(M, util)):
        a1.text(i, u + 1.5, f"{u:.0f}%", ha="center", fontsize=9.5, fontweight="bold")
    a1.set_xticks(range(len(M))); a1.set_xticklabels([f"M={m}" for m in M])
    a1.set_ylabel("array utilisation  (%)"); a1.set_ylim(0, max(util) * 1.2)
    a1.text(0.03, 0.96, "decode GEMV (M=1)\n~1% of a 32x32 array", transform=a1.transAxes,
            fontsize=8, va="top", color=WARM)
    a1.set_xlabel("activation batch  M")
    # right: the three best<->worst sensitivity ratios
    labels = ["align\n(N % 32)", "order\n(M<->N)", "shape\n(M batch)"]
    ratios = [sens["alignment_N"]["worst_over_best"], sens["order_MN_swap"]["worst_over_best"],
              sens["shape_M"]["worst_over_best"]]
    a2.barh(range(3), ratios, color=HERO, height=0.6)
    for i, r in enumerate(ratios):
        a2.text(r + 0.03, i, f"{r:.2f}x", va="center", fontsize=9.5, fontweight="bold")
    a2.set_yticks(range(3)); a2.set_yticklabels(labels, fontsize=8.5)
    a2.set_xlabel("worst / best  (cycles)"); a2.set_xlim(1, max(ratios) * 1.25)
    a2.invert_yaxis()
    fig.suptitle("native systolic sensitivity -- model-emergent (not tuned), NOT measured RKNPU2",
                 fontsize=8.5, color=SOFT, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    S.save(fig, OUT / "npu_sensitivity")


def main():
    fig_systolic(load(TREND), load(SPEC))
    nps = load(SCALESIM)
    fig_spread(nps)
    fig_sensitivity(nps)
    figs = sorted(p.name for p in OUT.glob("npu_*.png"))
    print(f"wrote {len(figs)} NPU site figures: {figs}")


if __name__ == "__main__":
    main()
