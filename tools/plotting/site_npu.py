"""M4-NPU (RKNPU2) editorial figures for the Phase-1 site (build artifact, nature-figure style).

THE MOST HONESTY-SENSITIVE UNIT: there is NO RKNPU2 silicon (board offline, issue #13). Every
number here is analytic-baseline or simulated -- NONE is calibrated, NONE is a ground truth. These
figures frame the NPU as an UNCERTAINTY SPREAD, not a validation: three sims (analytic roofline,
ONNXim heavy-sim, ScaleSim-pending) with no silicon judge between them.
Regenerable from committed JSON only. Writes PNG (+PDF/SVG) to docs/figures/phase1-site/.

  npu_systolic — §1: MEASURED per-sim answer to "does our NPU model show a 32-systolic staircase?"
                 (Phase 1.6b, no presupposed knee). ScaleSim steps at 32 BY CONSTRUCTION (positive
                 control); ONNXim does NOT (smooth at decode, non-monotone at prefill). Separate axes.
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

TREND = ROOT / "validation/reports/phase1.2/m4_npu.json"
ONNXIM = ROOT / "validation/reports/phase1.3/m4_npu_onnxim.json"
SCALESIM = ROOT / "validation/reports/phase1.6/npu_scalesim.json"
SPEC = ROOT / "simulator/specs/npu_rknpu2.json"
# Phase 1.6b characteristic measurement (each sim swept independently; no presupposed knee):
CHAR_ONX = ROOT / "simulated/onnxim/rknpu2_characteristic.json"
CHAR_SCL = ROOT / "simulated/scalesim/rknpu2_characteristic.json"
CHAR_CMP = ROOT / "validation/reports/phase1.6/npu_characteristic_compare.json"
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


def fig_systolic(onx, scl, cmp):
    """§1 -- DO our two NPU sims actually show a 32-systolic staircase? MEASURED independently
    (Phase 1.6b), no presupposed knee. Each sim swept over the output dim (K=2048 fixed) and drawn
    on ITS OWN y-axis -- cross-sim magnitudes are NOT comparable (#13). ScaleSim (a literal 32x32
    array sim) steps cleanly at 32 BY CONSTRUCTION = positive control. ONNXim (adds NoC/DRAM
    scheduling) is the informative one: smooth ∝N at decode, non-monotone at prefill -- it does NOT
    reproduce a 32-staircase. model, NOT measured RKNPU2."""
    K = 2048

    def series(rows, M, key):
        pts = sorted((r["shape"][2], r[key]) for r in rows
                     if r["shape"][0] == M and r["shape"][1] == K)
        return np.array([n for n, _ in pts]), np.array([c for _, c in pts], float)

    # SEPARATE panels (NOT twin-axis overlay) so the two sims can never visually "track" -- the
    # whole point is staircase(ScaleSim) vs smooth/non-monotone(ONNXim), not agreement.
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    for row, (M, tag) in enumerate([(1, "decode  M=1"), (128, "prefill  M=128")]):
        cs_v = cmp["scalesim_positive_control"][str(M)]; co_v = cmp["onnxim"][str(M)]
        ns, cs = series(scl["rows"], M, "cycles_1core")
        no, co = series(onx["rows"], M, "cycles")
        aS, aO = axs[row, 0], axs[row, 1]
        for ax in (aS, aO):
            _grid(ax)
            for k in range(160, 513, 32):
                ax.axvline(k, ls=":", color=GREY, lw=0.6, zorder=1)
        aS.step(ns, cs, where="post", color=SCALE, lw=1.7, zorder=3)
        aO.plot(no, co, "-o", color=WARM, lw=1.2, ms=2.4, zorder=4)
        aS.set_ylabel(f"{tag}\nScaleSim cycles", color=SCALE, fontsize=8.2)
        aS.tick_params(axis="y", colors=SCALE); aO.tick_params(axis="y", colors=WARM)
        aO.set_ylabel("ONNXim cycles", color=WARM, fontsize=8.2)
        aS.text(0.04, 0.95, f"clean 32-staircase (cv={cs_v['bulk_delta_cv']})\nBY CONSTRUCTION",
                transform=aS.transAxes, fontsize=7.2, va="top", color=SCALE)
        msg = (f"smooth ~N  (R²={co_v['smooth_loglog_R2']})\nsteps only @192/256/384, NOT @160"
               if co_v["monotone"] else
               f"NON-monotone (cv={co_v['bulk_delta_cv']})\nR²={co_v['smooth_loglog_R2']} -- not a staircase")
        aO.text(0.04, 0.95, msg, transform=aO.transAxes, fontsize=7.2, va="top", color=WARM)
    axs[0, 0].set_title("ScaleSim  (positive control)", fontsize=8.6, color=SCALE, fontweight="bold")
    axs[0, 1].set_title("ONNXim  (informative)", fontsize=8.6, color=WARM, fontweight="bold")
    for c in (0, 1):
        axs[1, c].set_xlabel("output dim  N  (K=2048, INT8)", fontsize=8.2)
    fig.suptitle("Does our NPU model show a 32-systolic staircase? -- MEASURED per sim, SEPARATE axes "
                 "(model, NOT silicon #13; magnitudes NOT comparable)",
                 fontsize=8.2, fontweight="bold", color=WARM, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
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
    fig_systolic(load(CHAR_ONX), load(CHAR_SCL), load(CHAR_CMP))
    nps = load(SCALESIM)
    fig_spread(nps)
    fig_sensitivity(nps)
    figs = sorted(p.name for p in OUT.glob("npu_*.png"))
    print(f"wrote {len(figs)} NPU site figures: {figs}")


if __name__ == "__main__":
    main()
