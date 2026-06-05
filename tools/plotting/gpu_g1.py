"""Phase 1.2 Fig G1 — Mali-G610 analytic roofline (FP16) vs micro-benchmark points.

Claim (D4): the FP16-calibrated roofline latency = max(compute, memory) is a SHAPE-TREND
+ LOWER BOUND — it tracks the measured FP16 points (decode = memory-bound, prefill/ksweep
= compute-bound) but sits AT-OR-BELOW them (an unoptimised kernel, 5 saturation pts -> not
transferable). INT8 = zero data; this is FP16 only.

Reads  validation/reports/phase1.2/m4_gpu_roofline.json (committed, the calibrated fit).
Writes docs/figures/phase1.2/G1.png (+pdf/svg).
Run: ./.venv/bin/python tools/plotting/gpu_g1.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _style import PALETTE, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "validation/reports/phase1.2/m4_gpu_roofline.json"
FIG = ROOT / "docs/figures/phase1.2"

_BOUND_MARK = {"memory": ("o", PALETTE["kv_cache"], "decode (memory-bound)"),
               "compute": ("s", PALETTE["matmul"], "prefill/ksweep (compute-bound)")}


def fig_g1():
    rep = json.loads(REPORT.read_text())
    pts = rep["error_vs_1p1_measured"]["per_point"]
    fit = rep["calibrated_fit"]

    fig, (ax, axr) = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.25, 1]})

    # --- (a) measured vs roofline-predicted latency (log-log), y=x reference ---
    lim = [5, 1e7]
    ax.plot(lim, lim, ls="--", lw=0.7, color="0.5", zorder=1)
    ax.fill_between(lim, lim, [lim[0], lim[0]], color="0.93", zorder=0)  # below y=x = lower-bound band
    seen = set()
    for p in pts:
        mark, col, lbl = _BOUND_MARK[p["bound"]]
        ax.scatter(p["meas_us"], p["pred_us"], s=20, marker=mark, color=col,
                   edgecolor="white", linewidth=0.3, zorder=3,
                   label=lbl if lbl not in seen else None)
        seen.add(lbl)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("measured FP16 latency (us)")
    ax.set_ylabel("roofline predicted (us)")
    ax.text(8, 2e6, "roofline ≤ measured\n(lower bound)", fontsize=5.5, color="0.4")
    ax.set_title("(a) roofline vs micro-benchmark (FP16)", fontsize=7.5)
    ax.legend(fontsize=5.8, loc="lower right")

    # --- (b) signed relative error per point, grouped by bound ---
    re = [p["rel_err"] for p in pts]
    cols = [_BOUND_MARK[p["bound"]][1] for p in pts]
    x = np.arange(len(pts))
    axr.bar(x, re, color=cols, width=0.85)
    axr.axhline(0, color="0.3", lw=0.6)
    axr.set_xticks([]); axr.set_xlabel(f"{len(pts)} micro-benchmark shapes")
    axr.set_ylabel("signed rel. error\n(pred − meas)/meas")
    e = rep["error_vs_1p1_measured"]
    axr.set_title(f"(b) median|err|={e['median_abs_relerr']:.0%}, p95={e['p95_abs_relerr']:.0%}",
                  fontsize=7.5)

    fig.suptitle(f"Mali-G610 analytic roofline (FP16-calibrated, LOWER BOUND): "
                 f"ceil={fit['saturated_f16_gflops']:.1f} GFLOP/s, BW={fit['mem_eff_BW_GBs']:.2f} GB/s "
                 f"— INT8: zero data", fontsize=8, y=1.04)
    fig.tight_layout()
    save(fig, str(FIG / "G1"))


def main():
    fig_g1()
    print(f"wrote {FIG}/G1.png")


if __name__ == "__main__":
    main()
