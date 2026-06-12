"""M4-GPU (Mali-G610) editorial figures for the Phase-1 site (build artifact, nature-figure style).

Clean, single-purpose, white-background figures sized for the report page. M4-GPU's basic
function is the attention OFFLOAD baseline: single-head QK^T + S.V bmm fitted as latency = a + b*kv
(measured FP16 on Mali-G610), plus an FP16 roofline/ksweep slot marked as an unoptimised LOWER
BOUND (INT8 absent). Regenerable from committed JSON only. Writes PNG (+PDF/SVG) to
docs/figures/phase1-site/.

  gpu_attn_fit  -- §1/§2: single-head attn meas (dots) vs fitted a+b*kv line, median annotated.
  gpu_roofline  -- §2/§3: FP16 ksweep throughput saturating ~20.12 GFLOP/s, LOWER BOUND, INT8 absent.

Run: ./.venv/bin/python tools/plotting/site_gpu.py
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

GPU11 = ROOT / "validation/reports/phase1.1/m4_gpu.json"
RF = ROOT / "validation/reports/phase1.2/m4_gpu_roofline.json"
MALI = ROOT / "measurements/aetina/mali_matmul.json"
PARAMS = ROOT / "simulator/models/params/m4_gpu.json"
OUT = ROOT / "docs/figures/phase1-site"

HERO = "#0072B2"; WARM = "#C45A12"; OK = "#1b7f5a"; GREY = "#b9b09c"
INK = "#17150f"; SOFT = "#5b554a"; PAPER = "#fbf6ec"; GRID = "#e8e1d2"

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


def fig_attn_fit(gate, P):
    """§1/§2 -- single-head attn QK^T+S.V: measured (dots) vs fitted a+b*kv line."""
    a = P["attn_bmm_a_us"]; b = P["attn_bmm_b_us_per_kv"]
    kv = np.array(sorted(int(k) for k in gate["meas_us"]))
    meas = np.array([gate["meas_us"][str(k)] for k in kv])
    med = gate["median_relerr"]; mx = gate["max_relerr"]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    _grid(ax)
    xs = np.linspace(0, kv.max() * 1.08, 60)
    ax.plot(xs, a + b * xs, "-", color=INK, lw=1.5, zorder=2,
            label=f"fit  {a:.1f} + {b:.3f}*kv")
    ax.scatter(kv, meas, s=70, color=HERO, edgecolors="white", linewidths=0.7, zorder=4,
               label="measured (FP16, single head)")
    for k, y in zip(kv, meas):
        ax.annotate(f"kv={k}", (k, y), textcoords="offset points", xytext=(7, -11),
                    fontsize=8, color=HERO, fontweight="bold")
    ax.set_xlim(0, kv.max() * 1.12)
    ax.set_ylim(0, meas.max() * 1.12)
    ax.set_xlabel("KV length  kv  (tokens)")
    ax.set_ylabel("single-head attn latency  (us, FP16)")
    ax.text(0.04, 0.95,
            f"median |err| {med*100:.1f}%\nmax {mx*100:.1f}%\nn = {len(kv)}",
            transform=ax.transAxes, fontsize=8.5, va="top", color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc=PAPER, ec=GRID, lw=0.8))
    ax.text(0.97, 0.06, "QK^T + S.V cost is linear in kv:\na = setup, b = per-kv slope",
            transform=ax.transAxes, fontsize=8, ha="right", va="bottom", color=SOFT)
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(0.0, 0.78))
    S.save(fig, OUT / "gpu_attn_fit")


def fig_roofline(mali, rf):
    """§2/§3 -- FP16 square ksweep: throughput saturating ~g_sat (LOWER BOUND); INT8 absent."""
    ks = sorted((r for r in mali["results"] if r["group"] == "ksweep"), key=lambda r: r["M"])
    M = np.array([r["M"] for r in ks])
    f16 = np.array([r["f16_gflops"] for r in ks])
    f32 = np.array([r["f32_gflops"] for r in ks])
    g_sat = rf["calibrated_fit"]["saturated_f16_gflops"]
    sat_M = next(r["M"] for r in mali["results"]
                 if r["group"] == "ksweep" and r["f16_gflops"] >= 0.97 * g_sat)
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    _grid(ax)
    ax.plot(M, f16, "-o", color=HERO, ms=6, lw=1.5, zorder=4, label="FP16 (calibrated)")
    ax.plot(M, f32, "-s", color=GREY, ms=5, lw=1.2, mfc="white", zorder=3, label="FP32")
    ax.axhline(g_sat, ls="--", color=WARM, lw=1.2, zorder=2)
    ax.text(M.max(), g_sat + 0.5, f"FP16 saturates {g_sat:.1f} GFLOP/s", fontsize=8,
            color=WARM, ha="right", va="bottom")
    ax.axvline(sat_M, ls=":", color="#999", lw=1.0, zorder=2)
    ax.text(sat_M * 1.15, 2.5, f"saturates at M={sat_M}", fontsize=7.6, color=SOFT, va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks(M); ax.set_xticklabels([str(m) for m in M])
    ax.set_ylim(0, max(f16.max(), f32.max()) * 1.18)
    ax.set_xlabel("square matrix dim  M = K = N")
    ax.set_ylabel("matmul throughput  (GFLOP/s)")
    ax.text(0.04, 0.95,
            "LOWER BOUND: unoptimised OpenCL kernel.\nINT8 not measured (FP16 only).",
            transform=ax.transAxes, fontsize=8, va="top", color=SOFT, style="italic")
    ax.legend(loc="lower right", fontsize=8)
    S.save(fig, OUT / "gpu_roofline")


def main():
    g11 = load(GPU11)
    fig_attn_fit(g11["attn_offload_gate"], load(PARAMS))
    fig_roofline(load(MALI), load(RF))
    figs = sorted(p.name for p in OUT.glob("gpu_*.png"))
    print(f"wrote {len(figs)} GPU site figures: {figs}")


if __name__ == "__main__":
    main()
