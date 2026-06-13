"""M7 (Energy) + end-to-end editorial figures for the Phase-1 site (build artifact, nature-figure style).

Clean, single-purpose, white-background figures sized for the report page. M7's basic function is to
ESTIMATE per-token energy from activity x hardware spec power (CIM 15 TOPS/W, datasheet pJ/bit) — there
is NO power telemetry, so energy is spec-based, validated only by +/-20% sensitivity (0 conclusion
flips) and the memory-dominated order-of-magnitude. The end-to-end figure is the recompose hold-out:
predict 8B decode tok/s from a 1B+3B BW fit (the system-level silicon-validated check).
Regenerable from committed JSON only. Writes PNG (+PDF/SVG) to docs/figures/phase1-site/.

  m7_energy    — sec1/sec2: per-token 8B decode energy breakdown (CIM / DRAM / CPU), log-y, DRAM dominates.
  m7_recompose — sec3/sec4 e2e: predicted vs measured decode tok/s (1B/3B fit, 8B held-out), +/-25% band.

Run: ./.venv/bin/python tools/plotting/site_m7.py
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

M7 = ROOT / "validation/reports/phase1.1/m7.json"
RC = ROOT / "validation/reports/phase1.1/recompose.json"
OUT = ROOT / "docs/figures/phase1-site"

HERO = "#0072B2"; WARM = "#C45A12"; OK = "#1b7f5a"; GREY = "#b9b09c"
INK = "#17150f"; SOFT = "#5b554a"; PAPER = "#fbf6ec"; GRID = "#e8e1d2"
MCOL = {"llama-3.2-1b": "#5b9bc4", "llama-3.2-3b": "#2680b4", "llama-3.1-8b": WARM}
MLAB = {"llama-3.2-1b": "1B", "llama-3.2-3b": "3B", "llama-3.1-8b": "8B (hold-out)"}

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


def fig_energy():
    """sec1/sec2 — per-token 8B decode energy breakdown (log-y): DRAM streaming dominates.

    Spec-based ESTIMATE (no power telemetry); the bar heights are not measured power, the claim
    is which TERM dominates, shown robust to +/-20% (0 conclusion flips across 16 corners)."""
    m = load(M7)
    e = m["per_token_8b_decode_mJ"]
    flips = m["sensitivity_pm20pct"]["conclusion_flips"]
    corners = m["sensitivity_pm20pct"]["corners_tested"]
    rows = [("DRAM stream", e["dram_stream_mJ"], WARM),
            ("CPU support", e["cpu_support_mJ"], HERO),
            ("CIM compute", e["cim_proj_mJ"], "#5b9bc4")]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    _grid(ax)
    x = np.arange(len(rows))
    ax.bar(x, [r[1] for r in rows], width=0.58, color=[r[2] for r in rows],
           edgecolor="white", linewidth=0.6, zorder=3)
    for xi, r in zip(x, rows):
        ax.text(xi, r[1] * 1.18, f"{r[1]:.3g} mJ", ha="center", va="bottom",
                fontsize=8.5, color=r[2], fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(0.5, e["dram_stream_mJ"] * 4)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("per-token energy  (mJ, 8B decode)")
    ax.text(0.97, 0.95,
            f"memory-dominated;\n{flips} conclusion flips +/-20%\n({corners} corners)",
            transform=ax.transAxes, fontsize=8, ha="right", va="top", color=SOFT)
    ax.text(0.50, 0.62, "spec-based ESTIMATE\nno power telemetry",
            transform=ax.transAxes, fontsize=7.8, ha="center", va="center",
            color=WARM, style="italic")
    S.save(fig, OUT / "m7_energy")


def fig_recompose():
    """sec3/sec4 e2e — recompose hold-out: predict 8B decode tok/s from a 1B+3B BW fit.

    tok/s = BW_eff / per_token_weight_bytes; fit BW on 1B+3B, predict 8B (held-out) -> 9.5% error,
    within +/-25%. This is silicon-validated tok/s (the energy bars are NOT). ASCII-only text."""
    rc = load(RC)
    models = ["llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b"]
    wb = {m: rc["per_token_weight_bytes"][m] for m in models}
    meas = {m: rc["measured_tok_s_1c"][m] for m in models}
    bw = rc["fit_BW_GBs"]
    pred8 = rc["pred_8b_tok_s"]; meas8 = rc["measured_8b_tok_s"]
    err = rc["rel_error_8b"]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    _grid(ax)
    xs = np.linspace(min(wb.values()) * 0.85, max(wb.values()) * 1.1, 60)
    fit = bw / xs
    ax.fill_between(xs, fit * 0.75, fit * 1.25, color=HERO, alpha=0.10, zorder=1, label="+/-25% band")
    ax.plot(xs, fit, "-", color=INK, lw=1.4, zorder=2,
            label=f"fit on 1B+3B: tok/s = BW / bytes  (BW={bw:.2f} GB/s)")
    # 1B / 3B: fit anchors (filled). 8B: measured (open) + predicted (star).
    for m in ("llama-3.2-1b", "llama-3.2-3b"):
        ax.scatter(wb[m], meas[m], s=72, color=MCOL[m], edgecolors="white", linewidths=0.7, zorder=5)
        ax.annotate(MLAB[m], (wb[m], meas[m]), textcoords="offset points", xytext=(9, 6),
                    fontsize=9, color=MCOL[m], fontweight="bold")
    m8 = "llama-3.1-8b"
    ax.scatter(wb[m8], meas8, s=90, facecolors="none", edgecolors=WARM, linewidths=1.6, zorder=5,
               label=f"8B measured ({meas8:.2f} tok/s, held-out)")
    ax.scatter(wb[m8], pred8, s=150, color=WARM, marker="*", edgecolors="white", linewidths=0.7,
               zorder=6, label=f"8B predicted ({pred8:.2f} tok/s)")
    ax.annotate(f"8B hold-out\nerror {err*100:.1f}%  (< 25%)", xy=(wb[m8], pred8),
                xytext=(wb[m8] * 0.62, pred8 + 1.9), fontsize=8.2, color=WARM, fontweight="bold",
                ha="center", arrowprops=dict(arrowstyle="->", color=WARM, lw=1.0))
    ax.set_xlabel("per-token weight bytes  (GB, INT8)")
    ax.set_ylabel("decode throughput  (tok/s, 1-core)")
    ax.legend(loc="upper right", fontsize=7.8)
    S.save(fig, OUT / "m7_recompose")


def main():
    fig_energy()
    fig_recompose()
    figs = sorted(p.name for p in OUT.glob("m7_*.png"))
    print(f"wrote {len(figs)} M7 site figures: {figs}")


if __name__ == "__main__":
    main()
