"""M1 (CIM) editorial figures for the Phase-1 site (build artifact, nature-figure style).

Clean, single-purpose, white-background figures sized for the report page — one claim each,
direct labels over legend boxes, no baked-in redundant titles (the page caption carries them).
Regenerable from committed JSON only. Writes PNG (+PDF/SVG) to docs/figures/phase1-site/.

  m1_geff    — §1/§2 hero: G_eff(N,K) measured dots vs 2D fit lines (per K).
  m1_parity  — §2 fit quality: measured vs predicted GOP/s, ±10/20% bands, gate annotated.
  m1_prefill — §2 prefill: tile_lat vs M affine + M=1 decode anchor + real ~M=510 wall.
  m1_cliff   — §3 sim/extrap: residency cliff throughput vs K·N, resident→knee→spill floor.

Run: ./.venv/bin/python tools/plotting/site_m1.py
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
import _style as S  # noqa: E402  (for save(); rcParams overridden below for editorial sizing)
from simulator.models.m1_cim_tile import CimTileModel  # noqa: E402

AET = ROOT / "measurements/aetina"
RAW = ROOT / "measurements/metis_card/cim_card_revalidate_raw.json"
MT = ROOT / "validation/reports/phase1.5/cim_multitile.json"
PREF = ROOT / "validation/reports/phase1.2/cim_prefill_fit.json"
PARAMS = ROOT / "simulator/models/params/m1_cim.json"
OUT = ROOT / "docs/figures/phase1-site"

# editorial palette — page CIM-blue hero, warm accent, restrained neutrals
HERO = "#0072B2"; WARM = "#C45A12"; INK = "#17150f"; SOFT = "#5b554a"; FAINT = "#b9b09c"
PAPER = "#fbf6ec"; GRID = "#e8e1d2"
# sequential blue ramp for ascending K (darker = wider K = higher throughput)
KRAMP = {1024: "#a8cbe0", 2048: "#5b9bc4", 3072: "#2680b4", 3584: "#0a6298", 4096: WARM}

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
    ax.grid(True, which="major", axis="both", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def fig_geff(P):
    """§1/§2 — G_eff(N,K): measured (dots) vs 2D fit (lines), direct-labelled per K."""
    m = CimTileModel(P)
    pts = P["native_throughput_points"]
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    _grid(ax)
    labs = []
    for K in sorted({p["K"] for p in pts}):
        kp = sorted((p for p in pts if p["K"] == K), key=lambda x: x["N"])
        c = KRAMP.get(K, "#999")
        Nf = np.linspace(64, 2048, 60)
        ax.plot(Nf, [m.g_eff(n, K) for n in Nf], "-", color=c, lw=1.6, zorder=2)
        ax.scatter([p["N"] for p in kp], [p["gops"] for p in kp], s=34, color=c,
                   edgecolors="white", linewidths=0.6, zorder=4)
        labs.append([m.g_eff(2048, K), c, f"K={K}"])
    labs.sort()                                   # declutter: enforce min vertical gap between labels
    for i in range(1, len(labs)):
        if labs[i][0] - labs[i - 1][0] < 9:
            labs[i][0] = labs[i - 1][0] + 9
    for y, c, t in labs:
        ax.text(2090, y, t, color=c, fontsize=8.5, va="center", fontweight="bold")
    ax.set_xlim(0, 2400)
    ax.set_xlabel("output channels  N")
    ax.set_ylabel("effective throughput  (GOP/s, INT8)")
    ax.text(0.015, 0.97, "measured · 2D fit", transform=ax.transAxes, fontsize=8.5,
            va="top", color=SOFT, style="italic")
    ax.annotate("wider K, higher throughput\n(K effect is fittable)", xy=(820, 145), fontsize=8,
                color=SOFT, ha="left")
    S.save(fig, OUT / "m1_geff")


def fig_parity(P):
    """§2 — measured vs predicted GOP/s parity with ±10/±20% bands (the decode gate, visual)."""
    m = CimTileModel(P)
    pts = P["native_throughput_points"]
    meas = np.array([p["gops"] for p in pts])
    pred = np.array([m.g_eff(p["N"], p["K"]) for p in pts])
    err = np.abs(pred - meas) / meas
    lim = [0, 250]
    fig, ax = plt.subplots(figsize=(4.0, 3.7))
    _grid(ax)
    x = np.array(lim)
    ax.fill_between(x, x * 0.8, x * 1.2, color=HERO, alpha=0.07, zorder=1, label="±20%")
    ax.fill_between(x, x * 0.9, x * 1.1, color=HERO, alpha=0.13, zorder=1, label="±10%")
    ax.plot(lim, lim, "-", color="#aaa", lw=1.0, zorder=2)
    cols = [KRAMP.get(p["K"], "#999") for p in pts]
    ax.scatter(meas, pred, s=44, c=cols, edgecolors="white", linewidths=0.6, zorder=4)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("measured  (GOP/s)")
    ax.set_ylabel("predicted  G_eff(N,K)  (GOP/s)")
    ax.text(0.04, 0.96,
            f"median |err| {np.median(err)*100:.1f}%\np95 {np.percentile(err,95)*100:.1f}%\nn = {len(pts)}",
            transform=ax.transAxes, fontsize=8.5, va="top", color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc=PAPER, ec=GRID, lw=0.8))
    ax.legend(loc="lower right", fontsize=8)
    S.save(fig, OUT / "m1_parity")


def fig_prefill(pref, P):
    """§2 — dense prefill M-amortization: affine tile_lat(M) + M=1 decode anchor + real ~M=510 wall."""
    fp = pref["fit_points"]
    M = np.array([r["M"] for r in fp]); lat = np.array([r["meas_us"] for r in fp])
    a = pref["affine_fit_tile_lat_us"]["a_weight_load_us"]
    b = pref["affine_fit_tile_lat_us"]["b_per_col_us"]
    anchor = pref["decode_anchor_M1_measured"]
    M_max = P["prefill_M_max"]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    _grid(ax)
    xs = np.linspace(1, M_max + 15, 80)
    ax.plot(xs, a + b * xs, "-", color=INK, lw=1.6, zorder=2,
            label=f"affine  {a:.1f} + {b:.3f}·M")
    ax.scatter(M, lat, s=26, color=HERO, edgecolors="white", linewidths=0.5, zorder=4,
               label="dense Card sweep")
    ax.scatter([anchor["M"]], [anchor["tile_lat_us"]], s=150, color=WARM, marker="*",
               edgecolors="white", linewidths=0.7, zorder=5, label="M=1 decode anchor")
    ax.axvline(256, ls=":", color=FAINT, lw=1.1)
    ax.text(248, a + 6, "old assumed M_MAX=256\n(2× too low)", fontsize=7.5,
            color=SOFT, va="bottom", ha="right")
    ax.axvline(510, ls="--", color=WARM, lw=1.3)
    ax.text(504, a + 8, f"real wall ~M=510\nM={M_max} ok · 511 fails", fontsize=7.8,
            color=WARM, va="bottom", ha="right")
    ax.set_xlim(-15, 575)
    ax.set_xlabel("activation columns  M   (canonical 2048×2048 tile)")
    ax.set_ylabel("tile latency  (µs)")
    ax.legend(loc="upper left", fontsize=8)
    S.save(fig, OUT / "m1_prefill")


def fig_cliff(raw, P):
    """§3 — native multi-tile residency cliff: M=1 throughput vs K·N, resident→knee→spill floor."""
    knee = P["multitile_knee_kn"]; a_r = P["multitile_resident_a_us"]
    b_r = P["multitile_resident_b_us"]; floor = P["multitile_floor_gops"]
    envelope = P["native_envelope_kn"]; W = 2048
    single, res, spill = [], [], []
    for r in raw.values():
        if r.get("M") != 1 or "dev_gflops" not in r or r.get("group") not in (
                "alpha13", "envelope_probe", "cliff_map", "multitile"):
            continue
        pt = (r["K"] * r["N"], r["dev_gflops"])
        (single if r["K"] <= W and r["N"] <= W else res if pt[0] <= knee else spill).append(pt)
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    _grid(ax)
    ax.axvspan(envelope, 3e7, color="#f3eee3", zorder=0)
    ax.text(envelope * 1.15, 250, "K·N > native envelope\n= floor-extrapolated", fontsize=7.3,
            color=SOFT, va="top")
    ax.scatter(*zip(*single), s=22, facecolors="none", edgecolors=FAINT, linewidths=1.0,
               label="single-tile (context)", zorder=3)
    ax.scatter(*zip(*res), s=34, color=HERO, edgecolors="white", linewidths=0.5,
               label="multi-tile · SRAM-resident", zorder=4)
    ax.scatter(*zip(*spill), s=34, color=WARM, marker="s", edgecolors="white", linewidths=0.5,
               label="multi-tile · DRAM-spill", zorder=4)
    kn_r = np.linspace(W * W, knee, 100)
    ax.plot(kn_r, 2 * kn_r / (a_r + b_r * kn_r) / 1e3, "-", color=HERO, lw=1.8, zorder=2)
    kn_s = np.linspace(knee, 2.6e7, 50)
    ax.plot(kn_s, [floor] * len(kn_s), "-", color=WARM, lw=1.8, zorder=2)
    ax.axvline(knee, ls=":", color="#999", lw=1.0)
    ax.annotate(f"knee ~{knee/1e6:.1f}M params\n(on-chip SRAM cap.)", xy=(knee, 158),
                xytext=(knee * 0.30, 158), fontsize=7.6, color=SOFT, ha="center",
                arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.8))
    ax.annotate(f"~3.5× collapse\nto ~{floor:.0f} GOP/s", xy=(8.9e6, 95), xytext=(1.2e7, 185),
                fontsize=8, color=WARM, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARM, lw=1.0))
    ax.set_xscale("log"); ax.set_ylim(0, 295); ax.set_xlim(3e5, 2.6e7)
    ax.set_xlabel("GEMM size  K·N  (params)")
    ax.set_ylabel("INT8 throughput  (GOP/s), M=1")
    ax.legend(loc="upper left", fontsize=8)
    S.save(fig, OUT / "m1_cliff")


def main():
    P = load(PARAMS)
    fig_geff(P)
    fig_parity(P)
    fig_prefill(load(PREF), P)
    fig_cliff(load(RAW), P)
    figs = sorted(p.name for p in OUT.glob("m1_*.png"))
    print(f"wrote {len(figs)} M1 site figures: {figs}")


if __name__ == "__main__":
    main()
