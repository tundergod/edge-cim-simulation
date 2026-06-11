"""Phase 1.5 CIM compute 補量 — campaign result figure (build artifact, nature-figure style).

One 4-panel quantitative grid, regenerable from committed JSON (validation/reports/phase1.5/ +
phase1.2/ + measurements/metis_card/cim_card_revalidate_raw.json + params/m1_cim.json):
  (a) HERO — native M=1 multi-tile RESIDENCY CLIFF: throughput vs K*N, measured + 2-regime model.
  (b) old tile-sum vs new cliff model: measured-vs-predicted latency (31% median -> 2.4%).
  (c) dense prefill M-sweep: tile_lat vs M, affine fit; the old M_MAX=256 'wall' is busted (M<=320).
  (d) KV-cache isolation SPIKE: memory-bound proxy eff_BW vs M, ~ M2 LPDDR4x streaming BW.

Writes PNG (embedded in the HTML report) + PDF + SVG to docs/figures/phase1.5/.
Run: ./.venv/bin/python tools/plotting/phase1_5_cim.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/plotting"))
import _style as S  # noqa: E402

RAW = ROOT / "measurements/metis_card/cim_card_revalidate_raw.json"
MT = ROOT / "validation/reports/phase1.5/cim_multitile.json"
PREF = ROOT / "validation/reports/phase1.2/cim_prefill_fit.json"
KV = ROOT / "validation/reports/phase1.5/kv_append_spike.json"
PARAMS = ROOT / "simulator/models/params/m1_cim.json"
FIG = ROOT / "docs/figures/phase1.5"

RESID = S.PALETTE["matmul"]       # #0072B2 resident / measured (hero blue)
SPILL = S.PALETTE["attention"]    # #D55E00 DRAM-spill / extrapolation (orange)
OLD = S.PALETTE["residual"]       # #999999 old tile-sum
KVC = S.PALETTE["kv_cache"]       # #E69F00 kv
INK = "#272727"
W = 2048


def load(p):
    return json.loads(Path(p).read_text())


def panel_cliff(ax, raw, P):
    """(a) HERO — the residency cliff: M=1 native throughput vs K*N."""
    knee, a_r, b_r, floor = (P["multitile_knee_kn"], P["multitile_resident_a_us"],
                             P["multitile_resident_b_us"], P["multitile_floor_gops"])
    single, res, spill = [], [], []
    for r in raw.values():
        if r.get("M") != 1 or "dev_gflops" not in r or r.get("group") not in (
                "alpha13", "envelope_probe", "cliff_map", "multitile"):
            continue
        pt = (r["K"] * r["N"], r["dev_gflops"])
        if r["K"] <= W and r["N"] <= W:
            single.append(pt)
        elif pt[0] <= knee:
            res.append(pt)
        else:
            spill.append(pt)
    ax.scatter(*zip(*single), s=10, c="none", edgecolors=S.PALETTE["norm"], linewidths=0.8,
               label="single-tile (context)", zorder=3)
    ax.scatter(*zip(*res), s=16, c=RESID, label="multi-tile, SRAM-resident", zorder=4)
    ax.scatter(*zip(*spill), s=16, c=SPILL, marker="s", label="multi-tile, DRAM-spill", zorder=4)
    # 2-regime model curve over the multi-tile envelope
    kn_r = np.linspace(W * W, knee, 100)
    ax.plot(kn_r, 2 * kn_r / (a_r + b_r * kn_r) / 1e3, "-", color=RESID, lw=1.3, zorder=2)
    kn_s = np.linspace(knee, 1.8e7, 50)
    ax.plot(kn_s, [floor] * len(kn_s), "-", color=SPILL, lw=1.3, zorder=2)
    ax.axvline(knee, ls=":", color="#aaa", lw=0.9)
    ax.annotate(f"knee ~{knee/1e6:.1f}M\n(SRAM cap.)", xy=(knee, 150), xytext=(knee * 0.36, 150),
                fontsize=5.3, color="#666", ha="center",
                arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.7))
    ax.annotate("~3.5x collapse", xy=(8.6e6, 90), xytext=(1.05e7, 165), fontsize=5.6, color=SPILL,
                arrowprops=dict(arrowstyle="->", color=SPILL, lw=0.8))
    ax.set_xscale("log")
    ax.set_xlabel("GEMM size  K·N (params)")
    ax.set_ylabel("INT8 throughput (GOP/s), M=1")
    ax.set_title("a · native multi-tile residency cliff", loc="left", fontweight="bold")
    ax.legend(fontsize=5.2, loc="upper left")
    ax.set_ylim(0, 290)


def panel_oldnew(ax, mt):
    """(b) measured vs predicted latency: old tile-sum vs new cliff model."""
    rows = mt["per_point"]
    meas = [r["meas_us"] for r in rows]
    old = [r["old_tilesum_us"] for r in rows]
    new = [r["new_cliff_us"] for r in rows]
    lim = [30, 600]
    ax.plot(lim, lim, "--", color="#bbb", lw=0.9, zorder=1)
    ax.scatter(meas, old, s=14, c=OLD, marker="^", label="old tile-sum", zorder=3)
    ax.scatter(meas, new, s=14, c=RESID, label="new cliff model", zorder=4)
    ov = mt["old_vs_new"]
    ax.text(0.04, 0.94, f"median |err|\n  old tile-sum: {ov['old_tilesum_median']*100:.0f}%\n"
            f"  new cliff: {ov['new_cliff_median']*100:.1f}%\n  (held-out {mt['resident_holdout']['median_relerr']*100:.1f}%, n={ov['n']})",
            transform=ax.transAxes, fontsize=5.4, va="top", color=INK,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f6f2ea", ec="#ddd5c5", lw=0.6))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("measured latency (µs)")
    ax.set_ylabel("predicted latency (µs)")
    ax.set_title("b · model error vs Card-native", loc="left", fontweight="bold")
    ax.legend(fontsize=5.4, loc="lower right")


def panel_prefill(ax, pref, P):
    """(c) dense prefill M-sweep + the busted M_MAX=256 wall."""
    fp = pref["fit_points"]
    M = np.array([r["M"] for r in fp]); lat = np.array([r["meas_us"] for r in fp])
    a, b = pref["affine_fit_tile_lat_us"]["a_weight_load_us"], pref["affine_fit_tile_lat_us"]["b_per_col_us"]
    anchor = pref["decode_anchor_M1_measured"]
    M_max = P["prefill_M_max"]
    xs = np.linspace(1, M_max + 15, 60)
    ax.plot(xs, a + b * xs, "-", color=INK, lw=1.1, zorder=2, label=f"affine  {a:.1f}+{b:.3f}·M")
    ax.scatter(M, lat, s=13, c=RESID, zorder=4, label="dense sweep (measured)")
    ax.scatter([anchor["M"]], [anchor["tile_lat_us"]], s=34, c=SPILL, marker="*", zorder=5,
               label="M=1 decode anchor")
    ax.axvline(256, ls=":", color="#bbb", lw=1.0)
    ax.text(248, 44, "old assumed\nM_MAX=256\n(2× too low)", fontsize=5.0, color="#999", va="top", ha="right")
    ax.axvline(510, ls="--", color=SPILL, lw=1.1)
    ax.text(506, 90, f"real wall ~M=510\n(M={M_max} ok, 511 fails)", fontsize=5.2, color=SPILL, va="top", ha="right")
    ax.set_xlim(-15, 560)
    ax.set_xlabel("activation columns  M  (canonical 2048×2048 tile)")
    ax.set_ylabel("tile latency (µs)")
    ax.set_title("c · prefill M-amortization, dense", loc="left", fontweight="bold")
    ax.legend(fontsize=5.3, loc="upper left")
    ho = pref["holdout"]["median_rel_err"]
    ax.text(0.97, 0.06, f"held-out median {ho*100:.1f}%", transform=ax.transAxes, fontsize=5.4,
            ha="right", color="#666")


def panel_kv(ax, kv):
    """(d) KV SPIKE: proxy can't reach DRAM (compile wall < knee, eff_BW rises non-converged); the
    CONVERGED DRAM BW comes from the cliff spill regime (flat across K·N 8-17M)."""
    pts = kv["proxy_points"]
    ws = np.array([p["workset_elems"] / 1e6 for p in pts]); bw = np.array([p["eff_BW_GBs"] for p in pts])
    sp = kv["spill_regime_dram_bw"]
    sx = np.array([s["kn_M"] for s in sp]); sy = np.array([s["dram_BW_GBs"] for s in sp])
    v = kv["verdict"]
    m2, knee, dram = v["m2_measured_eff_BW_GBs"], v["sram_knee_M_elems"], v["spill_dram_BW_GBs_converged"]
    ax.axvspan(0.05, knee, color="#eef2f6", zorder=0)
    ax.axvline(knee, ls=":", color="#999", lw=0.9)
    ax.text(knee * 0.9, 49, "SRAM | DRAM\nknee", fontsize=4.8, color="#888", ha="right", va="top")
    ax.plot(ws, bw, "-o", color=KVC, ms=4, lw=1.3, zorder=4, label="K=1 proxy (rises, SRAM-resident)")
    ax.plot(sx, sy, "s", color=SPILL, ms=4.5, zorder=5, label=f"cliff spill = DRAM-bound ({dram:.0f} GB/s, flat)")
    ax.axhline(dram, ls="-", color=SPILL, lw=0.9, alpha=0.6, zorder=2)
    ax.axhline(m2, ls="--", color=RESID, lw=1.0, zorder=2)
    ax.text(0.06, m2 - 3.3, f"M2 effective {m2:.1f}", fontsize=5.0, color=RESID)
    ax.annotate("compile wall\n(proxy can't grow\npast here)", xy=(2.1, 44.4), xytext=(0.42, 30),
                fontsize=4.8, color="#666", ha="center",
                arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.7))
    ax.set_xscale("log")
    ax.set_xlabel("data size (M-elements):  proxy N·M  /  spill K·N")
    ax.set_ylabel("bandwidth (GB/s)")
    ax.set_title("d · KV-append BW — proxy fails, spill converges", loc="left", fontweight="bold")
    ax.set_ylim(0, 52); ax.set_xlim(0.1, 22)
    ax.text(0.97, 0.05, "proxy never reaches DRAM\n→ CONVERGED DRAM BW from\n   cliff spill regime",
            transform=ax.transAxes, fontsize=5.0, ha="right", color=SPILL)
    ax.legend(fontsize=5.0, loc="upper left")


def main():
    raw, mt, pref, kv, P = load(RAW), load(MT), load(PREF), load(KV), load(PARAMS)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    panel_cliff(axes[0, 0], raw, P)
    panel_oldnew(axes[0, 1], mt)
    panel_prefill(axes[1, 0], pref, P)
    panel_kv(axes[1, 1], kv)
    fig.suptitle(f"Phase 1.5 — CIM compute supplement: on-Card measurement campaign (Metis Card, {len(raw)} tasks)",
                 fontsize=8.5, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    S.save(fig, str(FIG / "phase1_5_cim_campaign"))
    print(f"wrote {FIG/'phase1_5_cim_campaign'}.png (+pdf/svg)")


if __name__ == "__main__":
    main()
