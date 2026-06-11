"""Phase 1.5 CIM compute 補量 — campaign result figure (build artifact, nature-figure style).

One 5-panel quantitative grid, regenerable from committed JSON (validation/reports/phase1.5/ +
phase1.2/ + measurements/metis_card/cim_card_revalidate_raw.json + params/m1_cim.json):
  (a) HERO — native M=1 multi-tile RESIDENCY CLIFF: throughput vs K*N, measured + 2-regime model.
  (b) old tile-sum vs new cliff model: measured-vs-predicted latency (31% median -> 2.8%).
  (c) dense prefill M-sweep: tile_lat vs M, affine fit; old M_MAX=256 'wall' busted (real wall ~M=510).
  (d) KV-cache isolation SPIKE: proxy can't reach DRAM (structural); converged DRAM BW from spill regime.
  (e) K & N staircases: both compile natively past 2048 (= output tile width W), same K*N cliff.

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


def panel_staircase(ax, raw):
    """(e) K=2048 is NOT a K/N compile limit: sweep K (N=512) and N (K=512); both compile natively
    past 2048, throughput rising, same K·N cliff at 16384. 2048 is just the output tile width W."""
    kk = sorted((v["K"], v["dev_gflops"]) for v in raw.values() if v.get("group") == "k_staircase" and "dev_gflops" in v)
    nn = sorted((v["N"], v["dev_gflops"]) for v in raw.values() if v.get("group") == "n_staircase" and "dev_gflops" in v)
    ax.plot([k for k, _ in kk], [g for _, g in kk], "-o", color=RESID, ms=3.5, lw=1.2, label="sweep K (N=512, M=1)")
    ax.plot([n for n, _ in nn], [g for _, g in nn], "-s", color=S.PALETTE["ffn"], ms=3.3, lw=1.2,
            mfc="none", label="sweep N (K=512, M=1)")
    ax.axvline(2048, ls=":", color="#999", lw=1.0)
    ax.text(2200, 25, "W=2048\noutput tile width\n— NOT a K/N\ncompile limit", fontsize=5.0, color="#888", va="bottom")
    ax.annotate("both native to ≥16384;\nsame K·N cliff (8.4M)", xy=(16384, 70), xytext=(3000, 120),
                fontsize=5.2, color=SPILL, arrowprops=dict(arrowstyle="->", color=SPILL, lw=0.8))
    ax.set_xscale("log")
    ax.set_xlabel("swept dimension  K or N  (other = 512)")
    ax.set_ylabel("INT8 throughput (GOP/s)")
    ax.set_title("e · K & N compile natively past 2048", loc="left", fontweight="bold")
    ax.legend(fontsize=5.2, loc="lower right")


def panel_kv(ax, kv):
    """(d) KV SPIKE: proxy can't reach DRAM (compile wall < knee, eff_BW rises non-converged whether you
    grow M or N); the CONVERGED DRAM BW comes from the cliff spill regime (flat across K·N 8-17M)."""
    pts = kv["proxy_points"]
    msweep = [p for p in pts if p["N"] == 2048]          # clean M-sweep at N=2048 (the line)
    grow_n = [p for p in pts if p["N"] != 2048]          # N-varied DRAM-push (same-transfer cross-check)
    sp = kv["spill_regime_dram_bw"]
    sx = [s["kn_M"] for s in sp]; sy = [s["dram_BW_GBs"] for s in sp]
    v = kv["verdict"]
    m2, knee, dram = v["m2_measured_eff_BW_GBs"], v["sram_knee_M_elems"], v["spill_dram_BW_GBs_converged"]
    ax.axvspan(0.05, knee, color="#eef2f6", zorder=0)
    ax.axvline(knee, ls=":", color="#999", lw=0.9)
    ax.text(knee * 0.9, 49, "SRAM | DRAM\nknee", fontsize=4.8, color="#888", ha="right", va="top")
    ax.plot([p["workset_elems"] / 1e6 for p in msweep], [p["eff_BW_GBs"] for p in msweep], "-o",
            color=KVC, ms=4, lw=1.3, zorder=4, label="K=1 proxy, grow M (N=2048)")
    if grow_n:
        ax.plot([p["workset_elems"] / 1e6 for p in grow_n], [p["eff_BW_GBs"] for p in grow_n], "D",
                color=KVC, mfc="white", ms=5, zorder=5, label="proxy, grow N (lands on same trend)")
    ax.plot(sx, sy, "s", color=SPILL, ms=4.5, zorder=5, label=f"cliff spill = DRAM-bound ({dram:.0f} GB/s, flat)")
    ax.axhline(dram, ls="-", color=SPILL, lw=0.9, alpha=0.6, zorder=2)
    ax.axhline(m2, ls="--", color=RESID, lw=1.0, zorder=2)
    ax.text(0.06, m2 - 3.4, f"M2 effective {m2:.1f}", fontsize=5.0, color=RESID)
    ax.annotate("compile wall\n(proxy can't grow past)", xy=(2.1, 44.4), xytext=(0.45, 28),
                fontsize=4.8, color="#666", ha="center", arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.7))
    ax.set_xscale("log")
    ax.set_xlabel("transfer size (M-elements):  proxy N·M  /  spill K·N")
    ax.set_ylabel("bandwidth (GB/s)")
    ax.set_title("d · KV-append BW — proxy fails, spill converges", loc="left", fontweight="bold")
    ax.set_ylim(0, 52); ax.set_xlim(0.1, 22)
    ax.text(0.97, 0.05, "proxy never reaches DRAM\n→ CONVERGED DRAM BW from\n   cliff spill regime",
            transform=ax.transAxes, fontsize=5.0, ha="right", color=SPILL)
    ax.legend(fontsize=4.8, loc="upper left")


def main():
    raw, mt, pref, kv, P = load(RAW), load(MT), load(PREF), load(KV), load(PARAMS)
    fig, axd = plt.subplot_mosaic([["a", "a", "b", "b", "e", "e"],
                                   ["c", "c", "c", "d", "d", "d"]], figsize=(10.6, 5.9))
    panel_cliff(axd["a"], raw, P)
    panel_oldnew(axd["b"], mt)
    panel_staircase(axd["e"], raw)
    panel_prefill(axd["c"], pref, P)
    panel_kv(axd["d"], kv)
    fig.suptitle(f"Phase 1.5 — CIM compute supplement: on-Card measurement campaign (Metis Card, {len(raw)} tasks)",
                 fontsize=8.5, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    S.save(fig, str(FIG / "phase1_5_cim_campaign"))
    print(f"wrote {FIG/'phase1_5_cim_campaign'}.png (+pdf/svg)")


if __name__ == "__main__":
    main()
