"""M2 (Memory) editorial figures for the Phase-1 site (build artifact, nature-figure style).

Clean, single-purpose, white-background figures sized for the report page. M2's basic function is
"move bytes": DRAM streaming bandwidth (the decode memory wall) + the PCIe per-call transfer floor.
Regenerable from committed JSON only. Writes PNG (+PDF/SVG) to docs/figures/phase1-site/.

  m2_decode_wall — §1: decode tok/s vs weight bytes (1B/3B/8B) → the memory-bound signature.
  m2_pcie_floor  — §1: per-call host<->device floor across 30 shapes (Alpha topology).
  m2_bw_specs    — §2: 3 DRAM specs, peak vs eff_BW, honesty-coloured, 24.2 calibrated anchor.
  m2_ramulator2  — §3: Ramulator2 device eff (0.92) vs analytic system eff (0.65) — not a conflict.
  m2_kv_spike    — §3: KV-append SPIKE — proxy is SRAM-confined, can't reach DRAM (INCONCLUSIVE).

Run: ./.venv/bin/python tools/plotting/site_m2.py
"""
import json
import statistics
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

SPECS = ROOT / "simulator/specs"
RC = ROOT / "validation/reports/phase1.1/recompose.json"
RAWM = ROOT / "measurements/aetina/metis_alpha_matmul_raw.json"
RAM2 = ROOT / "validation/reports/phase1.3/m2_ramulator2.json"
KV = ROOT / "validation/reports/phase1.5/kv_append_spike.json"
OUT = ROOT / "docs/figures/phase1-site"

HERO = "#0072B2"; WARM = "#C45A12"; OK = "#1b7f5a"; GREY = "#b9b09c"
INK = "#17150f"; SOFT = "#5b554a"; PAPER = "#fbf6ec"; GRID = "#e8e1d2"
MCOL = {"llama-3.2-1b": "#5b9bc4", "llama-3.2-3b": "#2680b4", "llama-3.1-8b": HERO}
MLAB = {"llama-3.2-1b": "1B", "llama-3.2-3b": "3B", "llama-3.1-8b": "8B"}

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


def fig_decode_wall():
    """§1 — decode tok/s vs per-token weight bytes: the memory-bound signature (tok/s ∝ 1/bytes)."""
    rc = load(RC)
    models = ["llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b"]
    wb = np.array([rc["per_token_weight_bytes"][m] for m in models])      # GB/token
    tok = np.array([rc["measured_tok_s_1c"][m] for m in models])
    bw = rc["fit_BW_GBs"]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    _grid(ax)
    xs = np.linspace(wb.min() * 0.85, wb.max() * 1.1, 50)
    ax.plot(xs, bw / xs, "-", color=INK, lw=1.4, zorder=2, label=f"tok/s = BW / bytes  (BW≈{bw:.1f} GB/s)")
    for m in models:
        ax.scatter(rc["per_token_weight_bytes"][m], rc["measured_tok_s_1c"][m], s=70,
                   color=MCOL[m], edgecolors="white", linewidths=0.7, zorder=4)
        ax.annotate(MLAB[m], (rc["per_token_weight_bytes"][m], rc["measured_tok_s_1c"][m]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9, color=MCOL[m],
                    fontweight="bold")
    ax.set_xlabel("per-token weight bytes  (GB, INT8)")
    ax.set_ylabel("decode throughput  (tok/s, 1-core)")
    ax.text(0.97, 0.94, "decode is memory-bound:\ntok/s set by weight streaming,\nnot compute",
            transform=ax.transAxes, fontsize=8, ha="right", va="top", color=SOFT)
    ax.legend(loc="lower left", fontsize=8)
    S.save(fig, OUT / "m2_decode_wall")


def fig_pcie_floor():
    """§1 — per-call host<->device floor (system − device latency) across single-tile shapes."""
    raw = load(RAWM)
    pts = [(r["dev_lat_us"], r["system_lat_us"] - r["dev_lat_us"]) for r in raw.values()
           if "system_lat_us" in r and not r.get("tiled_extrapolated") and r.get("tiles", 1) == 1]
    devs = [d for d, _ in pts]; floors = [f for _, f in pts]
    med = statistics.median(floors); p95 = sorted(floors)[int(0.95 * len(floors))]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    _grid(ax)
    ax.scatter(devs, floors, s=40, color=HERO, alpha=0.85, edgecolors="white", linewidths=0.5,
               zorder=4, label=f"measured floor ({len(pts)} shapes)")
    ax.axhline(med, color=WARM, lw=1.5, zorder=3, label=f"model floor = {med:.0f} µs (median)")
    ax.axhline(p95, ls=":", color=SOFT, lw=1.1, zorder=3, label=f"p95 = {p95:.0f} µs")
    ax.set_ylim(0, max(floors) * 1.15)
    ax.set_xlabel("device compute latency  (µs)")
    ax.set_ylabel("per-call host-device floor  (µs)")
    ax.text(0.04, 0.10, "floor dominates: a fixed per-call\ncost, ~independent of compute size",
            transform=ax.transAxes, fontsize=8, color=SOFT, va="bottom")
    ax.legend(loc="upper right", fontsize=8)
    S.save(fig, OUT / "m2_pcie_floor")


def fig_bw_specs():
    """§2 — three DRAM specs: theoretical peak (faint) vs effective BW (honesty-coloured) + anchor."""
    rows = []
    for s, col, tag in [("lpddr4", GREY, "assumption"), ("lpddr4x", HERO, "calibrated"),
                        ("lpddr5", WARM, "simulated")]:
        d = load(SPECS / f"mem_{s}.json")
        rows.append((s.upper(), d["peak_GBs"], d["eff_BW_GBs"], col, tag))
    anchor = next(r[2] for r in rows if r[0] == "LPDDR4X")
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    _grid(ax)
    x = np.arange(len(rows))
    ax.bar(x, [r[1] for r in rows], width=0.62, color="#efe9dd", edgecolor=GREY, lw=0.8,
           zorder=2, label="theoretical peak (assumption)")
    ax.bar(x, [r[2] for r in rows], width=0.62, color=[r[3] for r in rows], zorder=3)
    ax.axhline(anchor, ls="--", color=HERO, lw=1.1, zorder=4)
    ax.text(len(rows) - 0.5, anchor + 0.7, f"calibrated anchor {anchor:.1f} GB/s", fontsize=8,
            color=HERO, ha="right", va="bottom")
    for xi, r in zip(x, rows):
        ax.text(xi, r[2] - 2.4, f"{r[2]:.1f}", ha="center", color="white", fontsize=8.5, fontweight="bold")
        ax.text(xi, r[1] + 0.6, r[4], ha="center", color=r[3], fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("bandwidth  (GB/s)")
    ax.set_ylim(0, max(r[1] for r in rows) * 1.16)
    ax.legend(loc="upper left", fontsize=8)
    S.save(fig, OUT / "m2_bw_specs")


def fig_ramulator2():
    """§3 — Ramulator2 DRAM-device eff (0.92) vs analytic system eff (0.65): not a contradiction."""
    d = load(RAM2)
    dev_e, dev_bw = d["ramulator2_device"]["efficiency"], d["ramulator2_device"]["eff_BW_GBs"]
    sys_e, sys_bw = d["analytic_system"]["efficiency"], d["analytic_system"]["eff_BW_GBs"]
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    _grid(ax)
    bars = ax.bar([0, 1], [dev_e, sys_e], width=0.55, color=[WARM, HERO], zorder=3)
    for xi, e, bw, lab in [(0, dev_e, dev_bw, "Ramulator2 device\n(DRAM timing only)"),
                           (1, sys_e, sys_bw, "analytic system\n(silicon-calibrated)")]:
        ax.text(xi, e + 0.02, f"{e:.2f}\n{bw:.1f} GB/s", ha="center", fontsize=8.5, fontweight="bold")
        ax.text(xi, -0.085, lab, ha="center", fontsize=8, color=SOFT)
    ax.annotate("", xy=(1, sys_e), xytext=(0, dev_e),
                arrowprops=dict(arrowstyle="<->", color=SOFT, lw=0.9))
    ax.text(0.5, (dev_e + sys_e) / 2 + 0.03, "gap = system overhead\n(controller/NoC/queue),\nnot a device limit",
            ha="center", fontsize=7.8, color=SOFT)
    ax.set_xticks([]); ax.set_ylim(0, 1.05)
    ax.set_ylabel("single-stream efficiency  (of peak)")
    S.save(fig, OUT / "m2_ramulator2")


def fig_kv_spike():
    """§3 — KV-append SPIKE: K=1 proxy is compile-confined below the DRAM knee (SRAM-bound,
    eff_BW never converges); the converged on-card DRAM BW comes from the cliff spill regime."""
    kv = load(KV)
    pts = kv["proxy_points"]
    msweep = [p for p in pts if p["N"] == 2048]
    grow_n = [p for p in pts if p["N"] != 2048]
    sp = kv["spill_regime_dram_bw"]
    v = kv["verdict"]
    knee, m2, dram = v["sram_knee_M_elems"], v["m2_measured_eff_BW_GBs"], v["spill_dram_BW_GBs_converged"]
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    _grid(ax)
    ax.axvspan(0.05, knee, color="#eef2f6", zorder=0)
    ax.axvline(knee, ls=":", color="#999", lw=1.0)
    ax.text(knee * 0.92, 50, "SRAM | DRAM\nknee", fontsize=7.5, color=SOFT, ha="right", va="top")
    ax.plot([p["workset_elems"] / 1e6 for p in msweep], [p["eff_BW_GBs"] for p in msweep], "-o",
            color="#E69F00", ms=5, lw=1.4, zorder=4, label="K=1 proxy, grow M (N=2048)")
    if grow_n:
        ax.plot([p["workset_elems"] / 1e6 for p in grow_n], [p["eff_BW_GBs"] for p in grow_n], "D",
                color="#E69F00", mfc="white", ms=6, zorder=5, label="proxy, grow N (same trend)")
    ax.plot([s["kn_M"] for s in sp], [s["dram_BW_GBs"] for s in sp], "s", color=WARM, ms=6, zorder=5,
            label=f"cliff spill = DRAM-bound ({dram:.0f} GB/s, flat)")
    ax.axhline(dram, ls="-", color=WARM, lw=0.9, alpha=0.6, zorder=2)
    ax.axhline(m2, ls="--", color=HERO, lw=1.1, zorder=2)
    ax.text(0.06, m2 + 1.2, f"M2 effective {m2:.1f} GB/s", fontsize=7.8, color=HERO)
    ax.annotate("compile wall:\nproxy can't grow past", xy=(v["proxy_max_workset_M_elems"], 44.4),
                xytext=(0.4, 30), fontsize=7.5, color=SOFT, ha="center",
                arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.8))
    ax.set_xscale("log"); ax.set_ylim(0, 54); ax.set_xlim(0.1, 22)
    ax.set_xlabel("transfer size  (M-elements):  proxy N·M  /  spill K·N")
    ax.set_ylabel("bandwidth  (GB/s)")
    ax.legend(loc="upper left", fontsize=7.6)
    S.save(fig, OUT / "m2_kv_spike")


def main():
    fig_decode_wall()
    fig_pcie_floor()
    fig_bw_specs()
    fig_ramulator2()
    fig_kv_spike()
    figs = sorted(p.name for p in OUT.glob("m2_*.png"))
    print(f"wrote {len(figs)} M2 site figures: {figs}")


if __name__ == "__main__":
    main()
