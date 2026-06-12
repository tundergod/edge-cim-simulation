"""M8 (Thermal) editorial figures for the Phase-1 site (build artifact, nature-figure style).

Regenerable from committed JSON: the on-board heat campaign (measurements/metis_card/thermal_heat_*.json)
+ the RC/perf fit (validation/reports/phase1.7/thermal.json). Two single-purpose figures:
  m8_heating    — §1/§2: max-core temp vs time (sustained 4-core matmul) + RC fit + plateau; headroom note.
  m8_perf_temp  — §3: dev throughput vs max-core temp + noise band -> flat, no throttle in range.

Run: ./.venv/bin/python tools/plotting/site_m8.py
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
sys.path.insert(0, str(ROOT / "tools/plotting"))
import _style as S  # noqa: E402

HEAT = ROOT / "measurements/metis_card/thermal_heat_20260612.json"
FIT = ROOT / "validation/reports/phase1.7/thermal.json"
OUT = ROOT / "docs/figures/phase1-site"
HERO = "#0072B2"; WARM = "#C45A12"; OK = "#1b7f5a"; INK = "#17150f"; SOFT = "#5b554a"
PAPER = "#fbf6ec"; GRID = "#e8e1d2"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11.5,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.9,
    "axes.edgecolor": "#888", "xtick.color": "#555", "ytick.color": "#555",
    "axes.labelcolor": INK, "text.color": INK, "legend.frameon": False, "figure.dpi": 150,
})


def load(p):
    return json.loads(Path(p).read_text())


def _grid(ax):
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)


def fig_heating(heat, fit):
    b = [x for x in heat["bursts"] if x.get("temp_C")]
    t = np.array([x["t_elapsed"] for x in b]); T = np.array([x["temp_C"] for x in b])
    rc = fit["rc_fit"]; T0, Tinf, tau = rc["T0_C"], rc["Tinf_C"], rc["tau_s"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    _grid(ax)
    ax.scatter(t, T, s=26, color=HERO, edgecolors="white", linewidths=0.5, zorder=4,
               label="max core temp (per burst)")
    xs = np.linspace(0, t.max(), 200)
    ax.plot(xs, Tinf - (Tinf - T0) * np.exp(-xs / tau), "-", color=INK, lw=1.6, zorder=3,
            label=f"RC fit  T_inf={Tinf}C  tau={tau:.0f}s")
    ax.axhline(Tinf, ls=":", color=WARM, lw=1.0)
    ax.text(t.max() * 0.5, Tinf + 0.15, f"plateau ~{Tinf:.0f}C (full 4-core load)", color=WARM,
            fontsize=9.5, ha="center", va="bottom")
    ax.set_ylim(T0 - 1.5, Tinf + 2)
    ax.set_xlabel("sustained-load time  (s)")
    ax.set_ylabel("max core temperature  (C)")
    head = fit["headroom_to_downscale_C"]
    ax.text(0.97, 0.06, f"freq-downscale threshold 110C\n= +{head:.0f}C headroom (never approached)",
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom", color=SOFT)
    ax.legend(loc="lower right", fontsize=9.5, bbox_to_anchor=(1.0, 0.18))
    S.save(fig, OUT / "m8_heating")


def fig_perf_temp(heat, fit):
    b = [x for x in heat["bursts"] if x.get("temp_C") and x.get("dev_fps")]
    T = np.array([x["temp_C"] for x in b]); F = np.array([x["dev_fps"] for x in b])
    pv = fit["perf_vs_temp"]; mean, noise = pv["fps_mean"], pv["noise_std_fps"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    _grid(ax)
    ax.axhspan(mean - 2 * noise, mean + 2 * noise, color=HERO, alpha=0.10, zorder=1,
               label=f"+/-2 sigma noise band ({noise:.0f} fps)")
    # jitter identical temps horizontally so the vertical spread at each temp is visible
    rng = np.linspace(-0.18, 0.18, 0)
    ax.scatter(T, F, s=30, color=HERO, edgecolors="white", linewidths=0.4, zorder=4,
               label="dev throughput (per burst)")
    ax.axhline(mean, color=INK, lw=1.0, zorder=3)
    ax.set_xlim(T.min() - 0.6, T.max() + 0.6)
    ax.set_xlabel("max core temperature  (C)")
    ax.set_ylabel("dev throughput  (fps)")
    ax.text(0.04, 0.95,
            f"slope {pv['slope_fps_per_C']:+.1f} fps/C  <<  noise {noise:.0f} fps\n"
            f"=> flat: no throttle in {int(T.min())}-{int(T.max())}C (CoV {pv['fps_cov_pct']:.2f}%)",
            transform=ax.transAxes, fontsize=9.5, va="top", color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc=PAPER, ec=GRID, lw=0.8))
    ax.legend(loc="lower right", fontsize=9)
    S.save(fig, OUT / "m8_perf_temp")


def fig_load_sweep(fit):
    sw = fit["load_sweep"]["per_M"]
    M = [str(s["M"]) for s in sw]; Tp = [s["plateau_C"] for s in sw]
    busy = [s["device_busy_pct"] for s in sw]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    _grid(ax)
    x = np.arange(len(M))
    bars = ax.bar(x, Tp, width=0.5, color=HERO, zorder=3)
    for xi, tp, bz in zip(x, Tp, busy):
        ax.text(xi, tp - 2.5, f"{tp}C", ha="center", fontsize=11, fontweight="bold", color="white")
        ax.text(xi, 2, ("device busy\n" + (f"{bz}%" if bz else "n/a")), ha="center", fontsize=8.5,
                color="white", va="bottom")
    ax.axhline(fit["load_sweep"]["ceiling_C"], ls="--", color=WARM, lw=1.1, zorder=4)
    ax.text(0.02, fit["load_sweep"]["ceiling_C"] + 0.6,
            f"ceiling ~{fit['load_sweep']['ceiling_C']}C", color=WARM, ha="left", fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels([f"M={m}" for m in M])
    ax.set_ylabel("plateau temperature  (C)")
    ax.set_ylim(0, 56)
    ax.text(0.5, 0.97, "heavier load is NOT hotter: host-feed bound (device <=48% busy),\n"
            "single-context only -> power capped -> 44C ceiling",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=SOFT)
    S.save(fig, OUT / "m8_load_sweep")


def main():
    heat, fit = load(HEAT), load(FIT)
    fig_heating(heat, fit)
    fig_perf_temp(heat, fit)
    fig_load_sweep(fit)
    print("wrote m8_heating, m8_perf_temp, m8_load_sweep")


if __name__ == "__main__":
    main()
