"""Figure M2-ramulator2 — Ramulator2 v2.1 LPDDR5 (device) vs Phase-1.2 analytic (system) single-stream BW.

Build artifact: reads validation/reports/phase1.3/m2_ramulator2.json, emits docs/figures/phase1.3/
M2-ramulator2.{png,pdf,svg}. Shows that Ramulator2 (DRAM device timing) reaches ~0.92 of peak on a
single stream, while the analytic 0.65 is the system-level wall (silicon-calibrated) — the gap is
system overhead Ramulator2's device model omits, not a contradiction.

Run: ./.venv/bin/python tools/plotting/mem_ramulator2_fig.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import save  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "validation/reports/phase1.3/m2_ramulator2.json"
FIG = ROOT / "docs/figures/phase1.3"


def main():
    r = json.loads(REPORT.read_text())
    peak = r["ramulator2_device"]["peak_GBs_single_channel"] * 4   # 4 x12.8 = 51.2 GB/s 64-bit peak
    ram_bw, ram_eff = r["ramulator2_device"]["eff_BW_GBs"], r["ramulator2_device"]["efficiency"]
    ana_bw, ana_eff = r["analytic_system"]["eff_BW_GBs"], r["analytic_system"]["efficiency"]

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    bars = [("LPDDR5-6400\npeak (51.2)", peak, "#bbbbbb", None),
            ("Ramulator2 v2.1\n(DRAM device)", ram_bw, "#0072B2", ram_eff),
            ("Analytic\n(system, calibrated)", ana_bw, "#C45A12", ana_eff)]
    xs = range(len(bars))
    ax.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], width=0.62, edgecolor="#333", linewidth=0.6)
    for x, (lbl, v, _, eff) in zip(xs, bars):
        tag = f"{v:.1f} GB/s" + (f"\n({eff:.0%} of peak)" if eff else "\n(peak)")
        ax.text(x, v + 0.8, tag, ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(xs)); ax.set_xticklabels([b[0] for b in bars], fontsize=9)
    ax.set_ylabel("single-stream eff. BW (GB/s)")
    ax.set_ylim(0, peak * 1.18)
    ax.set_title("LPDDR5-6400 single-stream: device (Ramulator2) vs system (analytic)", fontsize=10)
    ax.annotate("gap = system overhead\n(controller/NoC/queueing)\ncalibrated from silicon,\nnot in the device model",
                xy=(2, ana_bw), xytext=(1.55, peak * 0.5), fontsize=7.5, color="#555",
                arrowprops=dict(arrowstyle="->", color="#999", lw=0.8))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    save(fig, str(FIG / "M2-ramulator2"))
    print(f"wrote {FIG/'M2-ramulator2'}.png (+pdf/svg)")


if __name__ == "__main__":
    main()
