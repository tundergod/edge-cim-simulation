"""M1 figure (Phase 1.2 memory) — the 3 host-DRAM specs vs the 24.2 GB/s LPDDR4x anchor.

Reads validation/reports/phase1.2/m2_memory.json (committed). Bars = each spec's effective
bandwidth, coloured by honesty tag (LPDDR4x = measured/calibrated anchor; LPDDR4 = assumption;
LPDDR5 = simulated). The dashed line is the 24.2 GB/s measured anchor. Writes M1.png/.pdf/.svg
to docs/figures/phase1.2/.

Run: ./.venv/bin/python tools/plotting/mem_m1.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _style import save  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "validation/reports/phase1.2/m2_memory.json"
FIG = ROOT / "docs/figures/phase1.2"
ORDER = ["mem_lpddr4", "mem_lpddr4x", "mem_lpddr5"]
# colour by honesty tag (Okabe-Ito subset): calibrated=hero blue, assumption=grey, simulated=orange
TAGCOLOR = {"calibrated": "#0072B2", "assumption": "#999999", "simulated": "#D55E00"}


def main():
    rep = json.loads(REPORT.read_text())
    rows = rep["memory_specs"]
    anchor = rep["anchor"]["eff_BW_GBs"]
    labels = [rows[n]["memory_type"] for n in ORDER]
    eff = [rows[n]["eff_BW_GBs"] for n in ORDER]
    peak = [rows[n]["peak_GBs"] for n in ORDER]
    kinds = [rows[n]["eff_BW_tag"].split(" ")[0] for n in ORDER]   # calibrated/assumption/simulated
    colors = [TAGCOLOR[k] for k in kinds]

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    x = range(len(ORDER))
    ax.bar(x, peak, color="0.88", width=0.62, label="peak (assumption)", zorder=1)
    ax.bar(x, eff, color=colors, width=0.62, zorder=2)
    ax.axhline(anchor, color="0.35", ls="--", lw=0.8, zorder=3)
    ax.text(len(ORDER) - 0.45, anchor + 0.6, f"measured anchor {anchor} GB/s (LPDDR4x)",
            fontsize=5.6, color="0.35", ha="right")
    for xi, (e, p) in enumerate(zip(eff, peak)):
        ax.text(xi, e + 0.6, f"{e}", ha="center", fontsize=6.5)
        ax.text(xi, p + 0.6, f"{p:g}", ha="center", fontsize=5.5, color="0.5")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("bandwidth (GB/s)")
    ax.set_ylim(0, max(peak) * 1.18)
    handles = [plt.Rectangle((0, 0), 1, 1, color=TAGCOLOR[k]) for k in ("calibrated", "assumption", "simulated")]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="0.88"))
    ax.legend(handles, ["calibrated (measured)", "assumption", "simulated", "peak (assumption)"],
              fontsize=5.8, loc="upper left", ncol=1)
    ax.set_title("M2 host-DRAM effective bandwidth — 3 swappable specs\n"
                 "(LPDDR4x = measured 24.2 anchor; LPDDR5 = sim 0.65 discount)", fontsize=7.5)
    fig.tight_layout()
    save(fig, str(FIG / "M1"))
    print(f"wrote {FIG / 'M1'}.png")


if __name__ == "__main__":
    main()
