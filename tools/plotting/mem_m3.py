"""M3 figure (Phase 1.2 memory) — the SRAM tier residency what-if.

Reads validation/reports/phase1.2/m2_memory.json (committed). Shows that a working set
streamed against the Metis SRAM tier resolves to SRAM (CACTI BW, assumption) only while it
fits the 32 MiB L2; an 8B INT8 weight set (~8 GB >> 32 MiB) is NEVER resident and spills to
the DRAM tier (calibrated LPDDR4x 24.2 anchor) — residency='architecture-only'. Per-GB
stream time vs working-set size, with the L2 capacity wall marked. Writes M3.png/.pdf/.svg.

Run: ./.venv/bin/python tools/plotting/mem_m3.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _style import save  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "validation/reports/phase1.2/m2_memory.json"
FIG = ROOT / "docs/figures/phase1.2"
_MiB = 1024 * 1024
SRAM_C = "#009E73"   # resident (SRAM tier, CACTI assumption)
DRAM_C = "#0072B2"   # spilled to DRAM tier (calibrated anchor)


def main():
    rep = json.loads(REPORT.read_text())
    sw = rep["sram_what_if"]
    l2_bytes = sw["l2_MiB_shared"] * _MiB
    sram_bw, dram_bw = sw["bw_GBs"], rep["memory_specs"]["mem_lpddr4x"]["eff_BW_GBs"]

    # stream time per GB vs working-set size: SRAM BW below the L2 wall, DRAM BW above it
    sizes = np.logspace(np.log10(64 * 1024), np.log10(8e9), 200)   # 64 KiB .. 8 GB
    per_gb = np.where(sizes <= l2_bytes, 1e9 / (sram_bw * 1e9) * 1e6, 1e9 / (dram_bw * 1e9) * 1e6)

    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    resident = sizes <= l2_bytes
    ax.plot(sizes[resident] / _MiB, per_gb[resident], color=SRAM_C, lw=1.8,
            label=f"SRAM tier resident ({sram_bw:g} GB/s, CACTI assumption)")
    ax.plot(sizes[~resident] / _MiB, per_gb[~resident], color=DRAM_C, lw=1.8,
            label=f"spilled to DRAM ({dram_bw} GB/s, calibrated anchor)")
    ax.axvline(sw["l2_MiB_shared"], color="0.4", ls="--", lw=0.8)
    ax.text(sw["l2_MiB_shared"] * 0.92, ax.get_ylim()[1] * 0.5,
            f"L2 {sw['l2_MiB_shared']} MiB wall", rotation=90, va="center", ha="right",
            fontsize=5.8, color="0.4")
    # mark the 8B weight set (always on the DRAM branch)
    w8b_MiB = 8e9 / _MiB
    ax.scatter([w8b_MiB], [1e9 / (dram_bw * 1e9) * 1e6], s=28, color=DRAM_C, zorder=5,
               edgecolor="white", linewidth=0.4)
    ax.annotate("8B weights (~8 GB)\nnever resident", (w8b_MiB, 1e9 / (dram_bw * 1e9) * 1e6),
                textcoords="offset points", xytext=(0, -28), fontsize=5.8, ha="right", color=DRAM_C)
    ax.set_xscale("log")
    ax.set_ylim(0, per_gb.max() * 1.12)
    ax.set_xlabel("working-set size (MiB, log)")
    ax.set_ylabel("stream time per GB (us)")
    ax.legend(fontsize=5.8, loc="upper left")
    ax.set_title("M1-SPM SRAM residency what-if (architecture-only)\n"
                 "weights spill to DRAM above the 32 MiB L2 wall", fontsize=7.5)
    fig.tight_layout()
    save(fig, str(FIG / "M3"))
    print(f"wrote {FIG / 'M3'}.png")


if __name__ == "__main__":
    main()
