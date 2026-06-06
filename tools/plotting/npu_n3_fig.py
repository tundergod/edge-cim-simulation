"""Figure N3 — ONNXim (RKNPU2-approx) vs Phase-1.2 analytic systolic-roofline, NPU GEMM.

Build artifact: reads validation/reports/phase1.3/m4_npu_onnxim.json, emits docs/figures/phase1.3/
N3.{png,pdf,svg}. Left: the K=2048 channel staircase (M=1) — ONNXim vs analytic, both showing the
32-aligned staircase (HeteroInfer Fig3 shape); both SIMULATED (ONNXim dashed, analytic solid), NOT
silicon. Right: per-projection-shape delta. ONNXim != issue #13.

Run: ./.venv/bin/python tools/plotting/npu_n3_fig.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import save  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "validation/reports/phase1.3/m4_npu_onnxim.json"
FIG = ROOT / "docs/figures/phase1.3"


def main():
    rep = json.loads(REPORT.read_text())
    rows = rep["per_shape"]
    stair = sorted([r for r in rows if r["shape"][0] == 1 and r["shape"][1] == 2048],
                   key=lambda r: r["shape"][2])
    proj = [r for r in rows if not (r["shape"][0] == 1 and r["shape"][1] == 2048)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.7))

    # Left: staircase ONNXim vs analytic
    ns = [r["shape"][2] for r in stair]
    ax1.plot(ns, [r["analytic_us"] for r in stair], "-o", color="#0072B2", ms=4, lw=1.4,
             label="analytic roofline (sim)")
    ax1.plot(ns, [r["onnxim_us"] for r in stair], "--s", color="#C45A12", ms=4, lw=1.4,
             label="ONNXim RKNPU2-approx (sim)")
    for x in (32, 64, 128, 256):
        ax1.axvline(x, color="#ddd", lw=0.6, zorder=0)
    ax1.set_xlabel("N (output channels, K=2048, M=1)"); ax1.set_ylabel("latency (µs)")
    ax1.set_title("Channel staircase (HeteroInfer Fig3 shape)", fontsize=9.5)
    ax1.legend(fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # Right: per-projection delta (ONNXim vs analytic)
    labels = [f"{r['shape'][0]}×{r['shape'][1]}×{r['shape'][2]}" for r in proj]
    deltas = [r["delta_pct"] for r in proj]
    ys = range(len(proj))
    ax2.barh(list(ys), deltas, color=["#1b7f5a" if d >= 0 else "#C45A12" for d in deltas], height=0.6)
    ax2.set_yticks(list(ys)); ax2.set_yticklabels(labels, fontsize=6.5)
    ax2.axvline(0, color="#333", lw=0.8)
    ax2.set_xlabel("ONNXim − analytic (%)")
    ax2.set_title("Per-projection delta (sim vs sim)", fontsize=9.5)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.suptitle("N3 · ONNXim vs analytic NPU — both simulated, NOT RKNPU2 silicon (ONNXim ≠ #13)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG.mkdir(parents=True, exist_ok=True)
    save(fig, str(FIG / "N3"))
    print(f"wrote {FIG/'N3'}.png (+pdf/svg)")


if __name__ == "__main__":
    main()
