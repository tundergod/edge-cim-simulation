"""Phase 0.2 Fig 2 — predicted-side operational intensity / roofline placement.

Claim: decode is overwhelmingly memory-bound (low operational intensity) — the
regime where CIM weight-residency helps most — while prefill matmul/attention is
compute-bound. The ridge shown is illustrative; the measured roofline knee is
Phase 0.3/Phase 1 (this is the predicted-side input).

Reads measurements/op_profile/{model}_{task}.json only. Writes docs/figures/phase0.2/.
Run: ./.venv/bin/python tools/plotting/roofline.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _style import PALETTE, CAT_ORDER, TASK_ORDER, TASK_LABEL, MODEL_ORDER, MODEL_LABEL, save

OP = Path("measurements/op_profile")
FIG = Path("docs/figures/phase0.2")
RIDGE = 20.0  # illustrative compute/memory ridge (FLOP/byte); measured knee = Phase 0.3


def agg(doc, phase):
    """Per-category aggregate: weighted intensity, total FLOPs, total count."""
    out = {}
    for c in CAT_ORDER:
        rows = [r for r in doc["rows"] if r["phase"] == phase and r["category"] == c]
        f = sum(r["flops"] * r["count"] for r in rows)
        b = sum(r["bytes"] * r["count"] for r in rows)
        n = sum(r["count"] for r in rows)
        if n and b:
            out[c] = {"intensity": f / b, "flops": f, "count": n}
    return out


def fig_roofline(model="llama-3.1-8b"):
    docs = {t: json.loads((OP / f"{model}_{t}.json").read_text()) for t in TASK_ORDER}
    fig, axes = plt.subplots(1, len(TASK_ORDER), figsize=(7.4, 2.5), sharex=True, sharey=True)
    for ax, t in zip(axes, TASK_ORDER):
        ax.axvspan(1e-2, RIDGE, color="0.93", zorder=0)  # memory-bound region
        ax.axvline(RIDGE, color="0.5", ls="--", lw=0.7, zorder=1)
        for c, v in agg(docs[t], "prefill").items():  # prefill = filled circle
            ax.scatter(v["intensity"], v["flops"], s=20, marker="o", color=PALETTE[c],
                       edgecolor="white", linewidth=0.3, zorder=3)
        for c, v in agg(docs[t], "decode").items():   # decode = open square
            ax.scatter(v["intensity"], v["flops"], s=20, marker="s", facecolor="none",
                       edgecolor=PALETTE[c], linewidth=0.9, zorder=4)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(TASK_LABEL[t], fontsize=6.5)
        ax.set_xlabel("op intensity (FLOP/byte)", fontsize=6.5)
        ax.tick_params(length=2, labelsize=6)
    axes[0].set_ylabel("FLOPs contributed", fontsize=7)
    axes[0].text(RIDGE * 1.2, axes[0].get_ylim()[1] * 0.3, "compute-bound →", fontsize=5.5, color="0.4")
    axes[0].text(0.02, axes[0].get_ylim()[1] * 0.3, "← memory-bound", fontsize=5.5, color="0.4")
    cat_handles = [plt.Line2D([], [], marker="o", ls="", color=PALETTE[c], label=c) for c in CAT_ORDER]
    ph_handles = [plt.Line2D([], [], marker="o", ls="", color="0.3", label="prefill (filled)"),
                  plt.Line2D([], [], marker="s", ls="", mfc="none", mec="0.3", label="decode (open)")]
    fig.legend(cat_handles + ph_handles, [h.get_label() for h in cat_handles + ph_handles],
               ncol=11, loc="lower center", bbox_to_anchor=(0.5, -0.12), fontsize=5.8,
               handlelength=1.0, columnspacing=0.8)
    fig.suptitle(f"Predicted-side operational intensity — {MODEL_LABEL[model]} "
                 f"(decode collapses to memory-bound; ridge illustrative)", fontsize=8, y=1.03)
    fig.tight_layout()
    save(fig, str(FIG / f"fig2_roofline_{model}"))


def fig_intensity_shift():
    """matmul operational intensity, prefill vs decode, across tasks x models —
    the prefill(compute-bound) -> decode(memory-bound GEMV) collapse."""
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    x = np.arange(len(TASK_ORDER))
    width = 0.2
    for j, m in enumerate(MODEL_ORDER):
        docs = {t: json.loads((OP / f"{m}_{t}.json").read_text()) for t in TASK_ORDER}
        pre = [agg(docs[t], "prefill").get("matmul", {}).get("intensity", np.nan) for t in TASK_ORDER]
        dec = [agg(docs[t], "decode").get("matmul", {}).get("intensity", np.nan) for t in TASK_ORDER]
        ax.scatter(x + (j - 1.5) * width, pre, s=22, marker="o", color="#0072B2",
                   edgecolor="white", linewidth=0.3, zorder=3, label="prefill matmul" if j == 0 else None)
        ax.scatter(x + (j - 1.5) * width, dec, s=22, marker="s", facecolor="none",
                   edgecolor="#0072B2", linewidth=0.9, zorder=3, label="decode matmul" if j == 0 else None)
    ax.axhline(RIDGE, color="0.5", ls="--", lw=0.7)
    ax.text(len(TASK_ORDER) - 0.5, RIDGE * 1.3, "ridge (illustrative)", fontsize=5.5, color="0.4", ha="right")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([TASK_LABEL[t] for t in TASK_ORDER], fontsize=6.5)
    ax.set_ylabel("matmul op intensity (FLOP/byte)")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title("matmul: prefill (compute-bound, high intensity) collapses to decode GEMV "
                 "(memory-bound, ~1) — 4 models", fontsize=8)
    fig.tight_layout()
    save(fig, str(FIG / "fig2b_intensity_shift"))


def main():
    fig_roofline("llama-3.1-8b")
    fig_intensity_shift()
    print(f"wrote roofline figures to {FIG}/")


if __name__ == "__main__":
    main()
