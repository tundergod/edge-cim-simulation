"""Phase 0.2 Fig 1 — op-category breakdown across the workload spectrum.

Claim: weight-stationary matmul dominates both compute (FLOPs) and memory (bytes)
in every workload; attention's share rises only in long-context prefill. This is
the structural reason a CIM-centric design (which excels at weight-stationary
GEMM/GEMV) is well-matched to LLM inference, with attention offloaded.

Reads measurements/op_profile/{model}_{task}.json only. Writes docs/figures/phase0.2/.
Run: ./.venv/bin/python tools/plotting/op_breakdown.py
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


def shares(doc, metric, phase):
    """Fraction each category contributes to `metric` (flops|bytes|count) in a phase."""
    bp = doc["totals"]["by_phase"][phase]
    tot = sum(bp[c][metric] for c in bp) or 1
    return {c: bp.get(c, {}).get(metric, 0) / tot for c in CAT_ORDER}


def stacked(ax, doc, metric):
    """Two stacked bars (prefill, decode) of category shares for one (model,task)."""
    for i, phase in enumerate(["prefill", "decode"]):
        sh = shares(doc, metric, phase)
        bottom = 0.0
        for c in CAT_ORDER:
            ax.bar(i, sh[c], bottom=bottom, color=PALETTE[c], width=0.72,
                   edgecolor="white", linewidth=0.3)
            bottom += sh[c]
    ax.set_xticks([0, 1]); ax.set_xticklabels(["prefill", "decode"])
    ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(length=2)


def fig_breakdown(model):
    docs = {t: json.loads((OP / f"{model}_{t}.json").read_text()) for t in TASK_ORDER}
    metrics = [("flops", "FLOPs share"), ("bytes", "Memory (bytes) share")]
    fig, axes = plt.subplots(len(metrics), len(TASK_ORDER), figsize=(7.2, 3.6), sharey=True)
    for r, (metric, mlabel) in enumerate(metrics):
        for c, t in enumerate(TASK_ORDER):
            ax = axes[r, c]
            stacked(ax, docs[t], metric)
            if r == 0:
                P, D = docs[t]["prefill_len"], docs[t]["decode_len"]
                ax.set_title(f"{TASK_LABEL[t]}\nP={P}, D={D}", fontsize=6.5)
            if c == 0:
                ax.set_ylabel(mlabel, fontsize=7)
    # shared legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[c]) for c in CAT_ORDER]
    fig.legend(handles, CAT_ORDER, ncol=9, loc="lower center", bbox_to_anchor=(0.5, -0.04),
               fontsize=6.2, handlelength=1.0, columnspacing=1.0)
    fig.suptitle(f"Op-category composition — {MODEL_LABEL[model]} "
                 f"(matmul dominates compute & memory; attention rises in long-context prefill)",
                 fontsize=8, y=1.02)
    fig.tight_layout()
    save(fig, str(FIG / f"fig1_op_breakdown_{model}"))


def fig_model_scaling(task="sharegpt"):
    """How decode FLOPs/bytes per token split across model sizes (fixed task)."""
    docs = {m: json.loads((OP / f"{m}_{task}.json").read_text()) for m in MODEL_ORDER}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, metric, ylab in zip(axes, ["flops", "bytes"], ["decode FLOPs share", "decode bytes share"]):
        for i, m in enumerate(MODEL_ORDER):
            sh = shares(docs[m], metric, "decode")
            bottom = 0.0
            for cc in CAT_ORDER:
                ax.bar(i, sh[cc], bottom=bottom, color=PALETTE[cc], width=0.7,
                       edgecolor="white", linewidth=0.3)
                bottom += sh[cc]
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels([MODEL_LABEL[m].replace("Llama-3.2-", "").replace("Llama-3.1-", "")
                            .replace("Qwen2.5-", "Qwen-") for m in MODEL_ORDER], fontsize=6.2)
        ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1.0]); ax.set_ylabel(ylab); ax.tick_params(length=2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[c]) for c in CAT_ORDER]
    fig.legend(handles, CAT_ORDER, ncol=9, loc="lower center", bbox_to_anchor=(0.5, -0.08),
               fontsize=6.2, handlelength=1.0, columnspacing=1.0)
    fig.suptitle(f"Decode op-mix vs model size — {task} (matmul share grows with model)", fontsize=8, y=1.02)
    fig.tight_layout()
    save(fig, str(FIG / f"fig1b_model_scaling_{task}"))


def main():
    for m in MODEL_ORDER:
        fig_breakdown(m)
    fig_model_scaling("sharegpt")
    print(f"wrote op-breakdown figures to {FIG}/")


if __name__ == "__main__":
    main()
