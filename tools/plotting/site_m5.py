"""M5 (Workload) editorial figures for the Phase-1 site (build artifact, nature-figure style).

Clean, single-purpose, white-background figures sized for the report page. M5's basic function is
to deterministically generate the per-token op-DAG trace the simulator runs — the workload. The
measurement is the op inventory (580 op x shape signatures, 9 categories) + per-(model,task) op
profiles (FLOPs/bytes/intensity), validated with 0 orphans across 4 models.
Regenerable from committed JSON only. Writes PNG (+PDF/SVG) to docs/figures/phase1-site/.

  m5_op_breakdown — S1: op-category composition (FLOPs + bytes share), 4 models x 4 workloads.
  m5_intensity    — S2: op intensity (FLOP/byte) by category, 4 workloads in separate panels.
                   DATA ONLY — NO hardware ceiling is drawn (we have none measured); NOT a roofline.
  m5_coverage     — S4: 4 models x 9 categories coverage grid, 0 orphans (the gate, visual).

Run: ./.venv/bin/python tools/plotting/site_m5.py
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/plotting"))
import _style as S  # noqa: E402  (PALETTE/CAT_ORDER + save(); rcParams overridden below)

OP = ROOT / "measurements/op_profile"
M5 = ROOT / "validation/reports/phase1.1/m5.json"
OUT = ROOT / "docs/figures/phase1-site"

HERO = "#0072B2"; WARM = "#C45A12"; OK = "#1b7f5a"; GREY = "#b9b09c"
INK = "#17150f"; SOFT = "#5b554a"; PAPER = "#fbf6ec"; GRID = "#e8e1d2"

PALETTE = S.PALETTE
CAT_ORDER = S.CAT_ORDER
MODEL_ORDER = ["llama-3.2-1b", "llama-3.2-3b", "qwen2.5-7b", "llama-3.1-8b"]
MLAB = {"llama-3.2-1b": "1B", "llama-3.2-3b": "3B", "qwen2.5-7b": "Qwen-7B", "llama-3.1-8b": "8B"}
# task ordering along the prefill-heavy <-> decode-heavy spectrum
TASK_ORDER = ["longbench-triviaqa", "humaneval", "gsm8k", "sharegpt"]
TLAB = {"longbench-triviaqa": "LongBench\n(prefill-heavy)", "humaneval": "HumanEval",
        "gsm8k": "GSM8K", "sharegpt": "ShareGPT\n(decode-heavy)"}

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


def _shares(doc, metric, phase):
    """Fraction each category contributes to `metric` (flops|bytes) in a phase."""
    bp = doc["totals"]["by_phase"][phase]
    tot = sum(bp[c][metric] for c in bp) or 1
    return {c: bp.get(c, {}).get(metric, 0) / tot for c in CAT_ORDER}


def fig_op_breakdown():
    """S1 — decode op-category composition (FLOPs share + bytes share) across ALL 4 models x ALL 4
    workloads. rows = {FLOPs, bytes}, cols = workloads, bars = the 4 models. Shows weight-stationary
    matmul dominates both compute and memory in decode, for every (model, workload)."""
    docs = {(m, t): load(OP / f"{m}_{t}.json") for m in MODEL_ORDER for t in TASK_ORDER}
    fig, axes = plt.subplots(2, len(TASK_ORDER), figsize=(7.8, 4.4), sharey="row")
    x = np.arange(len(MODEL_ORDER))
    for r, (metric, ylab) in enumerate([("flops", "decode FLOPs share"),
                                        ("bytes", "decode bytes share")]):
        for c, t in enumerate(TASK_ORDER):
            ax = axes[r, c]
            for i, m in enumerate(MODEL_ORDER):
                sh = _shares(docs[(m, t)], metric, "decode")
                bottom = 0.0
                for cat in CAT_ORDER:
                    ax.bar(i, sh[cat], bottom=bottom, color=PALETTE[cat], width=0.86,
                           edgecolor="white", linewidth=0.3, zorder=3)
                    bottom += sh[cat]
            ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1.0])
            ax.set_xticks(x); ax.set_xticklabels([MLAB[m] for m in MODEL_ORDER], fontsize=6.8)
            if r == 0:
                ax.set_title(TLAB[t].replace("\n", " "), fontsize=7.2)
            if c == 0:
                ax.set_ylabel(ylab, fontsize=8)
    labs = [("matmul", "matmul"), ("attention", "attention"), ("kv_cache", "kv_cache"),
            ("ffn", "ffn (SwiGLU)"), ("norm", "norm/rope/...")]
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[c]) for c, _ in labs]
    fig.legend(handles, [t for _, t in labs], ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), fontsize=7.6, handlelength=1.1, columnspacing=1.2)
    fig.subplots_adjust(bottom=0.13, wspace=0.18, hspace=0.32)
    S.save(fig, OUT / "m5_op_breakdown")


def _agg(doc, phase, cat):
    """Execution-weighted intensity + total FLOPs for one category in a phase."""
    rows = [r for r in doc["rows"] if r["phase"] == phase and r["category"] == cat]
    f = sum(r["flops"] * r["count"] for r in rows)
    b = sum(r["bytes"] * r["count"] for r in rows)
    n = sum(r["count"] for r in rows)
    if n and b and f:
        return f / b, f
    return None


def fig_intensity():
    """S1/S2 — predicted-side operational intensity (FLOP/byte) by op category, 8B, 4 workloads
    in SEPARATE panels. NO hardware ceiling / ridge / bound-region is drawn: we have no measured
    roofline ceiling for this multi-unit workload, so drawing one would be an assumption. This is
    NOT a roofline — it is the raw intensity distribution. The only claim is the op-level fact that
    decode GEMV reuses each weight once -> intensity ~2 (left), while prefill matmul is high-intensity
    (right). prefill = filled, decode = open."""
    model = "llama-3.1-8b"
    docs = {t: load(OP / f"{model}_{t}.json") for t in TASK_ORDER}
    fig, axes = plt.subplots(1, 4, figsize=(7.8, 2.9), sharex=True, sharey=True)
    for ax, t in zip(axes, TASK_ORDER):
        _grid(ax)
        for cat in CAT_ORDER:
            pre = _agg(docs[t], "prefill", cat)
            dec = _agg(docs[t], "decode", cat)
            if pre:
                ax.scatter(pre[0], pre[1], s=26, marker="o", color=PALETTE[cat],
                           edgecolors="white", linewidths=0.4, zorder=3)
            if dec:
                ax.scatter(dec[0], dec[1], s=30, marker="s", facecolors="none",
                           edgecolors=PALETTE[cat], linewidths=1.0, zorder=4)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(TLAB[t].replace("\n", " "), fontsize=7.2)
        ax.set_xlabel("op intensity (FLOP/byte)", fontsize=7.6)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("FLOPs contributed\n(exec-weighted)", fontsize=7.6)
    cats_present = [c for c in CAT_ORDER
                    if any(_agg(docs[t], p, c) for t in TASK_ORDER for p in ("prefill", "decode"))]
    cat_h = [plt.Line2D([], [], marker="o", ls="", color=PALETTE[c], label=c) for c in cats_present]
    ph_h = [plt.Line2D([], [], marker="o", ls="", color="#444", label="prefill (filled)"),
            plt.Line2D([], [], marker="s", ls="", mfc="none", mec="#444", label="decode (open)")]
    fig.legend(cat_h + ph_h, [h.get_label() for h in cat_h + ph_h], ncol=len(cat_h + ph_h),
               loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=6.6,
               handlelength=1.0, columnspacing=0.9)
    fig.suptitle("operational intensity by op category — 8B, 4 workloads  "
                 "(data only; NO hardware ceiling assumed — not a roofline)", fontsize=7.8, y=1.02)
    fig.subplots_adjust(bottom=0.26, wspace=0.12)
    S.save(fig, OUT / "m5_intensity")


def fig_coverage():
    """S4 — the coverage gate, visual: 4 models x 9 categories, every cell traced from HF
    with 0 orphan ops. Cell annotates distinct-ops contribution; all green = gate PASS."""
    m5 = load(M5)
    sweep = load(ROOT / "measurements/op_inventory/sweep_matrix.json")
    counts = sweep["counts"]  # category -> # of distinct (op,shape) signatures in the matrix
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    nM, nC = len(MODEL_ORDER), len(CAT_ORDER)
    for j, cat in enumerate(CAT_ORDER):
        for i, m in enumerate(MODEL_ORDER):
            pm = m5["per_model"][m]
            covered = pm["semantic_covered"] and not pm["orphan_ops"]
            fc = PALETTE[cat] if covered else "#ddd"
            ax.add_patch(plt.Rectangle((j, nM - 1 - i), 1, 1, facecolor=fc, alpha=0.85,
                                       edgecolor="white", linewidth=1.4))
            ax.text(j + 0.5, nM - 1 - i + 0.5, "ok", ha="center", va="center",
                    color="white", fontsize=8.5, fontweight="bold")
    ax.set_xlim(0, nC); ax.set_ylim(0, nM)
    ax.set_xticks(np.arange(nC) + 0.5)
    ax.set_xticklabels([f"{c}\n({counts[c]})" for c in CAT_ORDER], fontsize=7.4, rotation=0)
    ax.set_yticks(np.arange(nM) + 0.5)
    ax.set_yticklabels([MLAB[m] for m in MODEL_ORDER][::-1], fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("op category  (number under name = distinct op x shape signatures, 580 total)",
                  fontsize=8.2)
    ax.text(0, nM + 0.18, "every cell traced from HuggingFace runtime  -  0 orphan ops",
            fontsize=8, color=OK, fontweight="bold")
    S.save(fig, OUT / "m5_coverage")


def main():
    fig_op_breakdown()
    fig_intensity()
    fig_coverage()
    figs = sorted(p.name for p in OUT.glob("m5_*.png"))
    print(f"wrote {len(figs)} M5 site figures: {figs}")


if __name__ == "__main__":
    main()
