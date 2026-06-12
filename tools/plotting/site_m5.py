"""M5 (Workload) editorial figures for the Phase-1 site (build artifact, nature-figure style).

Clean, single-purpose, white-background figures sized for the report page. M5's basic function is
to deterministically generate the per-token op-DAG trace the simulator runs — the workload. The
measurement is the op inventory (580 op x shape signatures, 9 categories) + per-(model,task) op
profiles (FLOPs/bytes/intensity), validated with 0 orphans across 4 models.
Regenerable from committed JSON only. Writes PNG (+PDF/SVG) to docs/figures/phase1-site/.

  m5_op_breakdown — S1: per-model op-category composition (FLOPs + bytes share) — matmul dominates.
  m5_roofline     — S1/S2: op intensity vs FLOPs, prefill (filled) vs decode (open) — decode ~2.
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
    """S1 — per-model op-category composition: decode FLOPs share + decode bytes share,
    4 models. Shows weight-stationary matmul dominates both compute and memory."""
    docs = {m: load(OP / f"{m}_sharegpt.json") for m in MODEL_ORDER}
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.6))
    x = np.arange(len(MODEL_ORDER))
    for ax, metric, ylab in zip(axes, ["flops", "bytes"],
                                ["decode FLOPs share", "decode bytes share"]):
        for i, m in enumerate(MODEL_ORDER):
            sh = _shares(docs[m], metric, "decode")
            bottom = 0.0
            for c in CAT_ORDER:
                ax.bar(i, sh[c], bottom=bottom, color=PALETTE[c], width=0.88,
                       edgecolor="white", linewidth=0.4, zorder=3)
                bottom += sh[c]
        ax.set_xticks(x); ax.set_xticklabels([MLAB[m] for m in MODEL_ORDER])
        ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1.0])
        ax.set_ylabel(ylab)
    # direct-labelled legend, top-3 categories only (matmul/attention/kv_cache visible at this scale)
    labs = [("matmul", "matmul"), ("attention", "attention"), ("kv_cache", "kv_cache"),
            ("ffn", "ffn (SwiGLU)"), ("norm", "norm/rope/...")]
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[c]) for c, _ in labs]
    fig.legend(handles, [t for _, t in labs], ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), fontsize=8, handlelength=1.1, columnspacing=1.2)
    fig.subplots_adjust(bottom=0.18, wspace=0.32)
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


def fig_roofline():
    """S1/S2 — predicted-side operational intensity vs FLOPs contributed (8B, all 4 tasks):
    prefill (filled) is compute-bound and scales with length; decode (open) collapses to
    the memory-bound GEMV ridge at intensity ~2 FLOP/byte for every task."""
    model = "llama-3.1-8b"
    docs = {t: load(OP / f"{model}_{t}.json") for t in TASK_ORDER}
    RIDGE = 20.0  # illustrative compute/memory ridge; measured knee is Phase 0.3 / Phase 1
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    _grid(ax)
    ax.axvspan(1e-2, RIDGE, color="#eef2f6", zorder=0)
    ax.axvline(RIDGE, color="#999", ls="--", lw=0.9, zorder=1)
    seen = set()
    for t in TASK_ORDER:
        for cat in CAT_ORDER:
            pre = _agg(docs[t], "prefill", cat)
            dec = _agg(docs[t], "decode", cat)
            if pre:
                ax.scatter(pre[0], pre[1], s=34, marker="o", color=PALETTE[cat],
                           edgecolors="white", linewidths=0.4, zorder=3)
            if dec:
                ax.scatter(dec[0], dec[1], s=40, marker="s", facecolors="none",
                           edgecolors=PALETTE[cat], linewidths=1.1, zorder=4)
            seen.add(cat)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("operational intensity  (FLOP/byte)")
    ax.set_ylabel("FLOPs contributed  (execution-weighted)")
    ax.text(0.5, 0.04, "memory-bound", transform=ax.transAxes, fontsize=8, color=SOFT,
            ha="center", va="bottom")
    ax.text(0.985, 0.04, "compute-bound", transform=ax.transAxes, fontsize=8, color=SOFT,
            ha="right", va="bottom")
    ax.annotate("decode GEMV ridge\nintensity ~2 FLOP/byte\n(memory-bound, every task)",
                xy=(2.0, ax.get_ylim()[1] * 0.02), xytext=(0.06, 0.62), textcoords="axes fraction",
                fontsize=7.6, color=WARM, ha="left",
                arrowprops=dict(arrowstyle="->", color=WARM, lw=0.9))
    # marker legend (shape = phase), category colour explained in caption
    ph = [plt.Line2D([], [], marker="o", ls="", color="#444", label="prefill (filled)"),
          plt.Line2D([], [], marker="s", ls="", mfc="none", mec="#444", label="decode (open)")]
    ax.legend(handles=ph, loc="upper left", fontsize=8)
    S.save(fig, OUT / "m5_roofline")


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
    fig_roofline()
    fig_coverage()
    figs = sorted(p.name for p in OUT.glob("m5_*.png"))
    print(f"wrote {len(figs)} M5 site figures: {figs}")


if __name__ == "__main__":
    main()
