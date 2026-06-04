"""Shared publication figure style (nature-figure skill: Python backend).

Restrained, colorblind-safe (Okabe-Ito) palette; editable text in vector exports;
600-dpi raster. All Phase 0.2/0.3 figures import this so the look is consistent and
every figure is a build artifact regenerable from committed JSON.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "figure.dpi": 150,
})

# Okabe-Ito colorblind-safe palette, one per op category (matmul = hero blue)
PALETTE = {
    "matmul":    "#0072B2",
    "attention": "#D55E00",
    "ffn":       "#009E73",
    "norm":      "#56B4E9",
    "softmax":   "#F0E442",
    "rope":      "#CC79A7",
    "residual":  "#999999",
    "kv_cache":  "#E69F00",
    "embedding": "#000000",
}
CAT_ORDER = ["matmul", "attention", "ffn", "norm", "softmax", "rope", "residual", "kv_cache", "embedding"]

# task display + ordering along the prefill-heavy <-> decode-heavy spectrum
TASK_ORDER = ["longbench-triviaqa", "humaneval", "gsm8k", "sharegpt"]
TASK_LABEL = {"longbench-triviaqa": "LongBench\n(prefill-heavy)", "humaneval": "HumanEval",
              "gsm8k": "GSM8K", "sharegpt": "ShareGPT\n(decode-heavy)"}
MODEL_ORDER = ["llama-3.2-1b", "llama-3.2-3b", "qwen2.5-7b", "llama-3.1-8b"]
MODEL_LABEL = {"llama-3.2-1b": "Llama-3.2-1B", "llama-3.2-3b": "Llama-3.2-3B",
               "qwen2.5-7b": "Qwen2.5-7B", "llama-3.1-8b": "Llama-3.1-8B"}


def save(fig, path_noext, dpi=600):
    """Write PNG (embedded in the HTML report) + PDF + SVG (vector archives)."""
    from pathlib import Path
    Path(path_noext).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path_noext}.png", dpi=dpi, bbox_inches="tight")
    # suppress the embedded creation timestamp so vector exports are byte-stable across regens
    fig.savefig(f"{path_noext}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(f"{path_noext}.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
