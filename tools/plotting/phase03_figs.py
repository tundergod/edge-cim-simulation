"""Phase 0.3 figures (nature-figure style) — CIM characterization + offload + two-pillar.

Reads only committed JSON:
  measurements/aetina/metis_alpha_matmul.json     (CIM, structured)
  measurements/aetina/cim_attention_composed.json (C4)
  measurements/aetina/mali_matmul.json            (A5)
  measurements/metis_card/twopillar_prediction.json (C5)
Writes docs/figures/phase0.3/*.{png,pdf,svg}.

Run: ./.venv/bin/python tools/plotting/phase03_figs.py
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _style import PALETTE, MODEL_ORDER, MODEL_LABEL, save

AET = Path("measurements/aetina")
MC = Path("measurements/metis_card")
FIG = Path("docs/figures/phase0.3")
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#999999"


def load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def fig_pcie_floor(cim):
    """A1d.5: dev vs system latency — the per-call DMA fixed floor dominates small GEMVs."""
    rows = [r for g in cim["by_group"].values() for r in g if "dev_lat_us" in r and r.get("tiles", 1) == 1]
    dev = np.array([r["dev_lat_us"] for r in rows])
    sysl = np.array([r["system_lat_us"] for r in rows])
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.scatter(dev, sysl, s=14, color=BLUE, edgecolor="white", linewidth=0.3, zorder=3)
    floor = cim["pcie_floor_A1d5"].get("fixed_overhead_us_median", 0)
    lo, hi = dev.min() * 0.8, sysl.max() * 1.2
    ax.plot([lo, hi], [lo, hi], "--", color=GREY, lw=0.7, label="system = dev (no overhead)")
    ax.axhline(floor, color=ORANGE, lw=0.8, ls=":", label=f"fixed floor ≈ {floor:.0f} µs")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("device compute latency (µs)"); ax.set_ylabel("end-to-end system latency (µs)")
    ax.set_title("CIM per-call DMA floor dominates\nsmall decode GEMVs (A1d.5)", fontsize=8)
    ax.legend(fontsize=6); ax.tick_params(length=2)
    fig.tight_layout(); save(fig, str(FIG / "fig3_pcie_floor"))


def fig_channel64(cim):
    """A1d.2: latency vs N channel-alignment staircase + off-64 probes."""
    st = sorted([r for r in cim["by_group"].get("staircase64", []) if "dev_lat_us" in r], key=lambda r: r["N"])
    off = [r for r in cim["by_group"].get("staircase_off64", []) if "dev_lat_us" in r]
    if not st:
        return
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.plot([r["N"] for r in st], [r["dev_lat_us"] for r in st], "-o", color=BLUE, ms=4, lw=1, label="N = 64·k")
    if off:
        ax.scatter([r["N"] for r in off], [r["dev_lat_us"] for r in off], s=28, marker="x",
                   color=ORANGE, zorder=4, label="off-64 N")
    ax.set_xlabel("output channels N (K=2048, M=1)"); ax.set_ylabel("device latency (µs)")
    ax.set_title("CIM channel-tile (64) staircase (A1d.2)", fontsize=8)
    ax.legend(fontsize=6.5); ax.tick_params(length=2)
    fig.tight_layout(); save(fig, str(FIG / "fig4_channel64_staircase"))


def fig_twopillar(tp):
    """C5: predicted vs measured decode tok/s (1b/3b fit -> 8b hold-out)."""
    wb = tp["weight_bytes"]; meas = tp["measured_tok_s_1c"]
    BW = tp["fit_BW_GBs"] * 1e9
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    xs = np.array([wb[m] for m in MODEL_ORDER if m in wb]) / 1e9
    for m in ["llama-3.2-1b", "llama-3.2-3b"]:
        ax.scatter(wb[m] / 1e9, meas[m], s=40, color=BLUE, edgecolor="white", zorder=3,
                   label="fit (1B, 3B)" if m == "llama-3.2-1b" else None)
    ax.scatter(wb["llama-3.1-8b"] / 1e9, meas["llama-3.1-8b"], s=55, marker="D", color=ORANGE,
               edgecolor="white", zorder=4, label="8B measured (held out)")
    ax.scatter(wb["llama-3.1-8b"] / 1e9, tp["pred_8b_tok_s"], s=55, marker="*", color=GREEN,
               edgecolor="white", zorder=5, label="8B predicted")
    g = np.linspace(xs.min() * 0.9, xs.max() * 1.1, 50)
    ax.plot(g, BW / (g * 1e9), "--", color=GREY, lw=0.8, label=f"{tp['fit_BW_GBs']:.0f} GB/s wall")
    ax.set_xlabel("model weight bytes (GB, INT8)"); ax.set_ylabel("decode tok/s (1-core)")
    ax.set_title(f"Two-pillar: micro→end-to-end\n8B pred {tp['pred_8b_tok_s']:.2f} vs meas "
                 f"{tp['measured_8b_tok_s']:.2f} ({tp['rel_error']*100:.0f}% err)", fontsize=7.5)
    ax.legend(fontsize=6); ax.tick_params(length=2)
    fig.tight_layout(); save(fig, str(FIG / "fig6_twopillar"))


def fig_cim_vs_gpu_attn(c4, mali):
    """Offload: CIM composed attention (penalty) vs GPU native attention, vs CIM floor."""
    if not c4 or not c4.get("rows"):
        return
    kv = [r["kv"] for r in c4["rows"]]
    floor = [r["floor_us"] for r in c4["rows"]]
    comp = [r["composed_us"] for r in c4["rows"]]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.plot(kv, floor, "-o", color=BLUE, ms=4, lw=1, label="CIM conv-proxy floor")
    ax.plot(kv, comp, "-s", color=ORANGE, ms=4, lw=1, label="CIM composed (+KV reload)")
    if mali:
        # GPU native decode attention = qkT_dec (N=kv) + sv_dec (K=kv) FP16, summed per kv
        gpu = []
        for r in c4["rows"]:
            kvv = r["kv"]
            tot = sum(x["f16_ms"] * 1000 for x in mali["results"] if x["group"] == "attn"
                      and ((x["tag"] == "qkT_dec" and x["N"] == kvv) or (x["tag"] == "sv_dec" and x["K"] == kvv)))
            gpu.append(tot if tot > 0 else np.nan)
        ax.plot(kv, gpu, "-^", color=GREEN, ms=4, lw=1, label="GPU (Mali) native FP16")
    ax.set_yscale("log"); ax.set_xlabel("kv length"); ax.set_ylabel("single-head attention latency (µs)")
    ax.set_title("Attention: CIM penalty vs GPU native\n(motivates offload)", fontsize=8)
    ax.legend(fontsize=6); ax.tick_params(length=2)
    fig.tight_layout(); save(fig, str(FIG / "fig7_cim_vs_gpu_attn"))


def fig_cim_roofline(cim):
    """CIM effective throughput (GFLOP/s) across projection families (decode M=1)."""
    rows = [r for r in cim["by_group"].get("proj_decode", []) if "dev_gflops" in r]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    fam_color = {"q_o": BLUE, "kv": ORANGE, "gate_up": GREEN, "down": GREY}
    for r in rows:
        ax.scatter(r["K"] * r["N"] / 1e6, r["dev_gflops"], s=20, color=fam_color.get(r["family"], "k"),
                   edgecolor="white", linewidth=0.3, zorder=3)
    for fam, col in fam_color.items():
        ax.scatter([], [], color=col, label=fam)
    ax.set_xscale("log"); ax.set_xlabel("weight size K·N (M params)")
    ax.set_ylabel("CIM device throughput (GFLOP/s)")
    ax.set_title("CIM matmul throughput vs op (decode M=1)\ntiled ops amortise differently", fontsize=8)
    ax.legend(fontsize=6.5, title="projection"); ax.tick_params(length=2)
    fig.tight_layout(); save(fig, str(FIG / "fig5_cim_throughput"))


def main():
    cim = load(AET / "metis_alpha_matmul.json")
    c4 = load(AET / "cim_attention_composed.json")
    mali = load(AET / "mali_matmul.json")
    tp = load(MC / "twopillar_prediction.json")
    if cim:
        fig_pcie_floor(cim); fig_channel64(cim); fig_cim_roofline(cim)
    if c4:
        fig_cim_vs_gpu_attn(c4, mali)
    if tp:
        fig_twopillar(tp)
    print(f"wrote Phase 0.3 figures to {FIG}/ (cim={bool(cim)} c4={bool(c4)} mali={bool(mali)} tp={bool(tp)})")


if __name__ == "__main__":
    main()
