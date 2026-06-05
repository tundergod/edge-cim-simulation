"""Phase 1 figures (build artifacts, nature-figure style). 7 figures P1-P7.

Each figure is regenerable from committed JSON (measurements/ + validation/reports/ +
simulator/models/params/). Writes PNG (for the HTML report) + PDF + SVG to docs/figures/phase1/.

Run: ./.venv/bin/python tools/plotting/phase1_figs.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/plotting"))
import _style as S  # noqa: E402
from simulator.models.m1_cim_tile import CimTileModel  # noqa: E402

AET = ROOT / "measurements/aetina"
MC = ROOT / "measurements/metis_card"
OP = ROOT / "measurements/op_profile"
REP = ROOT / "validation/reports"
FIG = ROOT / "docs/figures/phase1"
MCOL = {"llama-3.2-1b": S.PALETTE["matmul"], "llama-3.2-3b": S.PALETTE["ffn"],
        "llama-3.1-8b": S.PALETTE["attention"], "qwen2.5-7b": S.PALETTE["rope"]}


def load(p):
    return json.loads(Path(p).read_text())


def p1_staircase(mm):
    """P1: latency vs N at K=2048. Native points only; >2048 = rising UNVALIDATED extrapolation."""
    m = CimTileModel()
    nat = sorted((r["N"], r["dev_lat_us"]) for r in mm["by_group"]["staircase64"]
                 if not r.get("tiled_extrapolated"))          # exclude generated N=3072
    off = [(r["N"], r["dev_lat_us"]) for r in mm["by_group"]["staircase_off64"]]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot([n for n, _ in nat], [l for _, l in nat], "o", color=S.PALETTE["matmul"], ms=4,
            label="measured (native)")
    ax.plot([n for n, _ in off], [l for _, l in off], "x", color=S.PALETTE["residual"], ms=5,
            label="off-64 probe (native)")
    x1 = np.arange(64, 2049, 16)
    ax.plot(x1, [m.dev_lat_us(1, 2048, int(x)) for x in x1], "-", color="#333", lw=1.1,
            label="model (calibrated)")
    x2 = np.arange(2048, 4097, 16)
    ax.plot(x2, [m.dev_lat_us(1, 2048, int(x)) for x in x2], "--", color=S.PALETTE["attention"],
            lw=1.2, label="model (extrapolation, UNVALIDATED)")
    ax.axvline(2048, ls=":", color="#aaa", lw=0.8)
    ax.text(2120, 13, "4 cores x 512 = 2048\nno native data above", fontsize=5, color="#888")
    ax.set_xlabel("output channels N (K=2048, M=1)")
    ax.set_ylabel("dev latency (us)")
    ax.set_title("P1  CIM latency vs N — >2048 rises (not flat), unvalidated")
    ax.legend(loc="upper left", fontsize=5)
    S.save(fig, FIG / "P1_cim_staircase")


KCOL = {1024: "#56B4E9", 2048: "#0072B2", 3072: "#009E73", 3584: "#CC79A7", 4096: "#D55E00"}


def p2_geff(m1params):
    """P2: 2D effective throughput G_eff(N,K) — measured (dots) vs 2D fit (lines), per K."""
    m = CimTileModel(m1params)
    pts = m1params["native_throughput_points"]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for K in sorted({p["K"] for p in pts}):
        kp = sorted([p for p in pts if p["K"] == K], key=lambda x: x["N"])
        c = KCOL.get(K, "#999")
        ax.scatter([p["N"] for p in kp], [p["gops"] for p in kp], s=24, color=c, zorder=3,
                   label=f"K={K}")
        Nf = np.linspace(64, 2048, 40)
        ax.plot(Nf, [m.g_eff(n, K) for n in Nf], "-", color=c, lw=0.9, alpha=0.8)
    ax.set_xlabel("output channels N")
    ax.set_ylabel("effective throughput (GOP/s, INT8)")
    ax.set_title("P2  CIM G_eff(N,K): meas (dots) vs 2D fit (lines)")
    ax.legend(fontsize=5, title="wider K -> higher", title_fontsize=5)
    S.save(fig, FIG / "P2_cim_geff")


def p3_fiterr_cdf(m1params):
    """P3: CDF of the native throughput-fit relative error vs the 10%/20% targets."""
    m = CimTileModel(m1params)
    errs = np.sort([abs(m.g_eff(p["N"], p["K"]) - p["gops"]) / p["gops"]
                    for p in m1params["native_throughput_points"]])
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ys = np.arange(1, len(errs) + 1) / len(errs)
    ax.step(errs * 100, ys, where="post", color=S.PALETTE["matmul"], lw=1.5)
    ax.axvline(10, ls="--", color=S.PALETTE["attention"], lw=0.9, label="median target 10%")
    ax.axvline(20, ls=":", color=S.PALETTE["residual"], lw=0.9, label="p95 target 20%")
    ax.set_xlabel("G_eff(N,K) relative error (%)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("P3  M1 throughput fit error (median %.0f%%, n=%d)"
                 % (np.median(errs) * 100, len(errs)))
    ax.legend(fontsize=5)
    S.save(fig, FIG / "P3_m1_fiterr_cdf")


def p4_mali_ksweep(mali):
    ks = sorted((r["M"], r["f16_gflops"], r["f32_gflops"]) for r in mali["results"]
                if r.get("group") == "ksweep")
    Ms = [m for m, _, _ in ks]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.plot(Ms, [f for _, f, _ in ks], "o-", color=S.PALETTE["matmul"], ms=4, label="f16")
    ax.plot(Ms, [f for _, _, f in ks], "s--", color=S.PALETTE["residual"], ms=4, label="f32")
    ax.axvline(128, ls=":", color=S.PALETTE["attention"], lw=0.9)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("square matrix dim M")
    ax.set_ylabel("throughput (GFLOP/s)")
    ax.set_title("P4  Mali GEMM (saturates ~20 by M=128; LOWER BOUND)")
    ax.legend(fontsize=6)
    S.save(fig, FIG / "P4_mali_ksweep")


def p5_cpu(cpu):
    sl = cpu["softmax_linear"]
    cpu_ops = load(AET / "cpu_ops.json")["ops"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.2, 2.4))
    kvs = np.array([128, 512, 1024])
    for mdl in sl:
        c = sl[mdl]["fp16"]
        xs = np.linspace(128, 1024, 40)
        a1.plot(xs, c["a"] + c["b"] * xs, "-", lw=1, color=MCOL[mdl], label=mdl.split("-")[-1])
        meas = [cpu_ops[f"{mdl}/fp16/softmax_kv{kv}"]["median_us"] for kv in kvs]
        a1.scatter(kvs, meas, s=18, color=MCOL[mdl], zorder=3)  # measured points
    a1.set_xlabel("kv length"); a1.set_ylabel("softmax (us, fp16)")
    a1.set_title("P5a softmax: meas (dots) vs fit (line)"); a1.legend(fontsize=5)
    ops = ["rmsnorm", "rope_apply", "residual", "swiglu", "sampling_argmax"]
    vals = [cpu["const_us"][o]["llama-3.1-8b"]["fp16"] for o in ops]
    a2.bar(range(len(ops)), vals, color=S.PALETTE["norm"])
    a2.set_xticks(range(len(ops))); a2.set_xticklabels([o[:6] for o in ops], rotation=45, fontsize=5)
    a2.set_ylabel("us (8B, fp16 upper bound)"); a2.set_title("P5b non-GEMM ops")
    S.save(fig, FIG / "P5_cpu_nongemm")


def m7_energy():
    """M7: per-token 8B decode energy breakdown (log) — memory-dominated, +/-20% robust."""
    r = load(REP / "m7.json")
    b = r["per_token_8b_decode_mJ"]
    labels = ["CIM compute", "DRAM stream", "CPU support"]
    vals = [b["cim_proj_mJ"], b["dram_stream_mJ"], b["cpu_support_mJ"]]
    cols = [S.PALETTE["matmul"], S.PALETTE["kv_cache"], S.PALETTE["norm"]]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    bars = ax.bar(labels, vals, color=cols, width=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("energy per token (mJ, log)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.15, f"{v:.0f}" if v >= 10 else f"{v:.1f}",
                ha="center", fontsize=7)
    flips = r["sensitivity_pm20pct"]["conclusion_flips"]
    ax.set_title(f"M7  8B decode energy (memory-dominated; {flips} flips ±20%)", fontsize=7.5)
    S.save(fig, FIG / "M7_energy")


def m5_coverage():
    """M5 consistency: per-model op-category coverage grid (all traced) + 0 orphans."""
    import glob
    m5 = load(REP / "m5.json")
    cats = ["matmul", "attention", "softmax", "norm", "rope", "ffn", "residual", "kv_cache", "embedding"]
    models = ["llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b", "qwen2.5-7b"]
    grid = np.zeros((len(models), len(cats)))
    for i, m in enumerate(models):
        present = set()
        for f in glob.glob(str(OP / f"{m}_*.json")):
            for row in load(f)["rows"]:
                present.add(row["category"])
        for jc, c in enumerate(cats):
            grid[i, jc] = 1 if c in present else 0
    fig, ax = plt.subplots(figsize=(5.0, 1.9))
    ax.imshow(grid, cmap="Greens", vmin=0, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(models))); ax.set_yticklabels([m.split("-")[-1] for m in models], fontsize=6)
    for i in range(len(models)):
        for j in range(len(cats)):
            ax.text(j, i, "✓" if grid[i, j] else "✗", ha="center", va="center",
                    color="#1b7f5a" if grid[i, j] else "#C45A12", fontsize=7)
    orphans = sum(len(r["orphan_ops"]) for r in m5["per_model"].values())
    ax.set_title(f"M5  op-category coverage (4 models x 9 cats) — {orphans} orphans", fontsize=8)
    S.save(fig, FIG / "M5_coverage")


def p6_recompose(rc):
    models = ["llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b"]
    meas = [rc["measured_tok_s_1c"][m] for m in models]
    BW = rc["fit_BW_GBs"] * 1e9
    wb = {m: rc["per_token_weight_bytes"][m] * 1e9 for m in models}
    pred = [BW / wb[m] for m in models]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.bar(x - 0.2, meas, 0.4, color=S.PALETTE["matmul"], label="measured")
    ax.bar(x + 0.2, pred, 0.4, color=S.PALETTE["attention"], label="predicted")
    for i, m in enumerate(models):
        ax.errorbar(i + 0.2, pred[i], yerr=0.25 * pred[i], color="k", lw=0.8, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels([m.split("-")[-1] for m in models])
    ax.set_ylabel("decode tok/s (1-core)")
    ax.set_title("P6  recompose hold-out (8B err %.0f%%, +/-25%% band)" % (rc["rel_error_8b"] * 100))
    ax.annotate("8B held-out", (2.2, pred[2]), fontsize=5)
    ax.legend(fontsize=6)
    S.save(fig, FIG / "P6_recompose_holdout")


def m2_pcie_floor():
    """M2 meas-vs-pred: measured per-call floor (system-dev) vs the fixed 911us prediction."""
    import statistics
    raw = load(AET / "metis_alpha_matmul_raw.json")
    pts = [(r["dev_lat_us"], r["system_lat_us"] - r["dev_lat_us"]) for r in raw.values()
           if "system_lat_us" in r and not r.get("tiled_extrapolated") and r.get("tiles", 1) == 1]
    devs = [d for d, _ in pts]
    floors = [f for _, f in pts]
    med = statistics.median(floors)
    p95 = sorted(floors)[int(0.95 * len(floors))]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.scatter(devs, floors, s=24, color=S.PALETTE["matmul"], alpha=0.8, label="measured floor (30 shapes)")
    ax.axhline(med, color=S.PALETTE["attention"], lw=1.3, label=f"prediction = {med:.0f}us (median)")
    ax.axhline(p95, ls=":", color=S.PALETTE["residual"], lw=0.9, label=f"p95 = {p95:.0f}us")
    ax.set_ylim(0, max(floors) * 1.15)
    ax.set_xlabel("device compute latency (us)")
    ax.set_ylabel("per-call host<->device floor (us)")
    ax.set_title("M2  PCIe floor: measured vs fixed 911us")
    ax.legend(fontsize=5, loc="lower right")
    S.save(fig, FIG / "M2_pcie_floor")


def m4gpu_attn_fit(gpu_report):
    """M4-GPU meas-vs-pred: measured single-head attn (qkT+sv) vs fitted a+b*kv."""
    g = gpu_report["attn_offload_gate"]
    kvs = [int(k) for k in g["meas_us"]]
    meas = [g["meas_us"][str(k)] for k in kvs]
    fit = [g["fit_us"][str(k)] for k in kvs]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.scatter(kvs, meas, s=40, color=S.PALETTE["matmul"], zorder=3, label="measured (f16)")
    xs = np.linspace(min(kvs), max(kvs), 50)
    a = gpu_report["params"]["attn_bmm_a_us"]
    b = gpu_report["params"]["attn_bmm_b_us_per_kv"]
    ax.plot(xs, a + b * xs, "-", color=S.PALETTE["attention"], lw=1.2,
            label=f"fit = {a:.1f} + {b:.3f}·kv")
    ax.set_xlabel("KV length")
    ax.set_ylabel("single-head attn QK^T+S·V (us)")
    ax.set_title("M4-GPU  attention fit (median %.1f%%)" % (g["median_relerr"] * 100))
    ax.legend(fontsize=6, loc="upper left")
    S.save(fig, FIG / "M4gpu_attn_fit")


def p7_attn(c4, gpu_params):
    a, b = gpu_params["attn_bmm_a_us"], gpu_params["attn_bmm_b_us_per_kv"]
    kvs = [r["kv"] for r in c4["rows"]]
    cim_ms = [r["composed_us"] / 1000 for r in c4["rows"]]
    gpu_ms = [(a + b * kv) / 1000 for kv in kvs]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.plot(kvs, cim_ms, "o-", color=S.PALETTE["attention"], ms=4, label="CIM composed (C4)")
    ax.plot(kvs, gpu_ms, "s-", color=S.PALETTE["ffn"], ms=4, label="Mali GPU-native")
    ax.set_yscale("log")
    ax.set_xlabel("KV length")
    ax.set_ylabel("per-token attention (ms, log)")
    ax.set_title("P7  CIM penalty vs GPU offload (~2 orders, 96-370x)")
    ax.legend(fontsize=6)
    S.save(fig, FIG / "P7_attn_offload")


def main():
    mm = load(AET / "metis_alpha_matmul.json")
    m1params = load(ROOT / "simulator/models/params/m1_cim.json")
    p1_staircase(mm)
    p2_geff(m1params)
    p3_fiterr_cdf(m1params)
    p4_mali_ksweep(load(AET / "mali_matmul.json"))
    p5_cpu(load(REP / "m4_cpu.json"))
    m2_pcie_floor()
    m4gpu_attn_fit(load(REP / "m4_gpu.json"))
    m5_coverage()
    m7_energy()
    p6_recompose(load(MC / "twopillar_prediction_fitted.json"))
    p7_attn(load(AET / "cim_attention_composed.json"),
            load(ROOT / "simulator/models/params/m4_gpu.json"))
    figs = sorted(p.name for p in FIG.glob("*.png"))
    print(f"wrote {len(figs)} figures: {figs}")


if __name__ == "__main__":
    main()
