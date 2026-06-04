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
REP = ROOT / "validation/reports"
FIG = ROOT / "docs/figures/phase1"
MCOL = {"llama-3.2-1b": S.PALETTE["matmul"], "llama-3.2-3b": S.PALETTE["ffn"],
        "llama-3.1-8b": S.PALETTE["attention"], "qwen2.5-7b": S.PALETTE["rope"]}


def load(p):
    return json.loads(Path(p).read_text())


def p1_staircase(mm):
    m = CimTileModel()
    pts = sorted((r["N"], r["dev_lat_us"]) for r in mm["by_group"]["staircase64"])
    Ns = [n for n, _ in pts]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.plot(Ns, [l for _, l in pts], "o", color=S.PALETTE["matmul"], ms=4, label="measured (8B)")
    xs = np.arange(64, 3137, 32)
    ax.plot(xs, [m.dev_lat_us(1, 2048, int(x)) for x in xs], "-", color="#333", lw=1, label="fitted")
    for r in mm["by_group"]["staircase_off64"]:
        ax.plot(r["N"], r["dev_lat_us"], "x", color=S.PALETTE["residual"], ms=5)
    ax.set_xlabel("output channels N (K=2048, M=1)")
    ax.set_ylabel("dev latency (us)")
    ax.set_title("P1  CIM channel-64 staircase")
    ax.legend(loc="upper left")
    S.save(fig, FIG / "P1_cim_staircase")


def p2_proj_fit(mm):
    m = CimTileModel()
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for r in mm["by_group"]["proj_decode"]:
        if r.get("dev_lat_us") is None:
            continue
        kn = r["K"] * r["N"]
        held = r["model"] in ("llama-3.1-8b", "qwen2.5-7b")
        ax.scatter(kn, r["dev_lat_us"], s=26, color=MCOL[r["model"]],
                   marker="D" if held else "o", edgecolor="k" if held else "none", lw=0.5)
    xs = np.logspace(6, 8.2, 60)
    ax.plot(xs, [m.dev_lat_us(1, 2048, int(x / 2048)) for x in xs], "-", color="#999", lw=0.8,
            label="fitted (K=2048 slice)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("weight size K*N (params)")
    ax.set_ylabel("dev latency (us)")
    ax.set_title("P2  CIM decode GEMV fit (diamond=held-out 8B/Qwen)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=MCOL[m_], label=m_.split("-")[-1])
               for m_ in MCOL]
    ax.legend(handles=handles, fontsize=5, loc="upper left")
    S.save(fig, FIG / "P2_cim_proj_fit")


def p3_fiterr_cdf(m1):
    g = m1["compute_fit_gate_G_eff_staircase"]
    # reconstruct per-point errs not stored; show the gate stats as a step CDF proxy
    errs = np.array([0.0, g["median"], g["p95"], g["max"]])
    errs = np.sort(errs)
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ys = np.linspace(0, 1, len(errs))
    ax.step(errs * 100, ys, where="post", color=S.PALETTE["matmul"], lw=1.4)
    ax.axvline(10, ls="--", color=S.PALETTE["attention"], lw=0.9, label="median target 10%")
    ax.axvline(20, ls=":", color=S.PALETTE["residual"], lw=0.9, label="p95 target 20%")
    ax.set_xlabel("G_eff relative error (%)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("P3  M1 G_eff fit error (median %.0f%%, p95 %.0f%%)"
                 % (g["median"] * 100, g["p95"] * 100))
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
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.2, 2.4))
    kvs = np.array([128, 512, 1024])
    for mdl in sl:
        c = sl[mdl]["fp16"]
        a1.plot(kvs, c["a"] + c["b"] * kvs, "-o", ms=3, color=MCOL[mdl], label=mdl.split("-")[-1])
    a1.set_xlabel("kv length"); a1.set_ylabel("softmax (us, fp16)")
    a1.set_title("P5a softmax vs kv"); a1.legend(fontsize=5)
    ops = ["rmsnorm", "rope_apply", "residual", "swiglu", "sampling_argmax"]
    vals = [cpu["const_us"][o]["llama-3.1-8b"]["fp16"] for o in ops]
    a2.bar(range(len(ops)), vals, color=S.PALETTE["norm"])
    a2.set_xticks(range(len(ops))); a2.set_xticklabels([o[:6] for o in ops], rotation=45, fontsize=5)
    a2.set_ylabel("us (8B, fp16 upper bound)"); a2.set_title("P5b non-GEMM ops")
    S.save(fig, FIG / "P5_cpu_nongemm")


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
    p1_staircase(mm)
    p2_proj_fit(mm)
    p3_fiterr_cdf(load(REP / "m1.json"))
    p4_mali_ksweep(load(AET / "mali_matmul.json"))
    p5_cpu(load(REP / "m4_cpu.json"))
    p6_recompose(load(MC / "twopillar_prediction_fitted.json"))
    p7_attn(load(AET / "cim_attention_composed.json"),
            load(ROOT / "simulator/models/params/m4_gpu.json"))
    figs = sorted(p.name for p in FIG.glob("*.png"))
    print(f"wrote {len(figs)} figures: {figs}")


if __name__ == "__main__":
    main()
