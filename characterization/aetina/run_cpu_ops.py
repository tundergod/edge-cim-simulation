"""Phase 0.3 A6 — non-GEMM support ops on the RK3588 A76 (Cortex-A76 big cluster).

Times the LLM support ops the profile assigns to CPU — RMSNorm, RoPE-apply, softmax,
residual add, SwiGLU (silu*mul), and greedy sampling (argmax over vocab) — at decode
(M=1) shapes per model, in FP16 and FP32 (vendor-actual precisions). Reports median +
p95 over many iterations. Feeds the C5 support-latency term and the A6 deliverable.

Run pinned to A76: taskset -c 4-7 .rknnvenv/bin/python run_cpu_ops.py
"""
import json, time, statistics, os
from pathlib import Path
import numpy as np

OUT = Path.home() / "edge-cim-simulation/measurements/aetina"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "llama-3.2-1b": dict(H=2048, F=8192, heads=32, hd=64, V=128256),
    "llama-3.2-3b": dict(H=3072, F=8192, heads=24, hd=128, V=128256),
    "llama-3.1-8b": dict(H=4096, F=14336, heads=32, hd=128, V=128256),
    "qwen2.5-7b":   dict(H=3584, F=18944, heads=28, hd=128, V=152064),
}
KVS = [128, 512, 1024]
ITERS, WARMUP = 200, 20


def bench(fn, *args):
    for _ in range(WARMUP):
        fn(*args)
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn(*args)
        ts.append((time.perf_counter() - t0) * 1e6)  # us
    return {"median_us": round(statistics.median(ts), 3),
            "p95_us": round(sorted(ts)[int(0.95 * len(ts))], 3),
            "cov": round(statistics.pstdev(ts) / statistics.mean(ts), 3)}


def rmsnorm(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6) * w


def rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return np.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).reshape(x.shape)


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def main():
    res = {"cores": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None, "ops": {}}
    for model, c in MODELS.items():
        H, F, heads, hd, V = c["H"], c["F"], c["heads"], c["hd"], c["V"]
        for dt in [np.float16, np.float32]:
            tn = {np.float16: "fp16", np.float32: "fp32"}[dt]
            x = np.random.randn(1, H).astype(dt)
            w = np.random.randn(H).astype(dt)
            xf = np.random.randn(1, F).astype(dt)
            q = np.random.randn(1, heads, hd).astype(dt)
            cs = np.random.randn(1, heads, hd // 2).astype(dt)
            logits = np.random.randn(1, V).astype(dt)
            ops = {
                "rmsnorm": (rmsnorm, (x, w)),
                "rope_apply": (rope, (q, cs, cs)),
                "residual": (lambda a, b: a + b, (x, x.copy())),
                "swiglu": (lambda a, b: (a / (1 + np.exp(-a.astype(np.float32))).astype(dt)) * b, (xf, xf.copy())),
                "sampling_argmax": (lambda l: int(l.argmax(-1)[0]), (logits,)),
            }
            for kv in KVS:
                sc = np.random.randn(1, heads, 1, kv + 1).astype(dt)
                ops[f"softmax_kv{kv}"] = (softmax, (sc,))
            for name, (fn, args) in ops.items():
                key = f"{model}/{tn}/{name}"
                res["ops"][key] = {"model": model, "dtype": tn, "op": name, **bench(fn, *args)}
                print(f"{key:42s} {res['ops'][key]['median_us']:8.2f} us", flush=True)
    (OUT / "cpu_ops.json").write_text(json.dumps(res, indent=1))
    print(f"\nwrote {len(res['ops'])} op timings to cpu_ops.json")


if __name__ == "__main__":
    main()
