"""Phase 0.3 A4 (runner, runs on aetina in ~/edge-cim-simulation/.rknnvenv).

Loads the .rknn files produced by convert_rknn.py (rsync'd from metiscard) and times
on-board RKNPU2 inference (warmup + N iters, median). Projection = 1 input A[M,K];
attention = 2 inputs A[M,K], B[K,N] (native activation x activation). FP16.

Run: ~/edge-cim-simulation/.rknnvenv/bin/python run_rknpu2.py <rknn_dir>
"""
import json, sys, time, statistics
from pathlib import Path
import numpy as np
from rknnlite.api import RKNNLite

RDIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "rknn_out"
OUT = Path.home() / "edge-cim-simulation/measurements/aetina"; OUT.mkdir(parents=True, exist_ok=True)
WARMUP, ITERS = 5, 30


def bench(rknn, inputs):
    for _ in range(WARMUP):
        rknn.inference(inputs=inputs)
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        rknn.inference(inputs=inputs)
        ts.append((time.perf_counter() - t0) * 1e6)
    return {"median_us": round(statistics.median(ts), 2),
            "p95_us": round(sorted(ts)[int(0.95 * len(ts))], 2),
            "cov": round(statistics.pstdev(ts) / statistics.mean(ts), 3)}


def main():
    man = json.loads((RDIR / "manifest.json").read_text())
    results = {}
    for tag, m in man.items():
        if not m.get("ok"):
            results[tag] = {**m, "skip": "not_converted"}; continue
        M, K, N = m["M"], m["K"], m["N"]
        rk = RKNNLite()
        if rk.load_rknn(str(RDIR / f"{tag}.rknn")) != 0:
            results[tag] = {**m, "error": "load"}; continue
        if rk.init_runtime() != 0:
            results[tag] = {**m, "error": "init_runtime"}; rk.release(); continue
        inputs = [np.random.randn(M, K).astype(np.float32)]
        if m["two_input"]:
            inputs.append(np.random.randn(K, N).astype(np.float32))
        try:
            r = bench(rk, inputs)
            flops = 2 * M * K * N
            results[tag] = {**m, **r, "gflops": round(flops / (r["median_us"] / 1e6) / 1e9, 2)}
            print(f"{tag:24s} M{M}K{K}N{N} -> {r['median_us']:.1f}us {results[tag]['gflops']:.1f} GFLOP/s", flush=True)
        except Exception as e:
            results[tag] = {**m, "error": str(e)[:120]}
            print(f"{tag:24s} ERROR {e}", flush=True)
        rk.release()
    (OUT / "rknpu2_matmul.json").write_text(json.dumps(results, indent=1))
    n = sum(1 for v in results.values() if "median_us" in v)
    print(f"\n{n}/{len(results)} timed -> rknpu2_matmul.json")


if __name__ == "__main__":
    main()
