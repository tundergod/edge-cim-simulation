"""Phase 0.3 A1/A1d — CIM (Metis Alpha) characterization via the 1x1-conv proxy.

Runs INSIDE the aetina SDK container (venv active). For each (M,K,N) target it builds a
1x1 conv ONNX (weight [N,K,1,1] = the stationary matrix; input [1,K,1,M]; output [1,N,1,M]
== an [M,K]x[K,N] matmul), compiles INT8, and runs axrunmodel for dev/host/system FPS.
Faithful for weight-stationary GEMM/GEMV (linear projections); attention is a compute
LOWER BOUND only (both operands are activations) and is tagged as such.

Resumable + incremental: results append to results.json keyed by task id; re-run skips done
tasks. Any compile/axrunmodel failure is logged and skipped (collect-what-you-can).

Manifest covers: projection families (decode M=1 + prefill M grid + spot 2048/4096),
l2-vs-ddr residency, channel-64 staircase, (M,K,N) aspect, GQA-narrow, attention floor.

Run: python run_metis_cim.py [--seconds 5] [--dataset-len 20] [--only GROUP]
"""
import argparse, json, re, shutil, subprocess, time
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

WORK = Path("/tmp/cim_work"); WORK.mkdir(parents=True, exist_ok=True)
SDK = "/home/ubuntu/voyager-sdk"
RESULTS = WORK / "results.json"
LOG = WORK / "progress.log"
FPS_RE = re.compile(r"dev:([\d.]+)\s+host:([\d.]+)\s+system:([\d.]+)fps")

MODELS = {
    "llama-3.2-1b": dict(H=2048, F=8192, kv=512,  V=128256, hd=64,  L=16),
    "llama-3.2-3b": dict(H=3072, F=8192, kv=1024, V=128256, hd=128, L=28),
    "llama-3.1-8b": dict(H=4096, F=14336, kv=1024, V=128256, hd=128, L=32),
    "qwen2.5-7b":   dict(H=3584, F=18944, kv=512,  V=152064, hd=128, L=28),
}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def build_conv_onnx(M, K, N, bias, path):
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, K, 1, M])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, N, 1, M])
    inits = [numpy_helper.from_array(np.random.randn(N, K, 1, 1).astype(np.float32), "W")]
    conv_in = ["input", "W"]
    if bias:
        inits.append(numpy_helper.from_array(np.random.randn(N).astype(np.float32), "B"))
        conv_in.append("B")
    node = helper.make_node("Conv", conv_in, ["output"], kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0])
    g = helper.make_graph([node], "m", [X], [Y], inits)
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]), str(path))


def measure(M, K, N, bias=False, residency=None, seconds=5, dataset_len=20, timeout=900):
    """Compile + axrunmodel one shape. Returns dict (or {'error':...})."""
    d = WORK / "cur"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    try:
        build_conv_onnx(M, K, N, bias, d / "m.onnx")
        cmd = ["compile", "--input", str(d / "m.onnx"), "--input-shape", f"1,{K},1,{M}",
               "--output", str(d / "out"), "--overwrite", "--log-level", "WARNING",
               "--dataset-len", str(dataset_len)]
        if residency:
            (d / "cfg.json").write_text(json.dumps({"dpu_constants_home": f"global.{residency}"}))
            cmd += ["--config", str(d / "cfg.json")]
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=SDK, capture_output=True, text=True, timeout=timeout)
        ct = time.time() - t0
        if rc.returncode != 0:
            return {"error": "compile", "rc": rc.returncode, "stderr": rc.stderr[-300:], "compile_s": round(ct, 1)}
        mj = list((d / "out").rglob("model.json"))
        if not mj:
            return {"error": "no_model_json", "compile_s": round(ct, 1)}
        ax = subprocess.run(["axrunmodel", str(mj[0]), "--seconds", str(seconds)],
                            cwd=SDK, capture_output=True, text=True, timeout=180)
        m = FPS_RE.search(ax.stdout)
        if not m:
            return {"error": "no_fps", "stdout": ax.stdout[-200:], "compile_s": round(ct, 1)}
        dev, host, sys_ = float(m.group(1)), float(m.group(2)), float(m.group(3))
        flops = 2 * M * K * N
        return {"dev_fps": dev, "host_fps": host, "system_fps": sys_,
                "dev_lat_us": 1e6 / dev, "system_lat_us": 1e6 / sys_,
                "dev_gflops": flops * dev / 1e9, "compile_s": round(ct, 1)}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)


# Metis Alpha device-memory envelope: zeMemAllocDevice fails for conv weights with
# K*N > ~6M params (probed: [2048,3072]=6.3M OK; [2048,4096]/[3072,3072]/[8192,2048] fail).
# A matmul above the envelope is run as a grid of K_TILE x N_TILE crossbar tiles (the real
# CIM tiling). Tiles are uniform, so we measure ONE canonical tile per M (and residency) and
# extrapolate full_latency = n_tiles * tile_latency — this is exact for uniform tiles, avoids
# 100+ compiles for lm_head, and the n_tiles x per-call DMA floor IS the A1d tiling-cost finding.
SAFE_KN = 6_000_000
N_TILE = 2048
K_TILE = 2048
_TILE_CACHE = {}


def _ntiles(total, size):
    return (total + size - 1) // size


def measure_op(M, K, N, bias=False, residency=None, seconds=5, dataset_len=20):
    if K * N <= SAFE_KN:
        r = measure(M, K, N, bias, residency, seconds, dataset_len)
        r["tiles"] = 1
        return r
    key = (M, residency)
    if key not in _TILE_CACHE:
        _TILE_CACHE[key] = measure(M, K_TILE, N_TILE, False, residency, seconds, dataset_len)
    t = _TILE_CACHE[key]
    if "error" in t:
        return {"error": f"canonical_tile_fail: {t['error']}"}
    nt = _ntiles(K, K_TILE) * _ntiles(N, N_TILE)
    dev_lat, sys_lat = nt * t["dev_lat_us"], nt * t["system_lat_us"]
    flops = 2 * M * K * N
    return {"dev_lat_us": dev_lat, "system_lat_us": sys_lat,
            "dev_fps": 1e6 / dev_lat, "system_fps": 1e6 / sys_lat,
            "dev_gflops": flops / (dev_lat / 1e6) / 1e9,
            "tiles": nt, "tiled_extrapolated": True, "tile_KN": [K_TILE, N_TILE],
            "canonical_tile_dev_lat_us": t["dev_lat_us"], "canonical_tile_system_lat_us": t["system_lat_us"]}


def manifest():
    """Yield measurement tasks: (task_id, group, params dict)."""
    tasks = []

    def add(group, model, family, M, K, N, bias=False, residency=None):
        tid = f"{group}|{model}|{family}|M{M}|K{K}|N{N}|{'b' if bias else 'nb'}|{residency or 'def'}"
        tasks.append((tid, group, dict(model=model, family=family, M=M, K=K, N=N, bias=bias, residency=residency)))

    for model, c in MODELS.items():
        H, F, kv, V, hd, L = c["H"], c["F"], c["kv"], c["V"], c["hd"], c["L"]
        fams = {"q_o": (H, H), "kv": (H, kv), "gate_up": (H, F), "down": (F, H)}
        # proj decode GEMV (M=1) — the dominant decode ops
        for fam, (K, N) in fams.items():
            add("proj_decode", model, fam, 1, K, N)
        # lm_head decode: tiled (N=4096 chunk; full N is OOM-risk, handled in postproc)
        add("lmhead_tile", model, "lm_head_tile", 1, H, 4096)

    # prefill M scaling + LongBench spot-anchors, on the 8B dominant families
    c = MODELS["llama-3.1-8b"]; H, F = c["H"], c["F"]
    for M in [128, 256, 512, 1024, 2048, 4096]:
        add("proj_prefill", "llama-3.1-8b", "gate_up", M, H, F)
        add("proj_prefill", "llama-3.1-8b", "down", M, F, H)
        add("proj_prefill", "llama-3.1-8b", "q_o", M, H, H)

    # l2 vs ddr residency: 5 families x {1b (fits L2), 8b (spills)} at M=1
    for model in ["llama-3.2-1b", "llama-3.1-8b"]:
        c = MODELS[model]; H, F, kv = c["H"], c["F"], c["kv"]
        for fam, (K, N) in {"q_o": (H, H), "kv": (H, kv), "gate_up": (H, F), "down": (F, H)}.items():
            for res in ["l2", "ddr"]:
                add("l2_ddr", model, fam, 1, K, N, residency=res)

    # channel-64 staircase (M=1, K=2048 so N up to 3072 runs as a single device tile):
    for N in [64, 128, 256, 512, 1024, 1536, 2048, 3072]:
        add("staircase64", "llama-3.1-8b", "stair", 1, 2048, N)
    for N in [480, 544, 1000]:  # off-64 probes (pad-up behaviour)
        add("staircase_off64", "llama-3.1-8b", "stair_off", 1, 2048, N)

    # aspect (equal MAC = 4.19M, all device-runnable): wide vs tall vs square
    add("aspect", "llama-3.1-8b", "wide", 1, 1024, 4096)
    add("aspect", "llama-3.1-8b", "tall", 1, 4096, 1024)
    add("aspect", "llama-3.1-8b", "square", 1, 2048, 2048)

    # attention floor (conv proxy, single head, 8B hd=128): QK^T (K=hd,N=kv) + SV (K=kv,N=hd)
    for kvlen in [129, 513, 1025]:
        add("attn_floor", "llama-3.1-8b", "qkT", 1, 128, kvlen)
        add("attn_floor", "llama-3.1-8b", "sv", 1, kvlen, 128)
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=5)
    ap.add_argument("--dataset-len", type=int, default=20)
    ap.add_argument("--only", default=None, help="run only this group")
    args = ap.parse_args()

    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    tasks = manifest()
    if args.only:
        tasks = [t for t in tasks if t[1] == args.only]
    todo = [t for t in tasks if t[0] not in results]
    log(f"=== run_metis_cim: {len(tasks)} tasks, {len(todo)} to do (skip {len(tasks)-len(todo)} done) ===")
    for i, (tid, group, p) in enumerate(todo):
        r = measure_op(p["M"], p["K"], p["N"], p["bias"], p["residency"],
                       seconds=args.seconds, dataset_len=args.dataset_len)
        results[tid] = {**p, "group": group, **r}
        RESULTS.write_text(json.dumps(results, indent=1))
        tag = r.get("error", f"dev={r.get('dev_fps',0):.0f} sys={r.get('system_fps',0):.0f}fps")
        log(f"[{i+1}/{len(todo)}] {group}/{p['model']}/{p['family']} M{p['M']}K{p['K']}N{p['N']} {p['residency'] or ''} -> {tag}")
    log(f"=== DONE: {len(results)} total results ===")


if __name__ == "__main__":
    main()
