"""Phase 1.2 — CIM re-validation on the production Metis Card (same 800MHz quad-core AIPU).

Port of characterization/aetina/run_metis_cim.py (Alpha) to the Card (Voyager v1.6). The CIM
COMPUTE kernel is NOT frozen: the same AIPU is alive on the Card, so we re-measure the 1x1-conv
matmul proxy with axrunmodel (dev FPS = isolated compute) and cross-check vs the Alpha 13 points
(both boards 800MHz -> directly comparable, NO clock rescale). We also add the prefill /
compute-bound shapes Alpha could not reach.

STATUS: SPIKE-VERIFIED ON CARD (2026-06-06). The kernel re-validates: square M=1,K=2048,N=2048 =
Card 200.5 vs Alpha 203.7 GOP/s (1.5%, the committed full-run value). The v1.6 axcompile compiles a single ~2048x2048 tile only up
to M<=256 (SRAM L1/L2 tiling wall, NOT device-DRAM — the Card's 16GiB does not lift it); larger
prefill GEMMs are therefore measured tile-by-tile and extrapolated (n_tiles x tile_lat, the Alpha
run_metis_cim.py method; see measure_op).

PHASE 1.5 supplementary families (the M-amortization was fit on only 3 points / 1 shape): the M_MAX
and SAFE_KN walls are now PROBED, not assumed. `prefill_msweep` densifies M + a 1<M<64 no-bridge band
+ a wall-pin sweep, measured DIRECTLY on the canonical tile (so M>M_MAX genuinely tests the wall, not
the M_MAX short-circuit). `prefill_shapes` adds tile shapes (down_proj/lm_head/...) to test the
(a,b)-vs-(K,N) factorization. `mtile` serves M>256 via measure_m_chunks (m_tiled_chunked). `envelope_probe`
+ `multitile` attempt NATIVE multi-tile compile to recalibrate the +36% tile-sum over-prediction.
`kv_proxy` is the KV-cache isolation SPIKE (measure_mem, K=1 memory-bound proxy). All probe families are
fail-tolerant: a compile failure records an error/`compiles_native:false`, never aborts the run.
Run (AXCOMPILE = devkit binary, python needs onnx + axrunmodel on PATH):
    rsync -a characterization/metis_card/ metiscard:~/edge-cim-simulation/characterization/
    ssh metiscard 'cd ~/edge-cim-simulation/characterization && AXCOMPILE=/tmp/axc/bin/axcompile \
        PATH=/tmp/axc/bin:$HOME/tundergod/voyager-sdk/axelera-env/bin:$PATH \
        /tmp/axc/bin/python run_metis_cim_v16.py --spike'
    # then full:  ... run_metis_cim_v16.py   (or scoped: --only <group>)
    # NB: axrunmodel lives in the SDK's axelera-env/bin (the venv was renamed from venv/ -> axelera-env/);
    #     it MUST be on PATH or the compile succeeds but the run dies with FileNotFoundError: axrunmodel.
    rsync -a metiscard:~/edge-cim-simulation/measurements/metis_card/ measurements/metis_card/

COMPILE PATH (SPIKE-confirmed 2026-06-06): v1.6 ships the compiler as `axcompile` (axelera-devkit
wheel from Artifactory, officially Beta) — the old `compile` was just renamed, NOT removed. The
1×1-conv proxy compiles cleanly via `axcompile --input … --input-shape 1,K,1,M --output … --overwrite
--dataset-len N` (auto-generated calibration, no imageset). Raw MatMul/Gemm still fails
(ONNXGraphCleanerError: not topologically sorted) → the conv proxy is required. Set AXCOMPILE to the
devkit venv's binary (e.g. /tmp/axc/bin/axcompile) if it's not on PATH; the runtime `axrunmodel`
lives in the SDK env. If neither `axcompile` nor `compile` is present, report COMPILER_ABSENT.

Run: AXCOMPILE=/path/to/axcompile python run_metis_cim_v16.py [--spike] [--seconds 5] [--dataset-len 20] [--only GROUP]
"""
import argparse, json, os, re, shutil, subprocess, time
from pathlib import Path

AXCOMPILE = os.environ.get("AXCOMPILE", "axcompile")  # v1.6 devkit compiler; old `compile` is the fallback
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

WORK = Path("/tmp/cim_card_work"); WORK.mkdir(parents=True, exist_ok=True)
SDK = str(Path.home() / "tundergod" / "voyager-sdk")          # Card SDK (Alpha was /home/ubuntu/voyager-sdk)
RESULTS = WORK / "results.json"
LOG = WORK / "progress.log"
FPS_RE = re.compile(r"dev:([\d.]+)\s+host:([\d.]+)\s+system:([\d.]+)")  # axrunmodel: "dev:.. host:.. system:.. latency:..ms" (no literal 'fps')

# The 13 Alpha native single-tile throughput points (M=1) — the cross-validation ground truth
# (mirrors simulator/models/params/m1_cim.json native_throughput_points). dev_gflops here must
# match these (same AIPU, same 800MHz) to confirm the kernel re-validates.
ALPHA_13 = [(64,2048),(128,2048),(256,2048),(480,2048),(512,2048),(512,3584),(544,2048),
            (1000,2048),(1024,2048),(1024,3072),(1024,4096),(1536,2048),(2048,2048)]  # (N,K)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def build_conv_onnx(M, K, N, bias, path):
    """1x1 conv == [M,K]x[K,N] matmul (weight-stationary). Identical to the Alpha proxy."""
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


def _try_low_level_compile(d, M, K, residency, dataset_len, timeout):
    """Compile the conv proxy: v1.6 `axcompile` (axelera-devkit) first, old `compile` as fallback.
    Returns (model_json|None, info). Both absent => COMPILER_ABSENT. model.json lands at
    out/compiled_model/model.json (rglob finds it)."""
    for binary, tag in ((AXCOMPILE, "axcompile"), ("compile", "compile")):
        cmd = [binary, "--input", str(d / "m.onnx"), "--input-shape", f"1,{K},1,{M}",
               "--output", str(d / "out"), "--overwrite", "--log-level", "WARNING",
               "--dataset-len", str(dataset_len)]
        if residency:   # Alpha-era l2/ddr residency; not exercised by the Card alpha13/prefill manifest
            (d / "cfg.json").write_text(json.dumps({"dpu_constants_home": f"global.{residency}"}))
            cmd += ["--config", str(d / "cfg.json")]
        try:
            rc = subprocess.run(cmd, cwd=SDK, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError:
            continue   # this binary not present; try the next
        if rc.returncode != 0:
            return None, {"compile_path": tag, "error": "compile", "rc": rc.returncode, "stderr": rc.stderr[-300:]}
        mj = list((d / "out").rglob("model.json"))
        return (mj[0] if mj else None), {"compile_path": tag}
    return None, {"compile_path": "COMPILER_ABSENT"}   # neither axcompile nor compile present


def measure(M, K, N, bias=False, residency=None, seconds=5, dataset_len=20, timeout=900):
    """Compile (low-level `compile`, falling back to a documented STOP) + axrunmodel one shape."""
    d = WORK / "cur"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    try:
        build_conv_onnx(M, K, N, bias, d / "m.onnx")
        t0 = time.time()
        mj, info = _try_low_level_compile(d, M, K, residency, dataset_len, timeout)
        ct = time.time() - t0
        if info.get("compile_path") == "COMPILER_ABSENT":
            # Neither axcompile (devkit) nor compile present -> install axelera-devkit or set AXCOMPILE.
            return {"error": "compiler_absent",
                    "spike": "neither `axcompile` (axelera-devkit) nor `compile` found. Install the "
                             "devkit (pip --extra-index-url …/axelera-pypi/simple 'axelera-devkit[all]') "
                             "or set AXCOMPILE=/path/to/axcompile. Report user.", "compile_s": round(ct, 1)}
        if mj is None:
            return {**info, "error": info.get("error", "no_model_json"), "compile_s": round(ct, 1)}
        ax = subprocess.run(["axrunmodel", str(mj), "--seconds", str(seconds)],
                            cwd=SDK, capture_output=True, text=True, timeout=180)
        m = FPS_RE.search(ax.stdout)
        if not m:
            return {**info, "error": "no_fps", "stdout": ax.stdout[-200:], "compile_s": round(ct, 1)}
        dev, host, sys_ = float(m.group(1)), float(m.group(2)), float(m.group(3))
        flops = 2 * M * K * N
        return {**info, "dev_fps": dev, "host_fps": host, "system_fps": sys_,
                "dev_lat_us": 1e6 / dev, "system_lat_us": 1e6 / sys_,
                "dev_gflops": flops * dev / 1e9, "compile_s": round(ct, 1)}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)


def measure_mem(M, N, seconds=5, dataset_len=20):
    """Phase 1.5 Axis E — KV-cache isolation SPIKE. A K=1 conv is a MEMORY-BOUND proxy (arithmetic
    intensity ~2 MAC/byte: dev time ~ streaming N*M output bytes, not compute). Returns eff_BW_GBs
    + compute_negligible so we can ask: does AIPU memory-bound traffic hit the M2 streaming BW the
    analytic kv_append assumes? Compilable-size proxy; the kv_bytes mapping is applied downstream."""
    r = measure(M, 1, N, seconds=seconds, dataset_len=dataset_len)
    if "error" in r:
        return r
    bytes_moved = (1 * M) + (N * 1) + (N * M)        # in + weight + out, INT8 (1 byte/elem)
    flops = 2 * M * 1 * N
    r["bytes_moved"] = bytes_moved
    r["eff_BW_GBs"] = bytes_moved / (r["dev_lat_us"] * 1e-6) / 1e9
    r["arith_intensity"] = flops / bytes_moved
    r["compute_negligible"] = bool(flops / bytes_moved <= 4.0)   # memory-bound if intensity small
    r["tiles"] = 1
    return r


SAFE_KN = 2048 * 2048      # 4_194_304 — v1.6 axcompile compiles a single ~2048x2048 tile (at M<=M_MAX)
M_MAX = 256                # M>256 fails to compile ANY tile (SRAM L1/L2 wall, spike-confirmed) -> prefill cap
K_TILE = N_TILE = 2048     # AIPU native tile (matches m1_cim native_max_kn); big GEMMs = n_tiles x tile
_TILE_CACHE = {}           # canonical-tile measurement per M (the prefill M-amortization data)


def _ntiles(total, size):
    return (total + size - 1) // size


def measure_op(M, K, N, seconds=5, dataset_len=20):
    """Direct compile when M<=M_MAX and K*N<=SAFE_KN; else tile into canonical K_TILE x N_TILE blocks
    (measured once per M, cached) and extrapolate dev_lat = n_tiles * tile_lat (Alpha run_metis_cim.py
    method). M>M_MAX can't compile any tile (v1.6 SRAM wall) -> error: prefill is measurable to M_MAX,
    larger M is extrapolated analytically downstream. The cached tile throughput per M IS the
    prefill M-amortization the decode-only Alpha 13 points could not capture."""
    if M > M_MAX:
        return {"error": "M_exceeds_compile_limit",
                "note": f"M={M}>{M_MAX}: no tile compiles on v1.6 axcompile (SRAM L1/L2)."}
    if K * N <= SAFE_KN:
        r = measure(M, K, N, seconds=seconds, dataset_len=dataset_len)
        if "error" not in r:
            r["tiles"] = 1
        return r
    if M not in _TILE_CACHE:
        _TILE_CACHE[M] = measure(M, K_TILE, N_TILE, seconds=seconds, dataset_len=dataset_len)
    t = _TILE_CACHE[M]
    if "error" in t:
        return {"error": f"canonical_tile_fail: {t['error']}"}
    nt = _ntiles(K, K_TILE) * _ntiles(N, N_TILE)
    dev_lat, flops = nt * t["dev_lat_us"], 2 * M * K * N
    return {"compile_path": t.get("compile_path"), "tiles": nt, "tiled_extrapolated": True,
            "dev_lat_us": dev_lat, "system_lat_us": nt * t["system_lat_us"], "dev_gflops": flops / dev_lat / 1e3,
            "tile_dev_lat_us": t["dev_lat_us"], "tile_dev_gflops": t["dev_gflops"], "tile_dev_fps": t["dev_fps"]}


def measure_m_chunks(M_eff, K, N, chunk=256, seconds=5, dataset_len=20):
    """Phase 1.5 Axis C — M-axis tiling for M>M_MAX. A prefill of M_eff tokens is served as
    ceil(M_eff/chunk) resident-model inferences of the compilable chunk (M<=M_MAX). axrunmodel's
    steady-state FPS IS back-to-back execution of the resident model, so the chunked total =
    n * per-chunk latency; per_chunk_overhead_us = system-dev (host/DMA paid per inference). This is
    m_tiled_chunked — NOT a fused large-M compile (which the SRAM wall forbids)."""
    base = measure_op(chunk, K, N, seconds=seconds, dataset_len=dataset_len)
    if "error" in base:
        return {"error": f"chunk_fail: {base['error']}", "M_eff": M_eff, "chunk": chunk}
    n = _ntiles(M_eff, chunk)
    dev, sysl = base["dev_lat_us"], base["system_lat_us"]
    return {"m_tiled_chunked": True, "M_eff": M_eff, "chunk": chunk, "n_chunks": n,
            "tiles": base.get("tiles"), "compile_path": base.get("compile_path"),
            "chunk_dev_lat_us": dev, "chunk_system_lat_us": sysl,
            "total_dev_lat_us": n * dev, "total_system_lat_us": n * sysl,
            "per_chunk_overhead_us": sysl - dev,
            "note": "n resident-model inferences (axrunmodel steady-state); total = n x per-chunk; "
                    "NOT a fused large-M compile"}


def manifest(spike):
    """Card tasks: the 13 Alpha cross-validation shapes (M=1, direct) + prefill GEMMs (tiled).
    Prefill GEMMs exceed SAFE_KN -> measure_op tiles them; the cached tile throughput at the
    compilable M in {64,128,256} is the new M-amortization fit data (M>256 fails to compile).
    --spike trims to a fast feasibility set (the 203.7 anchor + one tiled prefill GEMM)."""
    tasks = []

    def add(group, family, M, K, N):
        tid = f"{group}|{family}|M{M}|K{K}|N{N}"
        tasks.append((tid, group, dict(family=family, M=M, K=K, N=N)))

    if spike:
        add("alpha13", "native", 1, 2048, 2048)        # the canonical tile @ M=1 (203.7 GOP/s anchor)
        add("prefill", "gate_up", 128, 4096, 14336)    # exercises measure_op tiling (tile @ M=128 x 14 tiles)
        return tasks

    # cross-validation: re-measure the 13 Alpha native single-tile points (M=1; all K*N<=SAFE_KN -> direct)
    for (N, K) in ALPHA_13:
        add("alpha13", "native", 1, K, N)
    # PREFILL M-amortization (the Alpha gap): Llama-3-8B FFN/attn GEMMs, tiled (n_tiles x tile @ M).
    # The tile throughput at M in {64,128,256} is the fit data; M>256 fails to compile (v1.6 SRAM wall).
    c8 = dict(H=4096, F=14336)
    for M in [64, 128, 256]:
        add("prefill", "gate_up", M, c8["H"], c8["F"])   # 14 tiles
    for M in [128, 256]:
        add("prefill", "q_o", M, c8["H"], c8["H"])       # 4 tiles (a 2nd shape at the same tile throughput)

    # ===== Phase 1.5 supplementary axes (the M-amortization is thin; densify + probe walls) =====
    # A — prefill M-sweep densify + no-bridge band (1<M<64) + compile-wall pin. Canonical 2048x2048
    #     tile measured DIRECTLY (group routed to measure(), NOT measure_op) so M>M_MAX genuinely
    #     PROBES the compile wall instead of short-circuiting on M_MAX. Fail-tolerant.
    for M in [2, 4, 8, 16, 32, 48, 96, 192, 224,
              248, 252, 254, 255, 256, 257, 258, 260, 264, 272, 288, 320,
              384, 448, 512, 640, 768, 1024, 1536, 2048, 3072, 4096]:   # probe the REAL M ceiling
        add("prefill_msweep", "tile", M, 2048, 2048)
    # B — more tile SHAPES @ M in {64,128,256}: does (a,b) depend on (K,N) or only tile-count?
    for fam, K, N in [("down_proj", 14336, 4096), ("lm_head", 4096, 128256),
                      ("small_k", 2048, 14336), ("large_k", 14336, 14336),
                      ("gate_up_ctrl", 4096, 14336)]:
        for M in [64, 128, 256]:
            add("prefill_shapes", fam, M, K, N)
    # C — M-axis chunked tiling for M>256 (gate_up; measure_m_chunks; m_tiled_chunked)
    for M_eff in [512, 1024, 2048]:
        add("mtile", "gate_up", M_eff, c8["H"], c8["F"])
    # D.1 — native compile ENVELOPE probe (M=1, measured DIRECTLY past SAFE_KN, fail-tolerant):
    #       find what gates native multi-tile compile (K? N? K*N?).
    for (K, N) in [(2048, 2304), (2048, 2560), (2048, 3072), (2048, 4096), (2304, 2048),
                   (2560, 2048), (3072, 2048), (4096, 2048), (2560, 2560), (3072, 3072),
                   (2048, 8192), (8192, 2048)]:
        add("envelope_probe", "native", 1, K, N)
    # D.1b — CLIFF map: native M=1 throughput collapses ~3.5x (≈220-250 -> ~70 GOP/s) past a knee at
    #        ~2 tiles' work (K*N ≈ 6.5-8.4M, envelope_probe finding). Densify to pin the knee.
    for (K, N) in [(2048, 3328), (2048, 3584), (2048, 3840), (2304, 3072),
                   (2560, 2816), (2816, 2816), (3072, 2560), (3328, 2048)]:
        add("cliff_map", "native", 1, K, N)
    # D.2 — native MULTI-TILE measure (decode M=1 + prefill M>=64; fail-tolerant). Compared vs the
    #       tile-sum model downstream to recalibrate the +36% (or kept extrapolated if uncompilable).
    for (K, N) in [(1024, 4096), (2048, 3072), (3072, 2048)]:
        add("multitile", "decode", 1, K, N)
    for M in [64, 128]:
        add("multitile", "prefill", M, 2048, 3072)
    # E — KV-cache isolation SPIKE: memory-bound K=1 conv proxy (measure_mem -> eff_BW). Transfer-size
    # sweep (bytes ~ N*M) to demonstrate eff_BW CONVERGES to a sustained plateau (3 rising points is not
    # convergence evidence). N=2048 (single tile, no N-tiling); grow M to amortize per-inference overhead.
    for M in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        add("kv_proxy", "mem", M, 1, 2048)
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spike", action="store_true", help="fast feasibility probe (~4 shapes)")
    ap.add_argument("--seconds", type=int, default=5)
    ap.add_argument("--dataset-len", type=int, default=20)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    tasks = manifest(args.spike)
    if args.only:
        tasks = [t for t in tasks if t[1] == args.only]
    todo = [t for t in tasks if t[0] not in results]
    log(f"=== run_metis_cim_v16 ({'SPIKE' if args.spike else 'FULL'}): {len(tasks)} tasks, "
        f"{len(todo)} to do; SDK={SDK} ===")
    spike_absent = False
    for i, (tid, group, p) in enumerate(todo):
        # Dispatch by group: native attempt (probes the compile wall) / M-chunk / mem-proxy / tiled.
        if group in ("prefill_msweep", "envelope_probe", "cliff_map", "multitile"):
            r = measure(p["M"], p["K"], p["N"], seconds=args.seconds, dataset_len=args.dataset_len)
            if "error" not in r:
                r["tiles"] = 1
            if group in ("envelope_probe", "cliff_map", "multitile"):
                r["compiles_native"] = "error" not in r
        elif group == "mtile":
            r = measure_m_chunks(p["M"], p["K"], p["N"], seconds=args.seconds, dataset_len=args.dataset_len)
        elif group == "kv_proxy":
            r = measure_mem(p["M"], p["N"], seconds=args.seconds, dataset_len=args.dataset_len)
        else:
            r = measure_op(p["M"], p["K"], p["N"], seconds=args.seconds, dataset_len=args.dataset_len)
        results[tid] = {**p, "group": group, **r}
        RESULTS.write_text(json.dumps(results, indent=1))
        if r.get("error") == "compiler_absent" and not spike_absent:
            spike_absent = True
            log("!!! SPIKE VERDICT: neither axcompile nor compile present. " + r["spike"])
        if "error" in r:
            tag = r["error"]
        elif r.get("m_tiled_chunked"):
            tag = (f"M{p['M']}={r['n_chunks']}x chunk {r['chunk_dev_lat_us']:.0f}us -> "
                   f"{r['total_dev_lat_us']:.0f}us dev (+{r['per_chunk_overhead_us']:.0f}us/chunk ovhd)")
        elif group == "kv_proxy":
            tag = f"mem {r.get('eff_BW_GBs',0):.1f}GB/s intensity={r.get('arith_intensity',0):.1f} [{r.get('compile_path','?')}]"
        elif r.get("tiled_extrapolated"):
            tag = (f"TILED x{r['tiles']} tile={r['tile_dev_fps']:.0f}fps -> full {r['dev_gflops']:.0f}GOP/s "
                   f"{r['dev_lat_us']:.0f}us [{r.get('compile_path','?')}]")
        else:
            tag = f"dev={r.get('dev_fps',0):.0f}fps {r.get('dev_gflops',0):.1f}GOP/s [{r.get('compile_path','?')}]"
        log(f"[{i+1}/{len(todo)}] {group}/{p['family']} M{p['M']}K{p['K']}N{p['N']} -> {tag}")
    out = Path.home() / "edge-cim-simulation" / "measurements" / "metis_card"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cim_card_revalidate_raw.json").write_text(json.dumps(results, indent=1))
    log(f"=== DONE: {len(results)} results -> {out/'cim_card_revalidate_raw.json'} ===")


if __name__ == "__main__":
    main()
