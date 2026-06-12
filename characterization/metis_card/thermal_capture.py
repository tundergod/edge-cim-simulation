#!/usr/bin/env python3
"""Phase 1.7 — Metis Card thermal + perf-vs-temperature capture harness (runs ON the board).

Decoupled from LLM: drives a compiled 1x1-conv == matmul proxy via `axrunmodel <model.json>
--seconds S --aipu-cores K` (synthetic compute stress). axrunmodel prints ONE summary line per run:
  `<model>,dev:<fps> host:<fps> system:<fps> latency:<ms> frames:<n> temp:<C>C`
so each burst yields a clean (dev throughput, end core temperature) point. Back-to-back bursts heat
the die -> temperature(time) heating curve; throughput-vs-temperature -> throttle/perf coupling.

SAFETY (Phase 1.7 plan; shared board, verified effectively single-user at run time):
  - per-burst end temperature is parsed; the campaign ABORTS if temp >= ABORT_CAP. Bursts are short
    (<=20 s) and the die rises only ~1 C/min, so a single burst physically cannot overshoot the cap
    between checks. axrunmodel timeout is a hard backstop.
  - host load is logged per burst (contention discard happens in analysis).

Usage (on board):
  cd ~/tundergod/voyager-sdk && source axelera-env/bin/activate
  export PATH=/tmp/axc/bin:$PATH AXCOMPILE=/tmp/axc/bin/axcompile
  python <this> --out /tmp/thermal.json --phase heat --secs 15
"""
import argparse, json, os, re, subprocess, time
from pathlib import Path

AXCOMPILE = os.environ.get("AXCOMPILE", "axcompile")
LINE_RE = re.compile(r"dev:([\d.]+)\s+host:([\d.]+)\s+system:([\d.]+)\s+latency:([\d.]+)ms"
                     r"\s+frames:(\d+)\s+temp:(\d+)C")
ABORT_CAP = float(os.environ.get("ABORT_CAP", "55"))    # C — stop campaign at/above this end-temp


def build_conv_onnx(M, K, N, path):
    """1x1 conv == [M,K]x[K,N] matmul (weight-stationary), identical to the Alpha/Card proxy."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    W = numpy_helper.from_array(np.random.randn(N, K, 1, 1).astype(np.float32), "W")
    node = helper.make_node("Conv", ["input", "W"], ["output"], kernel_shape=[1, 1],
                            strides=[1, 1], pads=[0, 0, 0, 0])
    g = helper.make_graph([node], "gemm",
                          [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, K, 1, M])],
                          [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, N, 1, M])],
                          [W])
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]), str(path))


def compile_gemm(K, N, workdir, m_candidates=(256, 192, 128, 64)):
    """Compile the conv proxy, trying descending M until L1 tiling fits. Returns (model.json, M)."""
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    last = ""
    for M in m_candidates:
        onnx_p = workdir / f"gemm_{K}x{N}_m{M}.onnx"
        out_d = workdir / f"build_{K}x{N}_m{M}"
        build_conv_onnx(M, K, N, onnx_p)
        r = subprocess.run([AXCOMPILE, "--input", str(onnx_p), "--input-shape", f"1,{K},1,{M}",
                            "--output", str(out_d), "--overwrite", "--dataset-len", "20"],
                           capture_output=True, text=True, timeout=900)
        mjs = list(out_d.rglob("model.json"))
        if mjs:
            return str(mjs[0]), M
        last = f"M={M}: {(r.stdout + r.stderr)[-300:]}"
    raise RuntimeError(f"compile failed for all M {m_candidates}\n{last}")


def host_load():
    try:
        return round(os.getloadavg()[0], 2)
    except OSError:
        return -1.0


def run_burst(model_json, secs, cores):
    """Run one axrunmodel burst; parse (dev_fps, temp_C) from its summary line."""
    secs = min(secs, 20)
    r = subprocess.run(["axrunmodel", model_json, "--seconds", str(secs),
                        "--aipu-cores", str(cores)],
                       capture_output=True, text=True, timeout=secs + 60)
    m = LINE_RE.search(r.stdout + r.stderr)
    if not m:
        return {"secs": secs, "cores": cores, "dev_fps": None, "temp_C": None,
                "host_load": host_load(), "err": (r.stdout + r.stderr)[-200:]}
    return {"secs": secs, "cores": cores, "dev_fps": float(m.group(1)),
            "latency_ms": float(m.group(4)), "frames": int(m.group(5)),
            "temp_C": int(m.group(6)), "host_load": host_load()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", default="heat", choices=["noise", "heat"])
    ap.add_argument("--secs", type=int, default=15)
    ap.add_argument("--max-bursts", type=int, default=40)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--workdir", default="/tmp/thermal_axc")
    args = ap.parse_args()

    out = {"meta": {"abort_cap_C": ABORT_CAP, "phase": args.phase, "cores": args.cores,
                    "secs": args.secs, "host": os.uname().nodename, "t_unix_start": int(time.time())},
           "bursts": []}
    mj, m_used = compile_gemm(2048, 2048, args.workdir)
    out["meta"]["model"], out["meta"]["M"], out["meta"]["KN"] = mj, m_used, "2048x2048"

    t0 = time.monotonic()
    n = 8 if args.phase == "noise" else args.max_bursts
    for i in range(n):
        b = run_burst(mj, args.secs, args.cores)
        b["i"], b["t_elapsed"] = i, round(time.monotonic() - t0, 1)
        out["bursts"].append(b)
        tC = b.get("temp_C")
        print(f"  burst {i:2d} t={b['t_elapsed']:6.1f}s fps={b['dev_fps']} temp={tC}C "
              f"host={b['host_load']}", flush=True)
        if tC is not None and tC >= ABORT_CAP:
            out["meta"]["aborted_at"] = {"i": i, "temp_C": tC}
            print(f"ABORT: temp {tC}C >= cap {ABORT_CAP}C", flush=True)
            break
        if b["dev_fps"] is None:
            out["meta"]["stopped_no_output"] = i
            print("stop: no axrunmodel output", flush=True)
            break

    Path(args.out).write_text(json.dumps(out, indent=1))
    temps = [b["temp_C"] for b in out["bursts"] if b.get("temp_C") is not None]
    print(f"wrote {args.out}: {len(out['bursts'])} bursts, temp {min(temps)}->{max(temps)}C"
          if temps else f"wrote {args.out}: no temps")


if __name__ == "__main__":
    main()
