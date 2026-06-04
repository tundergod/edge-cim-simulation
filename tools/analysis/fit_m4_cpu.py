"""Phase 1 — fit M4 CPU (A76) non-GEMM support ops.

Reads  measurements/aetina/cpu_ops.json
Writes simulator/models/params/m4_cpu.json
       validation/reports/m4_cpu.json

softmax = linear-in-kv fit per (model,dtype) (3 pts kv in {128,512,1024}); other ops =
per-(model,dtype) constants (no within-op sweep). fp16 = emulated upper bound.

Run: ./.venv/bin/python tools/analysis/fit_m4_cpu.py
"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from simulator.models.m4_cpu import CpuModel  # noqa: E402

AET = ROOT / "measurements/aetina"
CONST_OPS = ["rmsnorm", "rope_apply", "residual", "swiglu", "sampling_argmax"]


def main():
    ops = json.loads((AET / "cpu_ops.json").read_text())["ops"]

    const = {op: {} for op in CONST_OPS}
    softmax_pts = {}  # (model,dtype) -> {kv: us}
    for v in ops.values():
        op, model, dtype, med = v["op"], v["model"], v["dtype"], v["median_us"]
        if op in CONST_OPS:
            const[op].setdefault(model, {})[dtype] = round(med, 2)
        elif op.startswith("softmax_kv"):
            kv = int(op[len("softmax_kv"):])
            softmax_pts.setdefault((model, dtype), {})[kv] = med

    # linear fit softmax per (model,dtype); collect fit errors
    softmax_linear, sm_err = {}, []
    for (model, dtype), pts in softmax_pts.items():
        kvs = sorted(pts)
        b, a = np.polyfit(kvs, [pts[k] for k in kvs], 1)
        softmax_linear.setdefault(model, {})[dtype] = {"a": round(float(a), 2), "b": round(float(b), 4)}
        for kv in kvs:
            sm_err.append(abs((a + b * kv) - pts[kv]) / pts[kv])

    params = {"_doc": "M4 CPU A76 non-GEMM. softmax=linear-in-kv per (model,dtype); others "
                      "constants. fp16=emulated UPPER BOUND (provenance phase0.3-findings).",
              "const_us": const, "softmax_linear": softmax_linear}
    (ROOT / "simulator/models/params/m4_cpu.json").write_text(json.dumps(params, indent=1))
    m = CpuModel(params)

    # sanity: softmax increases with kv; rmsnorm increases with model hidden (observation, not law)
    sm_mono = all(
        m.op_us("softmax", mdl, "fp16", 128) <= m.op_us("softmax", mdl, "fp16", 512)
        <= m.op_us("softmax", mdl, "fp16", 1024)
        for mdl in softmax_linear)

    report = {
        "module": "m4_cpu",
        "equation": {"softmax": "us = a + b*kv per (model,dtype)",
                     "other": "per-(model,dtype) constant (no within-op sweep)"},
        "softmax_fit_gate": {
            "n": len(sm_err), "median": round(float(statistics.median(sm_err)), 3),
            "p95": round(float(np.percentile(sm_err, 95)), 3), "max": round(float(max(sm_err)), 3),
            "pass_median_le_0.10": bool(statistics.median(sm_err) <= 0.10),
            "pass_p95_le_0.20": bool(float(np.percentile(sm_err, 95)) <= 0.20)},
        "const_ops": list(const),
        "const_us": const,
        "softmax_linear": softmax_linear,
        "sanity": {"softmax_monotonic_in_kv": sm_mono,
                   "fp16_is_upper_bound": "numpy-emulated on A76 (provenance: phase0.3-findings, "
                                          "not a cpu_ops.json field)"},
        "notes": {"prefill": "decode (1-token) costs; prefill = x S tokens analytic, UNVALIDATED",
                  "issue_10": "measured latencies used, NOT analytic 1-flop/elem FLOPs"},
    }
    (ROOT / "validation/reports/m4_cpu.json").write_text(json.dumps(report, indent=1))
    g = report["softmax_fit_gate"]
    print(f"M4-CPU: softmax linear fit n={g['n']} median={g['median']} p95={g['p95']} "
          f"max={g['max']} PASS={g['pass_median_le_0.10'] and g['pass_p95_le_0.20']}")
    print(f"  const ops: {CONST_OPS}; softmax monotonic={sm_mono}")


if __name__ == "__main__":
    main()
