"""Phase 1 — fit M4 Mali GPU: attention-bmm offload model + GEMM lower-bound trend.

Reads  measurements/aetina/mali_matmul.json (groups: attn, ksweep, proj_decode)
Writes simulator/models/params/m4_gpu.json
       validation/reports/phase1/m4_gpu.json

attn_bmm_us(kv) (single-head QK^T+S.V, f16) is fit linear in kv -> the offload reference.
GEMM absolute throughput is a LOWER BOUND (unoptimised kernel); we record the ksweep
saturation point and the proj_decode trend only (no absolute gate).

Run: ./.venv/bin/python tools/analysis/fit_m4_gpu.py
"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from simulator.models.m4_gpu import MaliGpuModel  # noqa: E402

AET = ROOT / "measurements/aetina"


def main():
    d = json.loads((AET / "mali_matmul.json").read_text())
    res = d["results"]

    # --- attention bmm: combine QK^T + S.V per kv (decode, f16) ---
    qkt = {r["N"]: r["f16_ms"] for r in res if r.get("tag") == "qkT_dec"}   # N = kv+1
    sv = {r["K"]: r["f16_ms"] for r in res if r.get("tag") == "sv_dec"}     # K = kv+1
    kvs, combined = [], []
    for kvp1 in sorted(set(qkt) & set(sv)):
        kv = kvp1 - 1
        kvs.append(kv)
        combined.append((qkt[kvp1] + sv[kvp1]) * 1e3)  # ms -> us, single-head
    b, a = np.polyfit(kvs, combined, 1)                  # us = a + b*kv
    attn_fit_err = [abs((a + b * kv) - c) / c for kv, c in zip(kvs, combined)]

    # --- ksweep: GEMM throughput saturation (f16) ---
    ks = sorted([(r["M"], r["f16_gflops"]) for r in res if r.get("group") == "ksweep"])
    g_sat = ks[-1][1]
    sat_M = next(M for M, g in ks if g >= 0.95 * g_sat)

    params = {
        "_doc": "M4 Mali-G610. attn_bmm_us(kv)=a+b*kv single-head f16 (offload ref). "
                "GEMM absolute = LOWER BOUND (unoptimised kernel).",
        "attn_bmm_a_us": round(float(a), 3),
        "attn_bmm_b_us_per_kv": round(float(b), 5),
        "gemm_gflops_saturated_lowerbound": round(float(g_sat), 2),
        "ksweep_saturation_M": sat_M,
    }
    (ROOT / "simulator/models/params/m4_gpu.json").write_text(json.dumps(params, indent=1))
    m = MaliGpuModel(params)

    # proj_decode trend (informational, lower bound): does FLOPs/g_sat track the trend?
    proj = [r for r in res if r.get("group") == "proj_decode"]
    proj_trend = []
    for r in proj:
        meas = r["f16_ms"] * 1e3
        pred = m.gemm_lat_us(r["M"], r["K"], r["N"])
        proj_trend.append({"tag": r["tag"], "K": r["K"], "N": r["N"],
                           "meas_us": round(meas, 1), "lb_pred_us": round(pred, 1),
                           "note": "GEMV latency-bound; pred is a compute lower bound"})

    report = {
        "module": "m4_gpu",
        "equation": {"attn": "attn_bmm_us = a + b*kv (single-head qkT+sv, f16)",
                     "gemm": "gemm_lat_us = FLOPs/g_sat (LOWER BOUND)"},
        "params": params,
        "attn_offload_gate": {
            "set": "decode single-head attn, kv=%s" % kvs,
            "fit_us": {kv: round(a + b * kv, 1) for kv in kvs},
            "meas_us": {kv: round(c, 1) for kv, c in zip(kvs, combined)},
            "median_relerr": round(float(statistics.median(attn_fit_err)), 3),
            "p95_relerr": round(float(np.percentile(attn_fit_err, 95)), 3),
            "max_relerr": round(float(max(attn_fit_err)), 3),
            "pass_median_le_0.10": bool(statistics.median(attn_fit_err) <= 0.10),
            "pass_p95_le_0.20": bool(float(np.percentile(attn_fit_err, 95)) <= 0.20)},
        "ksweep_saturation": {"g_sat_gflops_f16": round(g_sat, 2), "saturates_at_M": sat_M,
                              "note": "absolute throughput = LOWER BOUND (unoptimised kernel)"},
        "gemm_proj_decode_trend_lowerbound": proj_trend,
        "notes": {"role": "GPU = attention offload; CIM does the projections. attn latency is "
                          "the validated GPU deliverable; GEMM absolute is a lower bound."},
    }
    (ROOT / "validation/reports/phase1/m4_gpu.json").write_text(json.dumps(report, indent=1))
    g = report["attn_offload_gate"]
    print(f"M4-GPU: attn_bmm_us = {params['attn_bmm_a_us']} + {params['attn_bmm_b_us_per_kv']}*kv")
    print(f"  attn offload fit: median={g['median_relerr']} p95={g['p95_relerr']} "
          f"max={g['max_relerr']} PASS={g['pass_median_le_0.10'] and g['pass_p95_le_0.20']}")
    print(f"  ksweep g_sat={g_sat} GFLOP/s (lower bound), saturates at M={sat_M}")


if __name__ == "__main__":
    main()
