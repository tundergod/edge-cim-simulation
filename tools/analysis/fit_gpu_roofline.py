"""Phase 1.2 — calibrate the Mali-G610 analytic ROOFLINE slot (D4, WP-GPU).

Reads  measurements/aetina/mali_matmul.json  (FP16 micro-benchmark, Phase 1.1)
Writes simulator/specs/gpu_mali_g610.json     ('roofline_fit' block, calibrated)
       validation/reports/phase1.2/m4_gpu_roofline.json
       (figure: tools/plotting/gpu_g1.py -> docs/figures/phase1.2/G1.png)

Two FP16-calibrated axes (shape-trend fit, NOT a strict lower bound -- ~1/3 of points over-predict by
up to +5%; the micro-benchmark model stays PRIMARY, this roofline is the model-swap slot):
  - eff_compute_fp16 = saturated f16 GFLOP/s (ksweep, 5 pts) / fp16 peak (1024) -> compute ceiling
  - mem_eff_BW_GBs   = lstsq fit of FP16 decode-GEMV latency vs nbytes (line through origin)
Records error vs the 1.1 measured points. HONESTY: FP16 only (INT8 = zero data);
5 saturation points -> not transferable; a shape-trend fit, mostly (~2/3) <= measured but NOT a
strict lower bound (~1/3 over-predict by up to +5%).

Run: ./.venv/bin/python tools/analysis/fit_gpu_roofline.py
"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from simulator.models.m4_gpu_roofline import GpuRooflineModel  # noqa: E402

AET = ROOT / "measurements/aetina"
SPEC = ROOT / "simulator/specs/gpu_mali_g610.json"
REPORT = ROOT / "validation/reports/phase1.2/m4_gpu_roofline.json"


def _nbytes_fp16(M, K, N):
    return (K * N + M * K + M * N) * 2     # weight + act-in + act-out, fp16


def main():
    res = json.loads((AET / "mali_matmul.json").read_text())["results"]
    spec = json.loads(SPEC.read_text())

    # --- compute ceiling: saturated f16 throughput (ksweep, the 5 calibration pts) ---
    ksweep = [(r["M"], r["f16_gflops"]) for r in res if r["group"] == "ksweep"]
    # converged tail = highest-M ksweep pt (~20.12 at M=1024), NOT the intermediate M=512 peak (20.29):
    # throughput saturates by M>=128 and 20.12 is the converged value the spec records as
    # measured_fp16_gflops; the M=512 bump is a transient measurement peak, so the tail is the
    # spec-consistent (slightly conservative) ceiling.
    g_sat = max(ksweep, key=lambda mg: mg[0])[1]
    eff_compute = g_sat / spec["fp16_peak_gflops"]

    # --- memory BW: lstsq fit of FP16 decode-GEMV latency vs nbytes (line through origin) ---
    dec = [r for r in res if r["group"] == "proj_decode"]
    nb = np.array([_nbytes_fp16(r["M"], r["K"], r["N"]) for r in dec], float)
    lat = np.array([r["f16_ms"] * 1e3 for r in dec], float)  # ms -> us
    slope, _, _, _ = np.linalg.lstsq(nb.reshape(-1, 1), lat, rcond=None)  # us/byte
    mem_eff_BW_GBs = float(1.0 / slope[0] / 1e3)            # bytes/us -> GB/s

    fit = {
        "_doc": "CALIBRATED roofline slot (FP16, shape-trend fit, not transferable). "
                "eff_compute_fp16 = converged f16 GFLOP/s / fp16 peak; mem_eff_BW_GBs = "
                "lstsq fit to FP16 decode-GEMV. INT8 = zero data. NOT a strict lower bound "
                "(~1/3 of points over-predict by up to +5%; see frac_pred_le_measured).",
        "eff_compute_fp16": round(float(eff_compute), 5),
        "saturated_f16_gflops": round(float(g_sat), 2),
        "mem_eff_BW_GBs": round(mem_eff_BW_GBs, 3),
        "calibration": "mali_matmul.json FP16 (ksweep 5 pts + 16 decode-GEMV pts)",
        "honesty": "shape-trend fit, FP16 only; mostly <= measured but ~1/3 over-predict (NOT a strict lower bound)",
    }
    spec["roofline_fit"] = fit
    SPEC.write_text(json.dumps(spec, indent=2) + "\n")

    # --- error vs ALL 1.1 measured points (compute- and memory-bound) ---
    m = GpuRooflineModel(spec)
    from simulator.models.engine import Workload
    pts, relerr = [], []
    for r in res:
        meas = r["f16_ms"] * 1e3
        wl = Workload(op="gemm", M=r["M"], K=r["K"], N=r["N"], dtype="fp16")
        out = m.predict(wl)
        re = (out["latency_us"] - meas) / meas              # signed; <0 = roofline below measured
        pts.append({"group": r["group"], "tag": r["tag"], "M": r["M"], "K": r["K"], "N": r["N"],
                    "meas_us": round(meas, 1), "pred_us": round(out["latency_us"], 1),
                    "bound": out["bound"], "rel_err": round(re, 3)})
        relerr.append(re)
    abs_re = [abs(x) for x in relerr]

    report = {
        "module": "m4_gpu_roofline",
        "role": "swap slot (analytic roofline); micro-benchmark (m4_gpu) = PRIMARY",
        "equation": {"latency_us": "max(compute_us, memory_us)",
                     "compute_us": "2*M*K*N / (eff_compute_fp16 * fp16_peak)",
                     "memory_us": "nbytes / mem_eff_BW_GBs",
                     "nbytes_default": "(K*N + M*K + M*N) * bytes_per_elem"},
        "calibrated_fit": fit,
        "error_vs_1p1_measured": {
            "n_points": len(pts),
            "median_abs_relerr": round(float(statistics.median(abs_re)), 3),
            "p95_abs_relerr": round(float(np.percentile(abs_re, 95)), 3),
            "max_abs_relerr": round(float(max(abs_re)), 3),
            "frac_within_5pct": round(float(np.mean([abs(r) <= 0.05 for r in relerr])), 3),
            "frac_pred_le_measured": round(float(np.mean([r <= 0.0 for r in relerr])), 3),
            "note": "frac_pred_le_measured = signed rel_err <= 0 (roofline at/below measured) -- this is "
                    "NOT 1.0: proj_decode shapes over-predict up to +5%, so the roofline is a shape-trend "
                    "fit, mostly (~2/3) <= measured, not a strict lower bound. frac_within_5pct = |rel_err|<=0.05.",
            "per_point": pts,
        },
        "honesty": {
            "int8": "ZERO INT8 GPU GEMM data; fit + predict are FP16 only.",
            "measured_point": "Phase 1.1 20.12 GFLOP/s = FP16, NOT INT8.",
            "fp32_peak_512": "spec assumption (may underestimate 2-4x); this model calibrates "
                             "against FP16, so it does not depend on the FP32 peak.",
            "roofline": "shape-trend fit (FP16); mostly <= measured but ~1/3 over-predict by up to "
                        "+5% (NOT a strict lower bound); 5 saturation pts -> NOT transferable "
                        "calibration. No numeric acceptance gate (no INT8 silicon).",
            "ksweep_saturation_M": "DEAD param in spec (kept, not deleted, per audit); unused here.",
            "primary_vs_slot": "micro-benchmark (m4_gpu.py) PRIMARY; this roofline = model-swap slot.",
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1) + "\n")

    e = report["error_vs_1p1_measured"]
    print(f"GPU roofline: ceil={m.ceil_gflops:.2f} GFLOP/s (eff_compute={eff_compute:.4f} vs "
          f"{spec['fp16_peak_gflops']} peak) | mem BW={mem_eff_BW_GBs:.3f} GB/s")
    print(f"  error vs 1.1 ({e['n_points']} pts): median|re|={e['median_abs_relerr']} "
          f"p95={e['p95_abs_relerr']} max={e['max_abs_relerr']} (shape-trend; pred<=meas {e['frac_pred_le_measured']})")


if __name__ == "__main__":
    main()
