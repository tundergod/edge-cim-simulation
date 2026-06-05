"""Phase 1.2 (WP-CPU, D1) — calibrate the CPU instruction-count roofline to fp32 cpu_ops.json.

Solves the CALIBRATED factors of simulator/models/m4_cpu.py's roofline against the single-A76-core,
single-thread, numpy fp32 support-op latencies (measurements/aetina/cpu_ops.json):

    latency_us = max(compute_us, memory_us) + overhead_op
    compute_us = (n_elem * ops_per_elem) / (W*IPC*freq) / eta_c        # eta_c CALIBRATED here
    memory_us  = working_set_bytes / (BW_tier(working_set) * eta_bw)   # eta_bw ASSUMPTION (see below)

Structural inputs (ASSUMPTION, instruction-count physics, NOT fit): ops_per_elem and byte-passes per
op (simulator/models/m4_cpu.py). exp() (softmax/swiglu) is the cost driver -> a large transcendental
instruction-weight, NOT a reduction/elementwise split.

Calibrated: ONE global eta_c (numpy-kernel fraction of peak NEON fp32 throughput) + per-op overhead_op
(the constant-dominated floor of rmsnorm/rope/residual/sampling-dispatch). eta_c is fit on the
compute-bound ops (exp ops, rmsnorm, rope, the per-element argmax compare); residual is overhead-only.

eta_bw is an ASSUMPTION, NOT calibrated: this fp32 decode dataset has NO bandwidth-resolved op (every
working set fits in L1/L2/L3 and every op is compute- or overhead-bound), and there is no CPU mem-BW
micro-benchmark (audit gap). eta_bw=0.6 is a literature-typical cache-efficiency placeholder; the
memory term only binds for the largest working set (qwen vocab -> L3) where it is corroborated (not
contradicted) by the data. It is present for prefill / architecture-study working sets.

fp16 cpu_ops.json rows are numpy-EMULATED on the A76 -> UPPER BOUND, NOT calibrated here (fp32 only).

Reads  measurements/aetina/cpu_ops.json
Writes simulator/models/params/m4_cpu_instrcount.json   (calibrated factors the engine consumes)
       validation/reports/phase1.2/m4_cpu.json           (per-op residuals + honesty notes)

Run: ./.venv/bin/python tools/analysis/fit_m4_cpu_instrcount.py
"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from simulator.models.m4_cpu import (  # noqa: E402
    MODELS, OPS_PER_ELEM, BYTE_PASSES, ETA_BW, _n_elem, _working_set_bytes, _tier_bw, _peak_lane_ops,
)

AET = ROOT / "measurements/aetina"
PARAMS_OUT = ROOT / "simulator/models/params/m4_cpu_instrcount.json"
REPORT_OUT = ROOT / "validation/reports/phase1.2/m4_cpu.json"

# Ops whose latency is set by arithmetic throughput (exp ops, norm, rope, per-element argmax compare).
# residual is the lone overhead-dominated op (a few us at decode N, below cache-scan resolution).
COMPUTE_OPS = ("softmax", "swiglu", "rmsnorm", "rope_apply", "sampling_argmax")


def _rows(ops, spec):
    """One row per fp32 cpu_ops.json entry: (op_class, model, measured_us, compute_raw_us, memory_us)."""
    peak = _peak_lane_ops(spec)
    rows = []
    for v in ops.values():
        if v["dtype"] != "fp32":
            continue
        op, model, med = v["op"], v["model"], v["median_us"]
        base = "softmax" if op.startswith("softmax") else op
        c = MODELS[model]
        n = _n_elem(op, c)
        wsb = _working_set_bytes(base, n)
        compute_raw = n * OPS_PER_ELEM[base] / peak * 1e6        # us at eta_c = 1
        memory_us = wsb / (_tier_bw(spec, wsb) * ETA_BW * 1e9) * 1e6
        rows.append((base, model, med, compute_raw, memory_us))
    return rows


def main():
    ops = json.loads((AET / "cpu_ops.json").read_text())["ops"]
    spec = json.loads((ROOT / "simulator/specs/cpu_rk3588.json").read_text())
    rows = _rows(ops, spec)

    # Calibrate ONE eta_c + per-op overhead on the compute-bound ops: med = (1/eta_c)*compute_raw + ovh[op].
    cops = sorted(set(COMPUTE_OPS))
    A = [[cr] + [1.0 if base == o else 0.0 for o in cops]
         for base, _, _, cr, _ in rows if base in COMPUTE_OPS]
    y = [med for base, _, med, _, _ in rows if base in COMPUTE_OPS]
    sol, *_ = np.linalg.lstsq(np.array(A), np.array(y), rcond=None)
    inv_eta_c = float(sol[0])
    eta_c = 1.0 / inv_eta_c
    overhead = {o: max(0.0, float(v)) for o, v in zip(cops, sol[1:])}
    # residual is overhead-only: ovh = mean(measured - roofline-without-overhead), clamped >= 0.
    res = [med - max(cr * inv_eta_c, mem) for base, _, med, cr, mem in rows if base == "residual"]
    overhead["residual"] = max(0.0, float(np.mean(res)))

    # Per-op residuals with the full max(compute, memory) + overhead roofline.
    per_op, all_err = {}, []
    for base, model, med, cr, mem in rows:
        compute_us = cr * inv_eta_c
        latency = max(compute_us, mem) + overhead[base]
        err = abs(latency - med) / med
        per_op.setdefault(base, []).append(err)
        all_err.append(err)
        b = "compute" if compute_us >= mem else "memory"
        per_op.setdefault(base + "__bounds", set()).add(b)

    params = {
        "_doc": "CALIBRATED factors for the CPU instruction-count roofline (m4_cpu.py), fit to fp32 "
                "cpu_ops.json on a single A76 core. eta_c calibrated; eta_bw=ASSUMPTION (no BW-resolved "
                "op in fp32 decode data); overhead_op per op. fp16 = emulated upper bound (not fit).",
        "eta_c": round(eta_c, 4),
        "eta_bw": ETA_BW,
        "eta_bw_tag": "assumption (no CPU mem-BW micro-benchmark; literature-typical cache efficiency)",
        "overhead_op_us": {o: round(overhead[o], 3) for o in sorted(overhead)},
        "ops_per_elem": OPS_PER_ELEM,
        "byte_passes": BYTE_PASSES,
        "calibration_basis": "single A76 core, single-thread, numpy fp32 (measurements/aetina/cpu_ops.json)",
    }
    PARAMS_OUT.write_text(json.dumps(params, indent=1))

    report = {
        "module": "m4_cpu",
        "honesty": "CALIBRATED to fp32 cpu_ops.json (single A76 core, single-thread). fp16 = numpy "
                   "UPPER BOUND (emulated); swiglu fp16 = mixed precision; A55 / multicore = simulated "
                   "(IPC=1 little, single-core measured -> extrapolated). eta_bw = ASSUMPTION.",
        "equation": "latency_us = max(compute_us, memory_us) + overhead_op; "
                    "compute_us = n_elem*ops_per_elem/(W*IPC*freq)/eta_c; "
                    "memory_us = working_set_bytes/(BW_tier*eta_bw)",
        "calibrated": {"eta_c": round(eta_c, 4), "basis": params["calibration_basis"]},
        "assumption": {"eta_bw": ETA_BW, "why": "no bandwidth-resolved op in fp32 decode data; no CPU "
                       "mem-BW micro-benchmark (audit gap). Corroborated (not contradicted) by the qwen "
                       "vocab op (594 KiB -> L3, the one decode op on the cache/memory branch)."},
        "structural_assumption": {"ops_per_elem": OPS_PER_ELEM, "byte_passes": BYTE_PASSES,
                                  "note": "exp() (softmax/swiglu) = cost driver (large transcendental "
                                          "instruction-weight), NOT a reduction/elementwise split."},
        "overhead_op_us": {o: round(overhead[o], 3) for o in sorted(overhead)},
        "residuals_pct": {o: {"median": round(float(np.median(per_op[o])) * 100, 2),
                              "max": round(float(np.max(per_op[o])) * 100, 2),
                              "bound": sorted(per_op[o + "__bounds"])}
                          for o in sorted(set(COMPUTE_OPS) | {"residual"})},
        "overall_residual_pct": {"median": round(float(np.median(all_err)) * 100, 2),
                                 "p95": round(float(np.percentile(all_err, 95)) * 100, 2),
                                 "max": round(float(np.max(all_err)) * 100, 2)},
        "notes": {
            "residual_noise": "residual is the noisiest op (cov up to 0.25 in cpu_ops.json) and "
                              "overhead-dominated at decode N; its model error is within its own "
                              "measurement noise.",
            "cache_branch": "decode support-op working sets reside in L1/L2/L3, NEVER LPDDR; the qwen "
                            "vocab op (594 KiB) spills to L3 and takes the memory/cache branch.",
            "prefill": "decode (1-token) costs; prefill = x S tokens (analytic, unvalidated). 'swap "
                       "LPDDR4->5 recompute' applies to PREFILL only.",
        },
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, indent=1))

    g = report["overall_residual_pct"]
    print(f"M4-CPU instr-count: eta_c={eta_c:.4f} eta_bw={ETA_BW}(assumption) "
          f"overall median={g['median']}% p95={g['p95']}% max={g['max']}%")
    for o in sorted(set(COMPUTE_OPS) | {"residual"}):
        r = report["residuals_pct"][o]
        print(f"  {o:16s} median={r['median']:5.2f}% max={r['max']:5.2f}% bound={r['bound']}")


if __name__ == "__main__":
    main()
