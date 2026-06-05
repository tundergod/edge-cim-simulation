"""Phase 1.2 (D2) — quantify the M4-NPU analytic roofline against HeteroInfer trend SHAPE.

NO RKNPU2 silicon (issue #13, board offline) -> there is NO numeric gate to pass; the only
acceptance is TREND-SHAPE agreement with the borrowed HeteroInfer characterization (SOSP'25,
papers/methodology-and-simulators/). This script drives simulator.models.m4_npu.NpuModel and
quantifies the three borrowed trends into explicit conditions, ALL tagged `simulated`:

  (a) Fig3 staircase: compute-bound latency vs N is monotone non-decreasing and STEPS exactly on
      multiples of the borrowed 32x32 systolic dim (knee aligned to 32). [vs Fig3 SHAPE]
  (b) Fig4 order/shape: the order/shape throughput penalty is bounded by <=6x. [vs Fig4]
  (c) Fig5 bandwidth: the effective-BW band is 59-66% of the 68 GB/s peak DENOMINATOR. [vs Fig5]

Writes validation/reports/phase1.2/m4_npu.json. The report carries the 3 quantified trend
conditions, a SIMULATED-acceptance sentence, and the issue-#13-vs-ONNXim upgrade distinction.

Run: ./.venv/bin/python tools/analysis/build_m4_npu.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from simulator.models.engine import Workload, check_return  # noqa: E402
from simulator.models.m4_npu import NpuModel  # noqa: E402
from simulator.specs.loader import load_spec  # noqa: E402

REP = ROOT / "validation/reports/phase1.2"


def trend_staircase(m, sd):
    """(a) vs Fig3: compute-bound N-sweep is monotone and steps only on multiples of sd."""
    Ns = list(range(1, 257))
    lat = {n: m.predict(Workload(op="matmul", M=512, K=2048, N=n))["latency_us"] for n in Ns}
    monotone = all(lat[Ns[i]] <= lat[Ns[i + 1]] + 1e-9 for i in range(len(Ns) - 1))
    # knee = an N where latency strictly increases vs N-1; every knee must sit at a multiple of sd.
    knees = [n for n in Ns[1:] if lat[n] > lat[n - 1] + 1e-9]
    knees_on_grid = all(n % sd == 1 for n in knees)  # step lands the instant a new pad-block opens
    return {
        "vs": "HeteroInfer Fig3 (32x32 systolic staircase) SHAPE",
        "regime": "compute-bound prefill M=512, K=2048, N=1..256",
        "monotone_nondecreasing": bool(monotone),
        "n_knees": len(knees),
        "knee_positions": knees,
        "knees_aligned_to_borrowed_sd": bool(knees_on_grid),
        "borrowed_systolic_dim": sd,
        "pass_simulated": bool(monotone and knees_on_grid and len(knees) > 0),
        "tag": "simulated (borrowed Fig3 shape; NO silicon, #13)",
    }


def trend_order_shape(m):
    """(b) vs Fig4: order/shape penalty bounded by <=6x across a wide N:K aspect sweep."""
    K = 512
    sweep = [(K, n) for n in (256, 512, 1024, 2048, 4096, 8192, K * 64)]
    factors = {f"N{n}_over_K{k}": round(m._order_shape_factor(k, n), 4) for k, n in sweep}
    fmax = max(factors.values())
    return {
        "vs": "HeteroInfer Fig4 (order/shape sensitivity, up to 6x)",
        "factors_by_aspect": factors,
        "max_factor": round(fmax, 4),
        "bound": 6.0,
        "pass_simulated": bool(fmax <= 6.0 + 1e-9),
        "tag": "simulated (borrowed Fig4 ceiling; NO silicon, #13)",
    }


def trend_bw_frac(spec):
    """(c) vs Fig5: the borrowed eff-BW band is 59-66% of the 68 GB/s peak DENOMINATOR.

    The Fig5 fractions (eff_frac_low/high) are 40-45 GB/s single-proc decode over the 68 GB/s peak.
    The spec also derives an absolute RKNPU2 band (eff_low/high) against RK3588's ~34 host BW; that
    is a SEPARATE number and is NOT what the 59-66% fraction is taken over."""
    bw = spec["bw_GBs"]
    denom = float(bw["heteroinfer_peak_denominator"])  # 68 (NOT RK3588's 34)
    frac_low = float(bw["eff_frac_low"])               # 0.59 = 40/68 (Fig5)
    frac_high = float(bw["eff_frac_high"])             # 0.66 = 45/68 (Fig5)
    return {
        "vs": "HeteroInfer Fig5 (single-proc decode 40-45 / 68 GB/s = 59-66%)",
        "denominator_GBs": denom,
        "fig5_abs_GBs": [round(frac_low * denom, 1), round(frac_high * denom, 1)],  # ~40-45
        "frac_low": round(frac_low, 4),
        "frac_high": round(frac_high, 4),
        "rknpu2_abs_band_GBs": [bw["eff_low"], bw["eff_high"]],  # 34 x frac (separate, RKNPU2 host)
        "target_band": [0.59, 0.66],
        "pass_simulated": bool(0.59 - 1e-9 <= frac_low and frac_high <= 0.66 + 1e-9),
        "note": "59-66% is of the 68 GB/s peak (Fig5 40-45/68), NOT of RK3588's ~34 host BW; the "
                "RKNPU2 absolute band (34 x frac) is recorded separately.",
        "tag": "simulated/borrowed (Fig5 band; NO silicon, #13)",
    }


def main():
    spec = load_spec("npu_rknpu2")
    m = NpuModel(spec)
    sd = int(spec["systolic_dim"][0])

    # contract self-check: predict() must return the frozen keys.
    for wl in (Workload(op="matmul", M=1, K=2048, N=2048),
               Workload(op="attn_bmm", kv=512, heads=8, layers=1, extra={"hd": 128})):
        check_return(m.predict(wl))

    a = trend_staircase(m, sd)
    b = trend_order_shape(m)
    c = trend_bw_frac(spec)

    report = {
        "module": "m4_npu",
        "engine": "analytic systolic-roofline (Phase 1.2, D2)",
        "honesty": "simulated",
        "acceptance": "SIMULATED-acceptance (NO silicon, issue #13): there is NO per-op numeric "
                      "gate. Acceptance = trend-SHAPE agreement with the borrowed HeteroInfer "
                      "characterization only: (a) the Fig3 32x32 staircase is reproduced (monotone "
                      "+ knee aligned to the borrowed systolic dim), (b) the Fig4 order/shape "
                      "penalty is bounded <=6x, and (c) the Fig5 effective-BW band is 59-66% of the "
                      "68 GB/s peak. Every condition below is `simulated`/`borrowed`, NOT calibrated.",
        "trend_conditions": {
            "a_staircase_vs_fig3": a,
            "b_order_shape_vs_fig4": b,
            "c_bw_frac_vs_fig5": c,
        },
        "all_trends_pass_simulated": bool(a["pass_simulated"] and b["pass_simulated"]
                                          and c["pass_simulated"]),
        "upgrade": {
            "issue_13_silicon": "superseded-not-satisfied: the RKNPU2 matmul/attention micro-"
                                "benchmark (#13) was NOT collected (board offline). This analytic "
                                "trend-shape model SUPERSEDES the blocked silicon gate as the "
                                "Phase-1.2 deliverable; it does NOT ACHIEVE it. The #13 median/p95 "
                                "silicon error gate is recorded as superseded-not-satisfied "
                                "(ADR-0006 gate revision).",
            "onnxim_phase_1_3": "Phase 1.3 will drop in ONNXim (a generic-systolic NPU simulator) "
                                "behind the same engine= interface and cross-check it against these "
                                "HeteroInfer trends. ONNXim is SIMULATED, NOT silicon.",
            "distinction": "issue #13 = real RKNPU2 silicon (absent, superseded-not-satisfied); "
                           "ONNXim = a heavier SIMULATOR (Phase 1.3), still not silicon. Neither is "
                           "calibrated to our RKNPU2 board.",
        },
        "honesty_notes": {
            "everything_simulated": "ALL fields are simulated/borrowed: 6 TOPS + dtypes from "
                                    "datasheet, 32x32 borrowed from Hexagon (HeteroInfer), BW band "
                                    "borrowed from Fig5. NONE is fit to RKNPU2 silicon.",
            "dtypes": "INT4/8/16 + FP16 only (datasheet); no BF16/TF32.",
            "energy": "no RKNPU2 power telemetry -> energy NOT determinable.",
        },
    }
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "m4_npu.json").write_text(json.dumps(report, indent=1))
    print("M4-NPU (simulated, NO silicon #13):")
    print(f"  (a) staircase vs Fig3: monotone={a['monotone_nondecreasing']} "
          f"knees={a['n_knees']} aligned_to_{sd}={a['knees_aligned_to_borrowed_sd']} "
          f"-> {a['pass_simulated']}")
    print(f"  (b) order/shape vs Fig4: max_factor={b['max_factor']} <= {b['bound']} "
          f"-> {b['pass_simulated']}")
    print(f"  (c) BW frac vs Fig5: {c['frac_low']:.2%}-{c['frac_high']:.2%} of {c['denominator_GBs']:.0f} "
          f"in {c['target_band']} -> {c['pass_simulated']}")
    print(f"  all trends pass (SIMULATED acceptance): {report['all_trends_pass_simulated']}")


if __name__ == "__main__":
    main()
