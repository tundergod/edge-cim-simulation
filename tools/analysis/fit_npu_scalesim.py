"""Phase 1.6 — assemble the 3-way NPU model spread (analytic / ONNXim / SCALE-Sim) + the native
systolic sensitivity finding. Emits validation/reports/phase1.6/npu_scalesim.json.

CRITICAL honesty: all three are SIMULATIONS with NO silicon ground truth (#13). This reports a
SPREAD (model divergence), not a validation. No pass/fail, no "central/likely" value, no "two agree
=> correct". ScaleSim is a third uncertainty point, not an adjudicator. The native_sensitivity
magnitudes are what a 32x32-WS systolic MODEL produces (emergent, not tuned) — NOT measured RKNPU2.

Run: ./.venv/bin/python tools/analysis/fit_npu_scalesim.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCALESIM = ROOT / "simulator/engines/scalesim/rknpu2_sim_matmul.json"
ONNXIM = ROOT / "simulator/engines/onnxim/rknpu2_sim_matmul.json"
SPEC = ROOT / "simulator/specs/npu_rknpu2.json"
OUT = ROOT / "validation/reports/phase1.6/npu_scalesim.json"


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    from simulator.models.m4_npu import NpuModel
    from simulator.models.engine import Workload

    spec = json.loads(SPEC.read_text())
    ss = json.loads(SCALESIM.read_text())
    ox = {tuple(r["shape"]): r["latency_us"] for r in json.loads(ONNXIM.read_text())["rows"]}
    ana = NpuModel(spec, engine="analytic")

    # 3-way spread on the COMMON tractable subset (the shapes ScaleSim could run)
    rows = []
    for r in ss["rows"]:
        M, K, N = r["shape"]
        a = ana.predict(Workload(op="matmul", M=M, K=K, N=N, dtype="int8",
                                 kv=0, heads=0, layers=0, extra={}))["latency_us"]
        o = ox.get((M, K, N))
        s = r["latency_us"]
        lats = {"analytic": round(a, 1), "onnxim": (round(o, 1) if o else None), "scalesim": round(s, 1)}
        vals = [v for v in lats.values() if v]
        rows.append({"shape": [M, K, N], "lat_us": lats,
                     "spread_max_over_min": round(max(vals) / min(vals), 2) if len(vals) > 1 else None})

    spreads = [r["spread_max_over_min"] for r in rows if r["spread_max_over_min"]]
    out = {
        "module": "m4_npu_scalesim", "phase": "1.6", "engine": "scalesim (SCALE-Sim v2)",
        "no_silicon_ground_truth": True,
        "honesty": ("all three engines are SIMULATIONS, no silicon (#13). This is a SPREAD = model "
                    "divergence, NOT validation. No value is more likely; ScaleSim is a third "
                    "uncertainty point, not a judge. NPU primary deferred to Phase 2 L4."),
        "config": ss["config"],
        "three_way_subset": rows,
        "spread_median_x": round(sorted(spreads)[len(spreads) // 2], 2) if spreads else None,
        "spread_max_x": max(spreads) if spreads else None,
        "skipped_shapes": ss["skipped_shapes"],
        "skipped_note": ("ScaleSim is cycle-accurate -> the giant FFN/prefill shapes "
                         f"({len(ss['skipped_shapes'])} of {len(ss['rows']) + len(ss['skipped_shapes'])}) "
                         "are intractable and skipped SYMMETRICALLY; the 3-way compares the common subset."),
        "decode_gemv_util_pct": ss["rows"][0]["overall_util_pct"],
        "native_sensitivity": {
            "tags": ["native (emergent, not tuned)", "model, NOT silicon — a 32x32-WS systolic MODEL, "
                     "not measured RKNPU2"],
            "framing": ("faithful-to-architecture exposure: future schedulers that EXPLOIT alignment/"
                        "order/batching get speedup; that IGNORE it pay the cost. Magnitudes are "
                        "model-native (not RKNPU2 facts)."),
            "heteroinfer_corroboration": ("directional only (same sign of effect: 32-alignment, order, "
                                          "shape) — NOT magnitude agreement; different array/config."),
            **{k: ss["native_sensitivity"][k] for k in ss["native_sensitivity"]},
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}: 3-way on {len(rows)} shapes, "
          f"spread median {out['spread_median_x']}x max {out['spread_max_x']}x; "
          f"decode util {out['decode_gemv_util_pct']}%; sens "
          f"align {out['native_sensitivity']['alignment_N']['worst_over_best']}x "
          f"order {out['native_sensitivity']['order_MN_swap']['worst_over_best']}x "
          f"shape {out['native_sensitivity']['shape_M']['worst_over_best']}x")


if __name__ == "__main__":
    main()
