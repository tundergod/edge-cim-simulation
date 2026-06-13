"""Phase 1.3 — ONNXim-vs-analytic NPU delta report -> validation/reports/phase1.3/m4_npu_onnxim.json.

Reads the ONNXim RKNPU2-approx per-shape latencies (simulator/engines/onnxim/rknpu2_sim_matmul.json,
produced on metiscard by tools/analysis/npu_onnxim_trace.py) and, for the SAME (M,K,N) shapes,
the Phase-1.2 analytic systolic-roofline (NpuModel engine='analytic'). Per-shape delta + re-checks
the HeteroInfer staircase trend against ONNXim.

The (M,K,N) set is the single source of truth = exactly the shapes ONNXim ran (read from the result
JSON), so the delta is apples-to-apples (N4). Each shape is ASSERTED to be an ONNXim HIT (the
engine='onnxim' provenance says "ONNXim ... NOT silicon", not the analytic-fallback note) BEFORE its
delta is computed (NB-3) — a silent fallback can never masquerade as an ONNXim-vs-analytic delta.

ONNXim = generic-systolic, RKNPU2-approx -> simulated, NOT silicon; ONNXim != issue #13 (which stays
superseded-not-satisfied, independent). Run AFTER the result JSON is rsync'd back.

Run: ./.venv/bin/python tools/analysis/build_m4_npu_onnxim.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from simulator.specs.loader import load_spec                  # noqa: E402
from simulator.models.engine import Workload                  # noqa: E402
from simulator.models.m4_npu import NpuModel                  # noqa: E402

SIM = ROOT / "simulator/engines/onnxim/rknpu2_sim_matmul.json"
OUT = ROOT / "validation/reports/phase1.3/m4_npu_onnxim.json"


def main():
    sim = json.loads(SIM.read_text())
    spec = load_spec("npu_rknpu2")
    ana = NpuModel(spec, engine="analytic")
    onx = NpuModel(spec, engine="onnxim")   # reads the same cache; must HIT for each shape

    rows = []
    for r in sim["rows"]:
        M, K, N = r["shape"]
        wl = Workload(op="matmul", M=M, K=K, N=N)
        o = onx.predict(wl)
        # NB-3: this MUST be an ONNXim hit, not the analytic fallback.
        assert "ONNXim" in o["provenance"] and "NOT silicon" in o["provenance"] and "fallback" not in o["provenance"], \
            f"shape {M}x{K}x{N} is NOT an ONNXim hit (fell back to analytic): {o['provenance']}"
        a_us = ana.predict(wl)["latency_us"]
        o_us = o["latency_us"]
        rows.append({"shape": [M, K, N], "onnxim_us": round(o_us, 3), "analytic_us": round(a_us, 3),
                     "delta_pct": round((o_us - a_us) / a_us * 100, 1), "bound": ana.predict(wl)["bound"]})

    # HeteroInfer staircase re-check on the K=2048 staircase shapes (M=1): monotone non-decreasing in N.
    stair = sorted([(r["shape"][2], r["onnxim_us"]) for r in rows if r["shape"][0] == 1 and r["shape"][1] == 2048])
    stair_monotone = all(b >= a for (_, a), (_, b) in zip(stair, stair[1:]))

    deltas = [abs(r["delta_pct"]) for r in rows]
    report = {
        "module": "m4_npu_onnxim",
        "phase": "1.3",
        "engine": "NpuModel(npu_rknpu2, engine='onnxim')",
        "source": "ONNXim %s, RKNPU2-approx (%s)" % (sim.get("onnxim_commit", "?")[:10], sim.get("config", "?")),
        "honesty": "simulated (ONNXim generic-systolic, RKNPU2-approx), NOT silicon",
        "n_shapes": len(rows),
        "median_abs_delta_pct": round(sorted(deltas)[len(deltas) // 2], 1) if deltas else None,
        "max_abs_delta_pct": round(max(deltas), 1) if deltas else None,
        "staircase_monotone_vs_heteroinfer": stair_monotone,
        "per_shape": rows,
        "upgrade": {
            "issue_13_silicon": "superseded-not-satisfied (independent; RKNPU2 micro-benchmark NOT "
                                "collected, board offline). ONNXim is a heavier SIMULATOR, NOT RKNPU2 "
                                "silicon — it does NOT achieve the #13 gate.",
            "onnxim_vs_13": "ONNXim != issue #13: ONNXim cross-checks the analytic systolic-roofline "
                            "(sim-vs-sim); neither is calibrated to RKNPU2 silicon.",
        },
        "note": "Both ONNXim and the analytic NPU are simulated (no RKNPU2 silicon). This is a "
                "sim-vs-sim trend cross-check; the analytic NpuModel stays the Phase-1.2 deliverable.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(f"m4_npu_onnxim: {len(rows)} shapes; median |delta| {report['median_abs_delta_pct']}% "
          f"(max {report['max_abs_delta_pct']}%); staircase monotone={stair_monotone} -> {OUT}")


if __name__ == "__main__":
    main()
