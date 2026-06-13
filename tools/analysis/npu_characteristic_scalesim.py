"""Phase 1.6b — ScaleSim POSITIVE CONTROL for the systolic-characteristic measurement (local).

Runs the SAME E1-E4 grid as npu_characteristic_trace.py (ONNXim) through SCALE-Sim v2, writing
simulated/scalesim/rknpu2_characteristic.json. ScaleSim is a literal 32x32 systolic-array cycle
simulator: a 32-period staircase is EXPECTED BY CONSTRUCTION (ceil(N/32) tiling) -> this run is a
positive control that confirms our probe/criterion can see a step we KNOW is there. The informative
comparison is whether ONNXim (which adds NoC/DRAM scheduling) shows the SAME period or smears it.
simulated, NOT silicon (#13). Records raw 1-core cycles (the shape lives in cycles, not in /cores µs).

Run: ./.venv/bin/python tools/analysis/npu_characteristic_scalesim.py
"""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.scalesim.run_rknpu2_scalesim import run_one, latency_us, folds, PASS_CAP  # noqa: E402
from tools.analysis.npu_characteristic_trace import shapes, K  # same grid, same fixed K  # noqa: E402

OUT = ROOT / "simulated/scalesim/rknpu2_characteristic.json"


def main():
    shps = shapes(pilot=False)
    t0 = time.time()
    rows, skipped = [], []
    with tempfile.TemporaryDirectory() as d:
        wd = Path(d)
        for i, (M, Kk, N) in enumerate(shps):
            if folds(M, Kk, N) > PASS_CAP:
                skipped.append({"shape": [M, Kk, N], "fold_passes": folds(M, Kk, N)})
                continue
            cyc, util = run_one(M, Kk, N, wd)
            rows.append({"shape": [M, Kk, N], "cycles_1core": cyc,
                         "latency_us": round(latency_us(cyc), 3), "util_pct": round(util, 2)})
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(shps)} ...")
    wall = round(time.time() - t0, 1)
    out = {
        "_doc": "SCALE-Sim v2 RKNPU2-approx (32x32-WS) characteristic sweep (Phase 1.6b) -- POSITIVE "
                "CONTROL. A 32-period staircase is EXPECTED BY CONSTRUCTION (ceil(N/32) tiling); this "
                "confirms the probe sees a known step. simulated, NOT silicon (#13). Raw 1-core cycles.",
        "config": {"array": [32, 32], "dataflow": "ws", "fixed_contraction_K": K},
        "role": "positive_control (32-step is tautological for a 32x32 array sim)",
        "honesty": "simulated (SCALE-Sim v2 generic-systolic, RKNPU2-approx), NOT silicon",
        "wall_seconds": wall, "skipped_shapes": skipped,
        "rows": sorted(rows, key=lambda r: r["shape"]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {len(rows)} shapes ({len(skipped)} skipped) in {wall}s -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
