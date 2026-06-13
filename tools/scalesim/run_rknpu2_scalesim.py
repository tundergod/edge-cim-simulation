"""Phase 1.6 — ScaleSim NPU third engine (runs LOCAL, pure Python, no board).

Faithful-native systolic model: configure SCALE-Sim v2 as a generic 32x32 weight-stationary array
(borrowed dim, NO RKNPU2 silicon, #13) and let the native systolic behaviours EMERGE — we do NOT
apply HeteroInfer's order-transpose optimisation. Produces:
  - simulated/scalesim/rknpu2_sim_matmul.json   (per-shape latency, mirrors the onnxim table)
  - the native_sensitivity sweep (alignment / order / large-M) for fit_npu_scalesim.py

3-core aggregation: SCALE-Sim models ONE 32x32 array (~2 TOPS); analytic's 6 TOPS + onnxim both
model 3 cores. We divide cycles by `cores` to put all three engines on the same 3-core (~6 TOPS)
basis. The /cores factor ASSUMES ideal linear 3-core scaling (no inter-core overhead) — a
load-bearing assumption, applied symmetrically. All of {32x32 dim, SRAM sizes, dataflow, /cores}
are simulated/borrowed, NOT measured RKNPU2.

Run: ./.venv/bin/python tools/scalesim/run_rknpu2_scalesim.py
"""
import csv
import glob
import json
import math
import os
import tempfile
from pathlib import Path

PASS_CAP = 8000   # skip shapes whose 32x32 fold-pass estimate exceeds this (cycle-sim too slow)


def folds(M, K, N):
    return math.ceil(M / 32) * math.ceil(N / 32) * math.ceil(K / 32)

from scalesim.scale_sim import scalesim

ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((ROOT / "simulator/specs/npu_rknpu2.json").read_text())
ONNXIM = ROOT / "simulated/onnxim/rknpu2_sim_matmul.json"
OUT = ROOT / "simulated/scalesim/rknpu2_sim_matmul.json"

ARRAY_H, ARRAY_W = SPEC["systolic_dim"]            # [32, 32] — borrowed (Hexagon/HeteroInfer), assumption
FREQ_GHZ = SPEC["freq_ghz"]                         # 1.0 — assumption
CORES = SPEC["cores"]                               # 3 — datasheet; /cores = ideal-scaling ASSUMPTION
# ScaleSim config knobs the spec does NOT have -> assumed inputs (honesty surface):
SRAM_KB = {"ifmap": 256, "filter": 256, "ofmap": 128}   # arbitrary on-chip buffer sizes [assumed]
DATAFLOW = "ws"                                          # weight-stationary (RKNPU2 approx) [assumed]

CFG = f"""[general]
run_name = rknpu2_scalesim
[architecture_presets]
ArrayHeight = {ARRAY_H}
ArrayWidth = {ARRAY_W}
ifmapsramszkB = {SRAM_KB['ifmap']}
filtersramszkB = {SRAM_KB['filter']}
ofmapsramszkB = {SRAM_KB['ofmap']}
IfmapOffset = 0
FilterOffset = 10000000
OfmapOffset = 20000000
Dataflow = {DATAFLOW}
Bandwidth = 10
[run_presets]
InterfaceBandwidth = CALC
"""


def run_one(M, K, N, workdir):
    """Run SCALE-Sim on a single GEMM (M,K,N). Returns (total_cycles, overall_util_pct)."""
    cfg = workdir / "rknpu2.cfg"; cfg.write_text(CFG)
    topo = workdir / "g.csv"
    topo.write_text(f"Layer, M, N, K,\ng_{M}_{N}_{K}, {M}, {N}, {K},\n")   # ScaleSim GEMM = M,N,K
    out = workdir / "out"
    s = scalesim(save_disk_space=True, verbose=False, config=str(cfg),
                 topology=str(topo), input_type_gemm=True)
    s.run_scale(top_path=str(out))
    rep = glob.glob(str(out) + "/**/COMPUTE_REPORT.csv", recursive=True)[0]
    rows = list(csv.reader(open(rep)))
    hdr = [h.strip() for h in rows[0]]; val = [c.strip() for c in rows[1]]
    rec = dict(zip(hdr, val))
    return int(rec["Total Cycles"]), float(rec["Overall Util %"])


def latency_us(cycles):
    """cycles on one array @ FREQ_GHz, divided across CORES (ideal-scaling assumption)."""
    return cycles / CORES / FREQ_GHZ / 1000.0


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shapes = [r["shape"] for r in json.loads(ONNXIM.read_text())["rows"]]   # [M,K,N], same 15 as onnxim
    with tempfile.TemporaryDirectory() as d:
        wd = Path(d)
        # --- main per-shape table (same shapes as onnxim, canonical orientation) ---
        # ScaleSim is cycle-accurate -> the giant FFN/prefill shapes are intractable; skip them
        # SYMMETRICALLY and note (the 3-way spread then compares on the common tractable subset).
        rows, skipped = [], []
        for M, K, N in shapes:
            if folds(M, K, N) > PASS_CAP:
                skipped.append({"shape": [M, K, N], "fold_passes": folds(M, K, N),
                                "reason": f"scalesim fold-passes > cap {PASS_CAP} (cycle-sim too slow)"})
                continue
            cyc, util = run_one(M, K, N, wd)
            rows.append({"shape": [M, K, N], "latency_us": round(latency_us(cyc), 1),
                         "cycles_1core": cyc, "overall_util_pct": round(util, 2)})

        # --- native sensitivity sweeps (best<->worst ratio of cycles), reported AS THEY FALL ---
        def sweep(cases):
            r = [{"label": lab, "MKN": [M, K, N], **dict(zip(("cycles", "util_pct"),
                  (lambda c, u: (c, round(u, 2)))(*run_one(M, K, N, wd))))} for lab, (M, K, N) in cases]
            cyc = [x["cycles"] for x in r]
            return {"cases": r, "worst_over_best": round(max(cyc) / min(cyc), 2)}

        # representative small shapes (the sensitivity RATIOS hold at small dims; keeps cycle-sim fast)
        sens = {
            # (1) alignment to the 32-wide array: N a multiple of 32 vs not
            "alignment_N": sweep([("N=128 (mult of 32)", (32, 512, 128)),
                                  ("N=130 (misaligned)", (32, 512, 130))]),
            # (2) operand order: SAME MACs (M*N*K const), swap which dim is large (MAC-invariant)
            "order_MN_swap": sweep([("M=256,N=64 (M-heavy)", (256, 512, 64)),
                                    ("M=64,N=256 (N-heavy)", (64, 512, 256))]),
            # (3) shape / activation-column count: tiny M (decode GEMV) vs batched M (fixed K,N)
            "shape_M": sweep([("M=1 (decode GEMV)", (1, 512, 256)),
                              ("M=32", (32, 512, 256)), ("M=128 (prefill batch)", (128, 512, 256))]),
        }

    out = {
        "_doc": "SCALE-Sim v2 RKNPU2-approx (32x32 weight-stationary, 3-core /cores aggregation) "
                "per-shape GEMM latency. SIMULATED, NOT silicon (#13). Native systolic behaviour is "
                "EMERGENT (not tuned, no order-transpose optimisation). Dims/SRAM/dataflow/cores-scaling "
                "all assumed/borrowed, NOT measured RKNPU2.",
        "config": {"array": [ARRAY_H, ARRAY_W], "dataflow": DATAFLOW, "freq_ghz": FREQ_GHZ,
                   "cores": CORES, "sram_kB_assumed": SRAM_KB,
                   "array_dim_provenance": "32x32 borrowed from Hexagon (HeteroInfer) [assumption]",
                   "cores_aggregation": "/cores assumes ideal linear 3-core scaling [assumption]"},
        "honesty": "simulated (SCALE-Sim v2 generic-systolic, RKNPU2-approx), NOT silicon",
        "tags": ["native (not tuned)", "model, NOT silicon"],
        "rows": rows,
        "skipped_shapes": skipped,
        "native_sensitivity": sens,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} shapes; "
          f"util {min(r['overall_util_pct'] for r in rows)}-{max(r['overall_util_pct'] for r in rows)}%; "
          f"sens worst/best: align {sens['alignment_N']['worst_over_best']}x "
          f"order {sens['order_MN_swap']['worst_over_best']}x shape {sens['shape_M']['worst_over_best']}x")


if __name__ == "__main__":
    main()
