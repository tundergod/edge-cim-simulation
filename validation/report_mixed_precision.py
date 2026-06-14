"""Phase 2.2b — CimHetero mixed-precision SIMULATED report (NOT a validation).

The project's CIM-INT8 matmul × GPU-FP16 attention config. There is NO concurrent-unit
silicon (Aetina out for repair, #52), so this is a SIMULATED demo: the conversion-op COST
+ the modeled heterogeneous decode vs the AllCim baseline. Mixed-precision OUTPUT QUALITY
is NOT modeled (D3) — only conversion COST. Named `report_` (not `validate_`) precisely
because there is no ground truth to validate against (honesty discipline).

Writes validation/reports/phase2/mixed_precision.json.
Run: ./.venv/bin/python validation/report_mixed_precision.py   (from repo root)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "trace_export"))
from simulator.runtime.config import SimConfig  # noqa: E402
from simulator.runtime.runner import run  # noqa: E402
from simulator.runtime.workload import build_token_dag  # noqa: E402
from simulator.runtime.scheduler import CimHeteroScheduler  # noqa: E402

OUT = ROOT / "validation/reports/phase2"
MODELS = ["llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b"]


def _cfg(model, policy):
    return SimConfig.from_dict({"workload": {"model": model, "context": 1024},
                                "platform": {"memory_spec": "mem_lpddr4x", "topology": "cim_topo_card"},
                                "scheduler": {"policy": policy}})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    for m in MODELS:
        base = run(_cfg(m, "all_cim"))
        het = run(_cfg(m, "cim_hetero"))
        convs = [n for n in CimHeteroScheduler().assign(build_token_dag(m, "decode", 512)).nodes
                 if n.category == "convert"]
        rows[m] = {"allcim_tok_s": base["tok_s"], "cimhetero_tok_s": het["tok_s"],
                   "cimhetero_over_allcim": round(het["tok_s"] / base["tok_s"], 3),
                   "conversions_per_decode_token": len(convs),
                   "conversion_bytes_per_token": sum(n.bytes_streamed for n in convs),
                   "calibrated_anchor": het["calibrated_anchor"]}
    out = {
        "module": "mixed_precision", "phase": "2.2b", "label": "simulated",
        "config": "CIM-INT8 matmul × GPU-FP16 attention (the project's mixed-precision config)",
        "honesty": "SIMULATED — no concurrent-unit silicon (Aetina out for repair, #52). Reports the "
                   "conversion-op COST + modeled heterogeneous decode vs the AllCim baseline ONLY; "
                   "mixed-precision OUTPUT QUALITY is NOT modeled (D3). NOT validated/measured. CimHetero "
                   "is compute-bound on the GPU attn_bmm_us (an unoptimised lower-bound), so it is SLOWER "
                   "than AllCim — the value is the conversion cost + faithful structure, NOT a speedup.",
        "models": rows,
    }
    (OUT / "mixed_precision.json").write_text(json.dumps(out, indent=1))
    print("CimHetero mixed-precision (SIMULATED; conversion COST only, quality NOT modeled — D3):")
    for m in MODELS:
        v = rows[m]
        print(f"  {m}: AllCim {v['allcim_tok_s']:.2f} -> CimHetero {v['cimhetero_tok_s']:.2f} tok/s "
              f"({v['cimhetero_over_allcim']}x); {v['conversions_per_decode_token']} converts/tok, "
              f"{v['conversion_bytes_per_token'] / 1e6:.2f} MB conversion traffic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
