"""Phase 0.2 — emit the op-profile measurement JSONs.

Layer-A: per (model x task) at the workload mean (prefill, decode) lengths ->
         measurements/op_profile/{model}_{task}.json  (16 files)
Layer-B: per model, on-grid scaling sweep ->
         measurements/op_profile/sweep_{model}.json   (4 files)

Counts/shapes/flops/bytes/intensity/measured come from op_profile.Model (which
self-validates against op_inventory on construction).

Run: ./.venv/bin/python tools/trace_export/gen_op_profile.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from op_profile import Model, GEMM_BYTES, ELT_BYTES

OUT = Path("measurements/op_profile")
MODELS = ["llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b", "qwen2.5-7b"]
TASKS = {"sharegpt": "ShareGPT", "gsm8k": "GSM8K",
         "longbench-triviaqa": "LongBench-TriviaQA", "humaneval": "HumanEval"}
DTYPE = {"gemm_bytes": GEMM_BYTES, "elt_bytes": ELT_BYTES,
         "note": "bytes: INT8 (1B) for matmul/bmm operands, FP16 (2B) for non-GEMM; "
                 "weights counted once per token (streamed, non-resident). "
                 "intensity is predicted-side (measured roofline/knee = Phase 0.3/1)."}
CATS = ["matmul", "attention", "ffn", "norm", "softmax", "rope", "residual", "kv_cache", "embedding"]


def summarize(rows):
    by_cat, by_phase = {}, {"prefill": {}, "decode": {}}
    for r in rows:
        c, p = r["category"], r["phase"]
        d = by_cat.setdefault(c, {"count": 0, "flops": 0, "bytes": 0, "rows": 0})
        d["count"] += r["count"]; d["flops"] += r["flops"] * r["count"]
        d["bytes"] += r["bytes"] * r["count"]; d["rows"] += 1
        e = by_phase[p].setdefault(c, {"count": 0, "flops": 0, "bytes": 0})
        e["count"] += r["count"]; e["flops"] += r["flops"] * r["count"]; e["bytes"] += r["bytes"] * r["count"]
    return {
        "by_category": {c: by_cat[c] for c in CATS if c in by_cat},
        "by_phase": by_phase,
        "total_flops": sum(r["flops"] * r["count"] for r in rows),
        "total_bytes": sum(r["bytes"] * r["count"] for r in rows),
        "total_rows": len(rows),
        "measured_rows": sum(1 for r in rows if r["measured"]),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    wl = json.loads(Path("measurements/op_inventory/workload_lengths.json").read_text())
    models = {m: Model(m) for m in MODELS}
    nA = nB = 0
    for m, M in models.items():
        for task, label in TASKS.items():
            P = round(wl[task][m]["prefill"]["mean"])
            D = round(wl[task][m]["decode"]["mean"])
            rows = M.profile(P, D)
            doc = {
                "model": m, "task": task, "dataset": label,
                "prefill_len": P, "decode_len": D, "config": M.config, "dtype": DTYPE,
                "scope_flag": (">8K context: prefill exceeds the 2K/8K simulator scope; "
                               "reported at natural length" if P > 8192 else None),
                "totals": summarize(rows),
                "rows": sorted(rows, key=lambda r: (r["phase"], -r["flops"] * r["count"])),
            }
            (OUT / f"{m}_{task}.json").write_text(json.dumps(doc, indent=1))
            nA += 1
        # Layer-B on-grid sweep
        grid = M.grid_profile()
        (OUT / f"sweep_{m}.json").write_text(json.dumps(
            {"model": m, "config": M.config, "dtype": DTYPE,
             "note": "on-grid Layer-B scaling: single-forward sigs at prefill {128,256,512,1024} "
                     "and decode kv {128,512,1024}; all measured=true (sweep_matrix grid).",
             "grid": grid}, indent=1))
        nB += 1
    print(f"wrote {nA} Layer-A + {nB} Layer-B profiles to {OUT}/")
    # quick report: dominant category per task by FLOPs (prefill+decode)
    print("\nDominant op category by total FLOPs (model=llama-3.1-8b):")
    for task in TASKS:
        doc = json.loads((OUT / f"llama-3.1-8b_{task}.json").read_text())
        bc = doc["totals"]["by_category"]
        top = sorted(bc.items(), key=lambda kv: -kv[1]["flops"])[:3]
        P, D = doc["prefill_len"], doc["decode_len"]
        print(f"  {task:20s} P={P:6d} D={D:4d}: " +
              ", ".join(f"{c}={v['flops']/1e9:.1f}GF" for c, v in top))


if __name__ == "__main__":
    main()
