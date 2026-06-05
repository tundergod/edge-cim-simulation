"""Phase 1 — validate M5 (workload/trace generator) against the Phase 0.1 oracle.

M5 has no fitted parameters (it is deterministic). Validation reuses the Phase 0.1
inventory oracle (expected_ops_check) and confirms every op category in each
(model x task) op_profile maps to an op the runtime tracer actually produced — i.e. the
profile introduces NO op that wasn't traced from HF (0 orphans). Counts come from the
inventory, never hand-rolled x layers (issue from phase0.2 findings).

Reads  measurements/op_inventory/{model}.json, measurements/op_profile/{model}_{task}.json
Writes validation/reports/phase1.1/m5.json

Run: ./.venv/bin/python validation/validate_m5_trace.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INV = ROOT / "measurements/op_inventory"
PROF = ROOT / "measurements/op_profile"
MODELS = ["llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b", "qwen2.5-7b"]


def main():
    per_model = {}
    all_ok = True
    for model in MODELS:
        inv = json.loads((INV / f"{model}.json").read_text())
        covered = inv["expected_ops_check"]["all_semantic_covered"]
        distinct = set(inv["distinct_ops"])

        # every op in every (model x task) profile must be an op the tracer produced
        orphans, n_rows, n_tasks = set(), 0, 0
        for pf in sorted(PROF.glob(f"{model}_*.json")):
            prof = json.loads(pf.read_text())
            n_tasks += 1
            for r in prof["rows"]:
                n_rows += 1
                if r["op"] not in distinct:
                    orphans.add(r["op"])

        ok = covered and not orphans
        all_ok &= ok
        per_model[model] = {"semantic_covered": covered, "n_distinct_ops": len(distinct),
                            "n_tasks": n_tasks, "n_profile_rows": n_rows,
                            "orphan_ops": sorted(orphans), "pass": ok}

    report = {
        "module": "m5_trace",
        "method": "reuse Phase 0.1 inventory oracle (expected_ops_check) + 0-orphan check "
                  "(every op_profile op was traced from HF); counts from inventory, no hand x L",
        "per_model": per_model,
        "pass_all": all_ok,
    }
    (ROOT / "validation/reports/phase1.1/m5.json").write_text(json.dumps(report, indent=1))
    for m, r in per_model.items():
        print(f"M5 {m:14s}: covered={r['semantic_covered']} distinct={r['n_distinct_ops']} "
              f"rows={r['n_profile_rows']} orphans={len(r['orphan_ops'])} PASS={r['pass']}")
    print(f"M5 pass_all={all_ok}")


if __name__ == "__main__":
    main()
