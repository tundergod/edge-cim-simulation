"""Phase 0.1 — completeness cross-check of the op inventory.

Traced ops are aten primitives; semantic ops (RMSNorm, RoPE, SwiGLU, attention,
matmul, embedding) are compositions of them. This maps each expected semantic op
to its aten primitives, confirms each is present, whitelists housekeeping
primitives (no semantic meaning), and flags anything left over as `unmatched`.
Writes the result back into each measurements/op_inventory/{model}.json.

Run: ./.venv/bin/python tools/trace_export/expected_ops.py
"""
import json
from pathlib import Path

OUT = Path("measurements/op_inventory")

# semantic op -> the aten primitives it decomposes into.
# All listed must appear, EXCEPT names in ALT (any one suffices — e.g. matmul is
# `mm` for bias-free linears (Llama) or `addmm` for biased ones (Qwen2.5 QKV)).
ALT = {"matmul (QKV/O/FFN/lm_head)"}
SEMANTIC = {
    "matmul (QKV/O/FFN/lm_head)": {"aten.mm.default", "aten.addmm.default"},
    "attention QK^T / S.V": {"aten.bmm.default"},
    "softmax": {"aten._softmax.default"},
    "RMSNorm": {"aten.pow.Tensor_Scalar", "aten.mean.dim", "aten.add.Tensor",
                "aten.rsqrt.default", "aten.mul.Tensor"},
    "RoPE": {"aten.cos.default", "aten.sin.default", "aten.neg.default",
             "aten.cat.default", "aten.mul.Tensor", "aten.add.Tensor"},
    "SwiGLU": {"aten.silu.default", "aten.mul.Tensor"},
    "residual add": {"aten.add.Tensor"},
    "embedding": {"aten.embedding.default"},
}

# housekeeping / shape-plumbing / causal-mask primitives — no semantic op, ignored
WHITELIST = {
    "aten.view.default", "aten._unsafe_view.default", "aten.transpose.int",
    "aten.t.default", "aten._to_copy.default", "aten.expand.default",
    "aten.slice.Tensor", "aten.arange.default", "aten.clone.default",
    "aten.where.self", "aten.unsqueeze.default", "aten.alias.default",
    "aten.lift_fresh.default", "aten.scalar_tensor.default", "aten.detach.default",
    "aten.eq.Tensor", "aten.ne.Scalar", "aten.le.Tensor", "aten.sub.Tensor",
    "aten.cumsum.default", "aten.bitwise_and.Tensor", "aten.new_ones.default",
    "aten.index.Tensor", "prim.device.default",
}


def check(distinct):
    distinct = set(distinct)
    covered = {name: (bool(prims & distinct) if name in ALT else prims.issubset(distinct))
               for name, prims in SEMANTIC.items()}
    semantic_prims = set().union(*SEMANTIC.values())
    unmatched = sorted(distinct - semantic_prims - WHITELIST)
    return {"covered": covered, "unmatched": unmatched,
            "all_semantic_covered": all(covered.values())}


def main():
    ok = True
    for f in sorted(OUT.glob("*.json")):
        doc = json.loads(f.read_text())
        res = check(doc["distinct_ops"])
        doc["expected_ops_check"] = res
        f.write_text(json.dumps(doc, indent=1))
        passed = res["all_semantic_covered"] and not res["unmatched"]
        ok = ok and passed
        miss = [k for k, v in res["covered"].items() if not v]
        print(f"{f.name}: {'PASS' if passed else 'FAIL'} "
              f"(missing semantic: {miss or 'none'}; unmatched: {res['unmatched'] or 'none'})")
    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
