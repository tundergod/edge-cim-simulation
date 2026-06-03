"""Phase 0.1 step 7 — export the Phase 0.2 micro-benchmark sweep matrix.

Collects the distinct (op, in_shapes, out_shape) signatures across all model
inventories, grouped by semantic op category. This is the set of (op, shape)
Phase 0.2 must measure per-unit timing for. Host-side / plumbing ops are excluded.

Categorization (issue #5): unambiguous aten ops are bucketed by name; **overloaded
ops** (bmm/add/mul/cat/sub — emitted by more than one semantic op) are bucketed by
their *source origin* (`src`, the emitting transformers module/function recorded by
the tracer), not by name. That splits `bmm` into attention vs RoPE-freq and drops
host-side mask/position add/mul/cat — generally, for ALL overloaded ops.

Note: operands are NOT tagged activation-vs-weight (unreliable under FakeTensor —
params are fake & transposed at the aten boundary); the full (M,K,N) signature is
recorded, which is exactly the matmul/bmm spec a Phase 0.2 benchmark needs.

Run: ./.venv/bin/python tools/trace_export/sweep_matrix.py
"""
import json
from pathlib import Path

OUT = Path("measurements/op_inventory")

# unambiguous ops: one semantic category each (verified single-origin)
BY_NAME = {
    "aten.mm.default": "matmul", "aten.addmm.default": "matmul",       # q/k/v/o, gate/up/down, lm_head
    "aten._softmax.default": "softmax",
    "aten.embedding.default": "embedding",
    "aten.silu.default": "ffn",                                        # SwiGLU activation
    "aten.rsqrt.default": "norm", "aten.mean.dim": "norm", "aten.pow.Tensor_Scalar": "norm",
    "aten.cos.default": "rope", "aten.sin.default": "rope", "aten.neg.default": "rope",  # single-origin RoPE prims
}
# overloaded ops: categorize by source origin (src) instead of name
OVERLOADED = {"aten.bmm.default", "aten.add.Tensor", "aten.mul.Tensor",
              "aten.cat.default", "aten.sub.Tensor"}


def src_category(src):
    if not src:
        return None
    if src.endswith("RotaryEmbedding.forward") or src in ("apply_rotary_pos_emb", "rotate_half"):
        return "rope"
    if src == "eager_attention_forward" or src.endswith("Attention.forward"):  # QK^T/SV/scale/mask-add + KV-cache cat
        return "attention"
    if src.endswith("RMSNorm.forward"):
        return "norm"
    if src.endswith("MLP.forward"):
        return "ffn"
    if src.endswith("DecoderLayer.forward"):
        return "residual"
    return None  # *Model.forward / sdpa_mask / find_packed_sequence_indices / masking -> host-side, drop


def categorize(r):
    op = r["op"]
    if op in BY_NAME:
        return BY_NAME[op]
    if op in OVERLOADED:
        return src_category(r.get("src"))
    return None  # housekeeping aten ops -> excluded


def main():
    buckets = {}  # cat -> {sig_key: sig}
    for f in sorted(OUT.glob("*.json")):
        if f.name in ("workload_lengths.json", "sweep_matrix.json"):
            continue
        doc = json.loads(f.read_text())
        for phase in doc["inventory"].values():
            for sigs in phase.values():
                for r in sigs:
                    cat = categorize(r)
                    if cat is None:
                        continue
                    key = (r["op"], json.dumps(r["in_shapes"]), json.dumps(r["out_shape"]))
                    buckets.setdefault(cat, {}).setdefault(
                        key, {"op": r["op"], "in_shapes": r["in_shapes"], "out_shape": r["out_shape"]})
    matrix = {cat: list(d.values()) for cat, d in sorted(buckets.items())}
    total = sum(len(v) for v in matrix.values())
    doc = {"note": "distinct (op, in_shapes, out_shape) to micro-benchmark in Phase 0.2; "
                   "unambiguous ops bucketed by name, overloaded ops by source origin (#5); "
                   "host-side ops excluded",
           "counts": {c: len(v) for c, v in matrix.items()}, "total": total, "matrix": matrix}
    (OUT / "sweep_matrix.json").write_text(json.dumps(doc, indent=1))
    print("sweep_matrix.json counts:", doc["counts"], "total", total)
    missing = [c for c in ("matmul", "attention", "softmax", "norm", "rope", "ffn") if c not in matrix]
    print("MISSING categories:", missing or "none")


if __name__ == "__main__":
    main()
