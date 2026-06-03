"""Phase 0.1 step 7 — export the Phase 0.2 micro-benchmark sweep matrix.

Collects the distinct (op, in_shapes, out_shape) signatures across all model
inventories, grouped by op category. This is the set of (op, shape) Phase 0.2
must measure per-unit timing for. Housekeeping/plumbing aten ops are excluded
(host-side, not worth micro-benchmarking).

Note: operands are NOT tagged activation-vs-weight — under FakeTensorMode params
are fake tensors and appear transposed at the aten boundary, so the tag is
unreliable. The full (M,K,N) signature is recorded instead, which is exactly the
matmul/bmm spec a Phase 0.2 benchmark needs.

Run: ./.venv/bin/python tools/trace_export/sweep_matrix.py
"""
import json
from pathlib import Path

OUT = Path("measurements/op_inventory")

CATEGORY = {
    "matmul": {"aten.mm.default", "aten.addmm.default"},
    "attention": {"aten.bmm.default"},  # QK^T and S.V
    "softmax": {"aten._softmax.default"},
    "norm": {"aten.rsqrt.default", "aten.mean.dim", "aten.pow.Tensor_Scalar"},
    "rope": {"aten.cos.default", "aten.sin.default", "aten.neg.default", "aten.cat.default"},
    "elementwise": {"aten.mul.Tensor", "aten.add.Tensor", "aten.silu.default", "aten.sub.Tensor"},
    "embedding": {"aten.embedding.default"},
}
OP2CAT = {op: cat for cat, ops in CATEGORY.items() for op in ops}


def main():
    buckets = {cat: {} for cat in CATEGORY}  # cat -> {sig_key: sig}
    for f in sorted(OUT.glob("*.json")):
        if f.name in ("workload_lengths.json", "sweep_matrix.json"):
            continue
        doc = json.loads(f.read_text())
        for phase in doc["inventory"].values():
            for sigs in phase.values():
                for r in sigs:
                    cat = OP2CAT.get(r["op"])
                    if cat is None:
                        continue  # housekeeping -> excluded
                    key = (r["op"], json.dumps(r["in_shapes"]), json.dumps(r["out_shape"]))
                    buckets[cat].setdefault(key, {"op": r["op"], "in_shapes": r["in_shapes"],
                                                  "out_shape": r["out_shape"]})
    matrix = {cat: list(d.values()) for cat, d in buckets.items()}
    total = sum(len(v) for v in matrix.values())
    doc = {"note": "distinct (op, in_shapes, out_shape) to micro-benchmark in Phase 0.2; "
                   "housekeeping aten ops excluded; operands not activation/weight-tagged",
           "counts": {c: len(v) for c, v in matrix.items()}, "total": total, "matrix": matrix}
    (OUT / "sweep_matrix.json").write_text(json.dumps(doc, indent=1))
    print("sweep_matrix.json counts:", doc["counts"], "total", total)
    missing = [c for c in ("matmul", "attention", "norm", "rope", "elementwise") if not matrix[c]]
    print("MISSING categories:", missing or "none")


if __name__ == "__main__":
    main()
