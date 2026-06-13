"""M6 (Phase 2.1 minimal) — AllCim static op->unit mapping.

The all-CIM baseline mirrors the L4 vendor (all-AIPU INT8) compute placement:
matmul on CIM, support ops (norm/rope/ffn/softmax/residual) on CPU, attention on
CIM (no separate CIM-attention compute model in 2.1 -> treated as memory-bound
via its bytes), kv-append + embedding through memory. The decode mechanism is
priced from CIM-GEMV (L1) + the mem_lpddr4x 24.2 GB/s wall + explicit
support/attention/kv terms — there is NO e2e-fitted bandwidth (fit_BW_GBs=18.33)
anywhere in this path. The `Scheduler` ABC + SOTA plugins (HeteroInfer) arrive in
Wave 2.2.
"""
from __future__ import annotations

ALLCIM_MAP = {
    "matmul": "cim",
    "attention": "cim",
    "softmax": "cpu", "norm": "cpu", "rope": "cpu", "ffn": "cpu", "residual": "cpu",
    "kv_cache": "mem", "embedding": "mem",
}


def all_cim_assign(dag):
    """Annotate every node.unit per the all-CIM mapping (in place); returns dag."""
    for n in dag.nodes:
        n.unit = ALLCIM_MAP.get(n.category, "cpu")
    return dag
