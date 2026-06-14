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


def _residency(node):
    """Memory DOMAIN of an op (R7). The Metis Card is all-AIPU with 16 GiB on-card LPDDR4x
    (verified via axdevice): weights/KV/embedding are resident there and stream at ~24.2 GB/s
    -> `dram`. CPU-support activations (norm/softmax/...) are a few KB/token and live in
    on-chip SRAM, priced INSIDE m4_cpu (max(compute,cache_mem)+overhead) -> `cpu_cache`, so
    they must NOT also be metered into the DRAM pool (the S-dc double-count). NB: the Card has
    no RK3588; m4_cpu(A76) is a labelled PROXY for the AIPU's on-chip support (≈0.4% of decode).
    No traffic -> `none`. No 'local' domain (scope-out)."""
    if node.bytes_streamed == 0:
        return "none"
    if node.unit == "cpu":
        return "cpu_cache"
    return "dram"


def all_cim_assign(dag):
    """Annotate every node.unit per the all-CIM mapping AND its memory domain (in place);
    returns dag."""
    for n in dag.nodes:
        n.unit = ALLCIM_MAP.get(n.category, "cpu")
        n.mem_domain = _residency(n)
    return dag


def domain_byte_audit(dag):
    """Byte-accounting oracle (R7): every node's bytes are attributed to exactly ONE memory
    domain (mutual exclusion, no op_bytes double-count). Returns per-domain totals + `ok`. In
    AllCim there are no cross-chip transfer_bytes (single accelerator), so total == op bytes."""
    out = {"dram": 0, "cpu_cache": 0, "none": 0, "total": 0}
    for n in dag.nodes:
        out[n.mem_domain] = out.get(n.mem_domain, 0) + n.bytes_streamed
        out["total"] += n.bytes_streamed
    out["ok"] = (out["dram"] + out["cpu_cache"] + out["none"] == out["total"])
    return out
