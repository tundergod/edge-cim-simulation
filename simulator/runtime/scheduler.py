"""M6 — op->unit scheduler (Phase 2.2b): a `Scheduler` ABC + plugins.

`Scheduler.assign(dag, cfg) -> dag` is a pure, idempotent annotator that sets only
`node.unit` + `node.mem_domain` (it never touches category/wl/deps). `AllCimScheduler`
is the 2.1/2.2a all-CIM baseline (mirrors the L4 vendor all-AIPU INT8 placement: matmul
on CIM, support ops on CPU, attention on CIM as memory-bound, kv-append + embedding
through memory) — the ONLY hard-silicon-gated path (AllCim L4). `CimHeteroScheduler`
(the project's CIM-INT8 matmul × GPU-FP16 attention mixed-precision config) lands later
in this wave and is SIMULATED. The thin `all_cim_assign` wrapper is kept for callers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

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


class Scheduler(ABC):
    """Pure, idempotent op->unit placement. `assign` sets ONLY node.unit + node.mem_domain
    (never category/wl/deps) and returns the dag; calling it twice gives the same result."""
    name: str = ""

    @abstractmethod
    def assign(self, dag, cfg=None):
        raise NotImplementedError


class AllCimScheduler(Scheduler):
    """All-CIM baseline (the L4-gated all-AIPU INT8 placement)."""
    name = "all_cim"

    def assign(self, dag, cfg=None):
        for n in dag.nodes:
            n.unit = ALLCIM_MAP.get(n.category, "cpu")
            n.mem_domain = _residency(n)
        return dag


SCHEDULERS = {"all_cim": AllCimScheduler()}


def all_cim_assign(dag):
    """Thin wrapper kept for existing callers (= AllCimScheduler().assign)."""
    return AllCimScheduler().assign(dag)


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
