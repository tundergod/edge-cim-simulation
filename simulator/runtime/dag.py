"""Per-token op-DAG types shared by M5 / M6 / M3 (Phase 2.1).

`OpNode` = one op to time (wraps the frozen `Workload`). `Dag` = nodes +
data-dependency edges. M5 (workload.py) produces a Dag; M6 (scheduler) sets
`node.unit`; M3 (events.py) walks it. Kept deliberately minimal: a node, a dep
list, a successor index, an acyclicity check — no scheduling logic here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from simulator.models.engine import Workload

# op categories M5 emits (from the Phase-0.2 9-class categorizer) + 'convert'
# (precision-boundary cast, inserted by M6 in Wave 2.2).
CATEGORIES = (
    "matmul", "attention", "norm", "rope", "ffn", "softmax",
    "residual", "embedding", "sampling", "kv_cache", "convert",
)


@dataclass
class OpNode:
    """One op in the per-token DAG. `wl` is the frozen Workload a unit engine
    eats; `unit` is filled by the scheduler; `bytes_streamed` is the
    shared-memory traffic this op pulls (used by the M3 contention model).

    Wave 2.2 value-flow: `in_values` are the value-ids this op consumes and
    `out_value` the one it produces (at node granularity value-id == producer
    node id, so in_values == deps); `precision` is the SIMULATED placement
    precision (ADR-0004c, fixture_io.PRECISION_CONTRACT); `pricing_group` lets
    2.2b price a composite op-pair (QK^T + S·V) once (R2)."""
    id: int
    category: str
    wl: Workload
    deps: list = field(default_factory=list)   # predecessor node ids
    unit: str | None = None                     # 'cim'|'gpu'|'npu'|'cpu'|'mem' (set by scheduler)
    bytes_streamed: int = 0
    in_values: list = field(default_factory=list)   # value-ids consumed (== deps in 2.2a)
    out_value: int | None = None                    # value-id produced (== id in 2.2a)
    precision: str | None = None                    # simulated placement precision
    pricing_group: int | None = None                # composite-pricing group (2.2b R2)


class Dag:
    """Acyclic op graph. Built once from a node list; exposes a successor
    index, roots, and an acyclicity check. Validates ids + dep references at
    construction (fail-loud)."""

    def __init__(self, nodes):
        self.nodes = list(nodes)
        self._by_id = {n.id: n for n in self.nodes}
        if len(self._by_id) != len(self.nodes):
            raise ValueError("duplicate OpNode id in DAG")
        self._succ = {n.id: [] for n in self.nodes}
        for n in self.nodes:
            for d in n.deps:
                if d not in self._by_id:
                    raise ValueError(f"node {n.id} references unknown dep {d}")
                self._succ[d].append(n.id)

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, nid) -> OpNode:
        return self._by_id[nid]

    def successors(self, nid):
        return self._succ[nid]

    def roots(self):
        """Node ids with no predecessors (dispatch-ready at t=0)."""
        return [n.id for n in self.nodes if not n.deps]

    def is_acyclic(self) -> bool:
        """Kahn's algorithm: True iff a topological order covers every node."""
        indeg = {n.id: len(n.deps) for n in self.nodes}
        queue = [nid for nid, d in indeg.items() if d == 0]
        seen = 0
        while queue:
            nid = queue.pop()
            seen += 1
            for s in self._succ[nid]:
                indeg[s] -= 1
                if indeg[s] == 0:
                    queue.append(s)
        return seen == len(self.nodes)


def wl_is_sane(wl: Workload) -> bool:
    """Shape sanity for a node's Workload: non-negative dims, a known dtype."""
    return (
        wl.M >= 0 and wl.K >= 0 and wl.N >= 0 and wl.kv >= 0
        and wl.nbytes >= 0 and wl.heads >= 1 and wl.layers >= 1
        and wl.dtype in ("int8", "int16", "fp16", "fp32")
        and isinstance(wl.op, str) and wl.op != ""
    )
