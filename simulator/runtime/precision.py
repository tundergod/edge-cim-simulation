"""M6 precision-boundary conversion ops (Phase 2.2b, ADR-0004).

A heterogeneous placement that lands matmul on the CIM (INT8) and attention on the
GPU (FP16) crosses precision boundaries. `insert_conversions` adds an explicit
`convert` OpNode on each value-flow edge (producer -> consumer) where EXACTLY ONE
endpoint is on the GPU (the fp16 attention island) AND the two sides differ in
sim_precision (int8<->fp16) — the dequant feeding the GPU and the requant leaving it.

Keyed on the (GPU unit-pair, precision) of the edge, NOT on the precision delta alone:
AllCim (no node on the GPU) inserts ZERO conversions, so its L4 path is byte-identical.
A conversion is a deterministic MEMORY-BOUND cast (read producer-precision bytes + write
consumer-precision bytes over the produced value's `out_elems`), priced by the existing
M3 memory pool (Platform.price returns compute 0); no new measurement / parameter
(ADR-0004 revision). Returns a REBUILT Dag (fresh successor index); idempotent.
"""
from __future__ import annotations

from dataclasses import replace

from simulator.models.engine import Workload
from simulator.runtime.dag import OpNode, Dag

_PREC_BYTES = {"int8": 1, "fp16": 2, "fp32": 4}


def _needs_conversion(p, c):
    """True iff edge p->c crosses the GPU boundary with a precision change (int8<->fp16)."""
    if p.category == "convert" or c.category == "convert":
        return False                                 # never convert a convert's own edges (idempotent)
    if p.precision == c.precision:
        return False
    return ((p.unit == "gpu") + (c.unit == "gpu")) == 1   # exactly one endpoint on the GPU


def _convert_node(cid, p, c, placement):
    """A memory-bound cast of p's value (p.precision -> c.precision). `placement` picks the
    unit that pays (default the consumer); mem_domain follows the same residency rule as any op."""
    from simulator.runtime.scheduler import _residency   # deferred: avoids a scheduler<->precision cycle
    nbytes = p.out_elems * (_PREC_BYTES[p.precision] + _PREC_BYTES[c.precision])
    unit = p.unit if placement == "producer" else c.unit
    node = OpNode(id=cid, category="convert",
                  wl=Workload(op="convert", nbytes=nbytes, dtype=c.precision,
                              extra={"from": p.precision, "to": c.precision}),
                  deps=[p.id], unit=unit, bytes_streamed=nbytes, in_values=[p.id],
                  out_value=cid, precision=c.precision, out_elems=p.out_elems)
    node.mem_domain = _residency(node)
    return node


def insert_conversions(dag, *, placement="consumer"):
    """Return a REBUILT Dag with a `convert` node on every int8<->fp16 GPU-boundary edge.
    Originals are not mutated (fresh OpNode copies with rewired deps), so the call is pure
    and idempotent. One convert per crossing (producer, consumer) edge."""
    if placement not in ("consumer", "producer"):
        raise ValueError(f"insert_conversions: placement must be 'consumer' or 'producer', "
                         f"got {placement!r}")
    by_id = {n.id: n for n in dag.nodes}
    next_id = max(by_id) + 1 if by_id else 0
    converts = {}                                    # (p_id, c_id) -> convert node
    new_nodes = []
    for c in dag.nodes:
        deps = []
        for d in c.deps:
            if _needs_conversion(by_id[d], c):
                key = (d, c.id)
                if key not in converts:
                    converts[key] = _convert_node(next_id, by_id[d], c, placement)
                    next_id += 1
                deps.append(converts[key].id)
            else:
                deps.append(d)
        new_nodes.append(replace(c, deps=deps, in_values=list(deps)))
    new_nodes.extend(converts.values())
    return Dag(new_nodes)
