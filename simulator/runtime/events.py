"""M3 — lightweight non-cycle-accurate discrete-event engine (Phase 2.1, ADR-0001).

Walks a per-token op DAG. Each compute unit serializes its ops (busy_until);
different units run concurrently. Per-op COMPUTE latency comes from the Phase-1
unit models via `platform.compute_us(node)` — the engine NEVER computes a
latency itself. Each op also streams `node.bytes_streamed` through the one
contended resource (`SharedBandwidth`); a node's duration is
`max(compute, memory)` (double-buffering / overlap-within-node, LLMCompass).

Hand-written heapq loop (~event count 10^2-10^3/token; ADR-0003). Two ablation
flags: `concurrency=False` collapses all units onto one serial clock;
`contention=False` removes the knee (memory sums linearly).

Modeling note (non-cycle-accurate): the number of concurrent memory streams is
evaluated at each op's dispatch and held for its duration (no mid-flight
re-share). For a serial DAG only one stream is ever active (full eff_BW); for
the synthetic concurrent workloads in validate_contention the streams co-start,
so the dispatch-time count is exact.
"""
from __future__ import annotations

import heapq


def run_dag(dag, platform, bw, *, concurrency=True, contention=True):
    """Return the token latency (microseconds) to execute the whole DAG."""
    indeg = {n.id: len(n.deps) for n in dag.nodes}
    free = {}                       # unit clock: busy_until per unit (or one shared clock)
    active_mem = {}                 # nid -> memory-stream finish time (for concurrency count)
    heap = []                       # (node_done_time, seq, nid)
    seq = 0
    last_finish = 0.0

    def ukey(u):
        return u if concurrency else "_serial"

    def n_active_mem(at):
        # prune finished streams, then count those still in flight
        for k in [k for k, f in active_mem.items() if f <= at]:
            del active_mem[k]
        return len(active_mem)

    def dispatch(nid, now):
        nonlocal seq, last_finish
        node = dag[nid]
        u = node.unit or "cpu"
        start = max(now, free.get(ukey(u), 0.0))
        compute_us = float(platform.compute_us(node))
        if node.bytes_streamed > 0:
            k = n_active_mem(start) + 1
            mem_us = bw.stream_us(node.bytes_streamed, k, contention=contention)
        else:
            mem_us = 0.0
        dur = max(compute_us, mem_us)
        finish = start + dur
        free[ukey(u)] = start + compute_us          # unit frees after compute (memory overlaps)
        if node.bytes_streamed > 0:
            active_mem[nid] = start + mem_us
        seq += 1
        heapq.heappush(heap, (finish, seq, nid))
        last_finish = max(last_finish, finish)

    for nid in [n.id for n in dag.nodes if indeg[n.id] == 0]:
        dispatch(nid, 0.0)

    while heap:
        finish, _, nid = heapq.heappop(heap)
        active_mem.pop(nid, None)
        for s in dag.successors(nid):
            indeg[s] -= 1
            if indeg[s] == 0:
                dispatch(s, finish)

    return last_finish
