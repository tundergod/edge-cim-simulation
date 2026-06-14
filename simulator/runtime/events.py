"""M3 — lightweight non-cycle-accurate discrete-event engine (Phase 2.1, ADR-0001).

Walks a per-token op DAG. Compute units are exclusive serial servers (one op at a
time per unit; busy_until); different units run concurrently. The shared memory is
a FLUID processor-sharing resource: all in-flight memory streams get an equal
fair-share of the bandwidth, and the share is RE-COMPUTED at every event (a stream
finishing frees bandwidth for the rest). A node's compute and memory overlap
(double-buffering); the node completes when BOTH finish, i.e. max(compute, memory).

Per-op COMPUTE latency comes from the Phase-1 unit models via
`platform.compute_us(node)` — the engine never computes a latency itself. Memory is
metered through `SharedBandwidth` (the saturating-knee model). Ablations:
`concurrency=False` collapses units onto one serial clock; `contention=False`
removes the knee (memory sums linearly); `price_compute=False` zeroes compute
(memory-only ablation).

The fluid loop advances time event-to-event (dispatch / compute-finish / memory
completion), so the fair-share is exact for any overlap pattern (not just the
serial AllCim path). Event count ~10^2-10^3/token (ADR-0003).
"""
from __future__ import annotations

_EPS = 1e-9        # time epsilon (microseconds)
_BYTE_EPS = 1.0    # bytes: a stream with < 1 byte remaining is done (guards FP residue ~1e-6
                   # on byte counts ~1e10, which a tiny time-epsilon would never clear -> #56 stall)
_INF = float("inf")


def _validate_mem_domains(dag):
    """Fail-loud: every node must carry a valid memory domain, and a byte-streaming node may
    not be tagged 'none' — otherwise the DRAM-metering rule (meter unless cpu_cache) would
    SILENTLY treat a mis-domained node as DRAM and emit a plausible-but-wrong latency."""
    for n in dag.nodes:
        if n.mem_domain not in ("dram", "cpu_cache", "none"):
            raise ValueError(f"M3: node {n.id} has invalid/missing mem_domain {n.mem_domain!r} "
                             f"(expected dram/cpu_cache/none — run a scheduler before the engine)")
        if n.bytes_streamed > 0 and n.mem_domain == "none":
            raise ValueError(f"M3: node {n.id} streams {n.bytes_streamed} bytes but mem_domain='none' "
                             f"(a byte-streaming op must reside in dram or cpu_cache)")


def run_serial(dag, platform, bw, *, price_compute=True):
    """SINGLE-ACCELERATOR execution (no cross-op pipeline): each op occupies the one
    accelerator for max(compute, memory) — intra-op double-buffering only — and the
    token latency is their sum. This is the FAITHFUL all-AIPU (AllCim) decode model. The
    L4 anchor is the Metis Card's 1c (SINGLE AIPU core) axllm decode: one core runs ops in
    dataflow order with no cross-core op pipeline, and the measured tok/s sits AT/BELOW the
    serial no-cross-op-overlap bound (1B measured 13.07 < this model's 14.47 tok/s) with a
    tiny 4c/1c ratio (~1.1x — decode is on-card-DRAM-bandwidth-bound), i.e. the silicon shows
    no cross-op compute/memory hiding. (`double_buffer` overlaps only host<->device PCIe,
    negligible for decode.) concurrency/contention are no-ops here (one resource, k=1).
    Order-independent (a sum of per-node maxima)."""
    _validate_mem_domains(dag)
    total = 0.0
    for n in dag.nodes:
        c = float(platform.compute_us(n)) if price_compute else 0.0
        # only DRAM-domain bytes hit the 24.2 GB/s wall; cpu_cache bytes are already inside
        # compute_us via m4_cpu (the S-dc double-count fix).
        m = bw.stream_us(n.bytes_streamed, 1) if (n.bytes_streamed > 0 and n.mem_domain != "cpu_cache") else 0.0
        total += max(c, m)
    return total


def run_dag(dag, platform, bw, *, concurrency=True, contention=True, price_compute=True,
            pipeline=True):
    """Return the token latency (microseconds) to execute the whole DAG.
    `price_compute=False` zeroes every op's compute term (memory-only ablation).
    `pipeline=False` runs the SINGLE-ACCELERATOR serial model (run_serial) — no cross-op
    overlap, matching the measured all-AIPU silicon; `pipeline=True` (default) uses the
    fluid concurrent event loop (cross-op overlap), a SIMULATED forward-looking mode."""
    # fail-loud preconditions (#56): never return a plausible latency for an invalid run
    if not dag.is_acyclic():
        raise ValueError("M3: cyclic DAG — a dependency cycle cannot be scheduled")
    if any(n.unit is None for n in dag.nodes):
        raise ValueError("M3: unscheduled node(s) with unit=None — run a scheduler before run_dag")
    _validate_mem_domains(dag)
    if any(n.bytes_streamed > 0 and n.mem_domain != "cpu_cache" for n in dag.nodes) and bw.eff_BW <= 0:
        raise ValueError("M3: DRAM memory traffic present but SharedBandwidth eff_BW <= 0 "
                         "(degenerate bandwidth would stall the simulation)")
    if not pipeline:
        return run_serial(dag, platform, bw, price_compute=price_compute)
    n_total = len(dag.nodes)
    indeg = {n.id: len(n.deps) for n in dag.nodes}
    unit_free = {}                       # unit clock -> next-free time (compute serialization)
    pending = {nid for nid, d in indeg.items() if d == 0}   # deps satisfied, not yet dispatched
    started, done = set(), set()
    active_mem = {}                      # nid -> remaining bytes (memory stream in flight)
    compute_done_at = {}                # nid -> absolute time its compute finishes
    clock = 0.0
    last = 0.0

    def ukey(u):
        return u if concurrency else "_serial"   # u is validated non-None above

    def compute_us(node):
        return float(platform.compute_us(node)) if price_compute else 0.0

    def finish(nid):
        nonlocal last
        if nid in done or nid not in started:
            return
        if compute_done_at[nid] <= clock + _EPS and nid not in active_mem:
            done.add(nid)
            last = max(last, clock)
            for s in dag.successors(nid):
                indeg[s] -= 1
                if indeg[s] == 0:
                    pending.add(s)

    def rate_bytes_per_us():
        # equal fair-share per active memory stream (GB/s -> bytes/us = x1e3)
        return bw.per_stream_GBs(len(active_mem), contention=contention) * 1e3

    while pending or active_mem or (started - done):
        # 1) dispatch every pending node whose unit is free (compute serialization)
        progressed = True
        while progressed:
            progressed = False
            for nid in list(pending):
                u = ukey(dag[nid].unit)
                if unit_free.get(u, 0.0) <= clock + _EPS:
                    node = dag[nid]
                    pending.discard(nid)
                    started.add(nid)
                    ct = compute_us(node)
                    unit_free[u] = clock + ct
                    compute_done_at[nid] = clock + ct
                    # only DRAM-domain bytes contend on the shared channel; cpu_cache bytes
                    # are priced inside compute_us (m4_cpu) and don't drag the DRAM pool.
                    if node.bytes_streamed > 0 and node.mem_domain != "cpu_cache":
                        active_mem[nid] = float(node.bytes_streamed)
                    progressed = True
        for nid in list(started):
            finish(nid)
        if not (pending or active_mem or (started - done)):
            break

        # 2) advance to the next event: memory completion / compute finish / unit free
        nxt = _INF
        if active_mem:
            r = rate_bytes_per_us()
            if r > 0:
                nxt = min(nxt, min(active_mem.values()) / r)
        for nid in started - done:
            if compute_done_at[nid] > clock + _EPS:
                nxt = min(nxt, compute_done_at[nid] - clock)
        for nid in pending:
            uf = unit_free.get(ukey(dag[nid].unit), 0.0)
            if uf > clock + _EPS:
                nxt = min(nxt, uf - clock)
        if nxt is _INF or nxt <= _EPS:
            # work remains (the clean all-done break above did not fire) but no event can
            # advance time -> a genuine stall; fail loud rather than return `last` (#56).
            unresolved = len(pending) + len(active_mem) + len(started - done)
            raise RuntimeError(
                f"M3: event loop stalled — {unresolved} node-state(s) unresolved with no "
                f"progressible event (pending={len(pending)}, active_mem={len(active_mem)}, "
                f"running={len(started - done)}); likely degenerate bandwidth or an "
                f"unsatisfiable dependency.")

        # 3) drain memory at the (constant-over-interval) fair-share rate, then finish
        clock += nxt
        if active_mem:
            drained = nxt * rate_bytes_per_us()
            for nid in list(active_mem):
                active_mem[nid] -= drained
                if active_mem[nid] <= _BYTE_EPS:
                    del active_mem[nid]
        for nid in list(started):
            finish(nid)

    # completion invariant (#56): every node must have finished
    if len(done) != n_total:
        raise RuntimeError(f"M3: incomplete simulation — {n_total - len(done)} of {n_total} "
                           f"nodes never completed")
    return last
