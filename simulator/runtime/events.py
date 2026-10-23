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

The fluid loop advances time event-to-event (dispatch / fixed-delay end / compute-finish /
memory completion), so the fair-share is exact for any overlap pattern (not just the
serial AllCim path). Event count ~10^2-10^3/token (ADR-0003).

FIXED PRE-OP DELAY (ADR-0009 revision F1): `platform.price(node)` may return `fixed_us`
(a scheduler-attached sync / launch overhead). It is a SERIAL delay at the node's start —
the unit is busy, nothing streams — after which compute and memory overlap as usual, so
node duration = fixed + max(compute, memory) in BOTH run_dag and run_serial (the solver uses
the same rule). PER-UNIT STREAM CAPS (F10/F15): with a profile's `SharedBandwidth.unit_caps`
the streams in flight are grouped by unit (a `mem` node's stream is attributed to the compute
unit that consumes — else produces — its value) and rated by `SharedBandwidth.share`.
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


def _critical_path_us(dag, dur):
    """Longest dependency path with per-node durations `dur` (resource-free lower bound)."""
    order = []
    indeg = {n.id: len(n.deps) for n in dag.nodes}
    ready = [nid for nid, d in indeg.items() if d == 0]
    while ready:
        nid = ready.pop()
        order.append(nid)
        for s in dag.successors(nid):
            indeg[s] -= 1
            if indeg[s] == 0:
                ready.append(s)
    finish = {}
    for nid in order:
        start = max((finish[d] for d in dag[nid].deps), default=0.0)
        finish[nid] = start + dur[nid]
    return max(finish.values(), default=0.0)


def _fill_stats(stats, dag, latency_us, unit_busy, dur, bytes_drained, mem_active_us):
    """ADR-0009 run statistics: per-unit busy time, sync/merge counts, achieved aggregate
    bandwidth (bytes drained / time with >= 1 active DRAM stream), critical path."""
    syncs = [n.wl.extra.get("sync_us", 0.0) for n in dag.nodes]
    stats.update({
        "latency_us": latency_us,
        "unit_busy_us": dict(sorted(unit_busy.items())),
        "unit_utilisation": {u: (b / latency_us if latency_us > 0 else 0.0) for u, b in sorted(unit_busy.items())},
        "n_sync": sum(1 for x in syncs if x > 0),
        "sync_us": float(sum(syncs)),
        "n_merge": sum(1 for n in dag.nodes if n.category == "merge"),
        "dram_bytes": int(bytes_drained),
        "mem_active_us": mem_active_us,
        "achieved_GBs": (bytes_drained / mem_active_us * 1e-3) if mem_active_us > 0 else 0.0,
        "critical_path_us": _critical_path_us(dag, dur),
    })
    return stats


def _price(platform, node, price_compute):
    """(fixed_us, compute_us) of one node from the pricing contract. `fixed_us` (default 0) is
    the serial pre-op delay; `price_compute=False` zeroes the compute term only (the fixed delay
    is a sync/launch cost, not unit compute). A platform without `price` (engine stubs) prices
    compute through `compute_us` with no fixed delay."""
    pr = getattr(platform, "price", None)
    if pr is None:
        return 0.0, (float(platform.compute_us(node)) if price_compute else 0.0)
    r = pr(node)
    return float(r.get("fixed_us", 0.0)), (float(r["latency_us"]) if price_compute else 0.0)


def _dram(node):
    """True iff the node's bytes are metered on the shared DRAM pool (cpu_cache bytes are priced
    inside m4_cpu, the S-dc double-count fix). WATCH (#67): 24.2 was fit on WEIGHT bytes only;
    KV/attn/embedding stream through this same pool -> latent double-count (not in scope)."""
    return node.bytes_streamed > 0 and node.mem_domain != "cpu_cache"


def _stream_owner(dag, node):
    """Unit whose single-stream cap a node's DRAM stream counts against (F15): the node's own
    unit, or for a `mem` node (embedding gather / KV append / a cast on mem) the first non-mem
    consumer's unit, else the first non-mem producer's unit, else 'mem'."""
    if node.unit != "mem":
        return node.unit
    for s in dag.successors(node.id):
        if dag[s].unit != "mem":
            return dag[s].unit
    for d in node.deps:
        if dag[d].unit != "mem":
            return dag[d].unit
    return "mem"


def run_serial(dag, platform, bw, *, price_compute=True, stats=None):
    """SINGLE-ACCELERATOR execution (no cross-op pipeline): each op occupies the one
    accelerator for fixed + max(compute, memory) — intra-op double-buffering only — and the
    token latency is their sum. This is the FAITHFUL all-AIPU (AllCim) decode model. The
    L4 anchor is the Metis Card's 1c (SINGLE AIPU core) axllm decode: one core runs ops in
    dataflow order with no cross-core op pipeline, and the measured tok/s sits AT/BELOW the
    serial no-cross-op-overlap bound (1B measured 13.07 < this model's 14.47 tok/s) with a
    tiny 4c/1c ratio (~1.1x — decode is on-card-DRAM-bandwidth-bound), i.e. the silicon shows
    no cross-op compute/memory hiding. (`double_buffer` overlaps only host<->device PCIe,
    negligible for decode.) concurrency/contention are no-ops here (one resource, k=1).
    Order-independent (a sum of per-node maxima). The legacy AllCim placement attaches no
    `fixed_us`, so its sum is byte-identical to the Phase-2.1 formula."""
    _validate_mem_domains(dag)
    if any(_dram(n) for n in dag.nodes) and bw.eff_BW <= 0:
        raise ValueError("M3: DRAM memory traffic present but SharedBandwidth eff_BW <= 0 "
                         "(degenerate bandwidth would stall the simulation)")
    total = 0.0
    unit_busy, dur, bytes_drained, mem_us = {}, {}, 0, 0.0
    for n in dag.nodes:
        fixed, c = _price(platform, n, price_compute)
        # only DRAM-domain bytes hit the memory wall; cpu_cache bytes are already inside
        # compute_us via m4_cpu (the S-dc double-count fix).
        dram = _dram(n)
        m = bw.stream_us(n.bytes_streamed, 1) if dram else 0.0
        total += fixed + max(c, m)
        if stats is not None:
            unit_busy[n.unit] = unit_busy.get(n.unit, 0.0) + fixed + c
            dur[n.id] = fixed + max(c, m)
            if dram:
                bytes_drained += n.bytes_streamed
                mem_us += m
    if stats is not None:
        _fill_stats(stats, dag, total, unit_busy, dur, bytes_drained, mem_us)
    return total


def run_dag(dag, platform, bw, *, concurrency=True, contention=True, price_compute=True,
            pipeline=True, stats=None):
    """Return the token latency (microseconds) to execute the whole DAG.
    `price_compute=False` zeroes every op's compute term (memory-only ablation).
    `pipeline=False` runs the SINGLE-ACCELERATOR serial model (run_serial) — no cross-op
    overlap, matching the measured all-AIPU silicon; `pipeline=True` (default) uses the
    fluid concurrent event loop (cross-op overlap), a SIMULATED forward-looking mode.
    `stats` (a dict, optional) is filled with per-unit busy us, sync/merge counts, the
    bandwidth-time integral (achieved GB/s) and the critical path (ADR-0009).
    Node life cycle: pending -> dispatched (unit reserved for fixed + compute) -> activated
    (fixed delay elapsed: compute runs, DRAM stream in flight) -> done (compute finished AND
    stream drained)."""
    # fail-loud preconditions (#56): never return a plausible latency for an invalid run
    if not dag.is_acyclic():
        raise ValueError("M3: cyclic DAG — a dependency cycle cannot be scheduled")
    if any(n.unit is None for n in dag.nodes):
        raise ValueError("M3: unscheduled node(s) with unit=None — run a scheduler before run_dag")
    _validate_mem_domains(dag)
    if any(_dram(n) for n in dag.nodes) and bw.eff_BW <= 0:
        raise ValueError("M3: DRAM memory traffic present but SharedBandwidth eff_BW <= 0 "
                         "(degenerate bandwidth would stall the simulation)")
    if not pipeline:
        return run_serial(dag, platform, bw, price_compute=price_compute, stats=stats)
    n_total = len(dag.nodes)
    unit_busy, dur, bytes_drained, mem_active_us = {}, {}, 0.0, 0.0   # stats accumulators
    indeg = {n.id: len(n.deps) for n in dag.nodes}
    unit_free = {}                       # unit clock -> next-free time (compute serialization)
    pending = {nid for nid, d in indeg.items() if d == 0}   # deps satisfied, not yet dispatched
    started, activated, done = set(), set(), set()
    active_mem = {}                      # nid -> remaining bytes (memory stream in flight)
    owner = {}                           # nid -> unit whose stream cap the node's bytes count against
    fixed_done_at = {}                   # nid -> absolute time its serial pre-op delay ends
    compute_done_at = {}                # nid -> absolute time its compute finishes
    clock = 0.0
    last = 0.0

    def ukey(u):
        return u if concurrency else "_serial"   # u is validated non-None above

    def finish(nid):
        nonlocal last
        if nid in done or nid not in activated:
            return
        if compute_done_at[nid] <= clock + _EPS and nid not in active_mem:
            done.add(nid)
            last = max(last, clock)
            for s in dag.successors(nid):
                indeg[s] -= 1
                if indeg[s] == 0:
                    pending.add(s)

    def rates_bytes_per_us():
        # per-stream rate by owning unit (GB/s -> bytes/us = x1e3): legacy = equal fair-share
        # over all streams; with unit_caps = per-unit cap, then knee water-filling (F10)
        streams = {}
        for nid in active_mem:
            streams[owner[nid]] = streams.get(owner[nid], 0) + 1
        return {u: r * 1e3 for u, r in bw.share(streams, contention=contention).items()}

    while pending or active_mem or (started - done):
        # 1) dispatch every pending node whose unit is free (compute serialization), activate
        # every dispatched node whose fixed delay has elapsed, and finish zero-cost nodes in the
        # same pass — a finish can free a unit / make a successor pending, which must dispatch
        # at THIS clock (else the loop stalls).
        progressed = True
        while progressed:
            progressed = False
            for nid in list(pending):
                u = ukey(dag[nid].unit)
                if unit_free.get(u, 0.0) <= clock + _EPS:
                    node = dag[nid]
                    pending.discard(nid)
                    started.add(nid)
                    fixed, ct = _price(platform, node, price_compute)
                    start = clock + fixed
                    fixed_done_at[nid] = start
                    unit_free[u] = start + ct
                    compute_done_at[nid] = start + ct
                    unit_busy[node.unit] = unit_busy.get(node.unit, 0.0) + fixed + ct
                    dur[nid] = fixed + max(ct, bw.stream_us(node.bytes_streamed, 1) if _dram(node) else 0.0)
                    owner[nid] = _stream_owner(dag, node)
                    progressed = True
            for nid in list(started - activated):
                if fixed_done_at[nid] <= clock + _EPS:
                    activated.add(nid)
                    node = dag[nid]
                    # only DRAM-domain bytes contend on the shared channel; cpu_cache bytes
                    # are priced inside compute_us (m4_cpu) and don't drag the DRAM pool.
                    if _dram(node):
                        active_mem[nid] = float(node.bytes_streamed)
                    progressed = True
            for nid in list(started):
                if nid not in done:
                    n_done = len(done)
                    finish(nid)
                    if len(done) > n_done:        # newly finished -> re-dispatch newly-ready nodes
                        progressed = True
        if not (pending or active_mem or (started - done)):
            break

        # 2) advance to the next event: memory completion / fixed-delay end / compute finish /
        # unit free
        nxt = _INF
        if active_mem:
            rates = rates_bytes_per_us()
            for nid, remaining in active_mem.items():
                r = rates[owner[nid]]
                if r > 0:
                    nxt = min(nxt, remaining / r)
        for nid in started - done:
            if nid not in activated:
                nxt = min(nxt, fixed_done_at[nid] - clock)
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

        # 3) drain memory at the (constant-over-interval) per-stream rates, then finish
        clock += nxt
        if active_mem:
            rates = rates_bytes_per_us()
            mem_active_us += nxt
            for nid in list(active_mem):
                drained = nxt * rates[owner[nid]]
                bytes_drained += min(drained, active_mem[nid])
                active_mem[nid] -= drained
                if active_mem[nid] <= _BYTE_EPS:
                    del active_mem[nid]
        for nid in list(started):
            finish(nid)

    # completion invariant (#56): every node must have finished
    if len(done) != n_total:
        raise RuntimeError(f"M3: incomplete simulation — {n_total - len(done)} of {n_total} "
                           f"nodes never completed")
    if stats is not None:
        _fill_stats(stats, dag, last, unit_busy, dur, bytes_drained, mem_active_us)
    return last
