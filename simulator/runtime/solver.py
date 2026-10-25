"""Per-op partition solver (ADR-0009): HeteroInfer §4.4 generalised to any unit set / dim.

For one op (or attention block) the solver enumerates every feasible strategy and picks
the one minimising the estimated critical path:

    single-unit u:   T = sync_in(u) + sum_chain [fixed + max(compute, bytes/BW)] + [sync_out if u != host]
    split (k parts): T = max( max_u [sync_in(u) + sum_{parts on u} T_part],  sum_i bytes_i / BW_agg(k) )
                         + sync_merge + merge_bytes/BW + [sync if merge_unit != host]

which is HeteroInfer's  T_total = min(max(T_GPU^p1, T_NPU^p2) + T_sync + T_copy, T_GPU^all,
T_NPU^all + T_sync + T_copy)  with: T_copy = 0 (UMA), T_sync charged per cross-unit
boundary exactly as the engine charges it (tiling.apply_sync; a serial pre-op `fixed` delay,
ADR-0009 rev F1), parts on the SAME unit summed (serial on one unit, rev F7), the per-unit
weight bytes the engine will meter (rev F13) and the shared-bandwidth knee
(resources.SharedBandwidth) bounding the parts' aggregate streaming. Strategies:

| strategy            | dim   | parts                                                     |
|---------------------|-------|-----------------------------------------------------------|
| <u>_only            | -     | [(u, 1)]                                                  |
| weight_centric      | N     | ratio grid (multiples of 1/partition_grid) over a unit subset; M standard or 1 |
| hybrid              | N     | same, at a non-standard M > 1 (the NPU pads M internally) |
| activation_centric  | M     | floor(M/s) chunks of standard length s on a static-graph unit + remainder on another unit |
| token_split         | M     | ratio grid on M (tiled_hetero only)                       |
| k_split             | K     | ratio grid on K + partial-sum merge (tiled_hetero only)   |
| head_split/kv_split | heads/kv | ratio grid over an attention block (tiled_hetero only) |
| row_split           | rows  | ratio grid over a support op's rows (prefill only)        |
| capacity_split      | N     | the CIM takes the output columns that FIT its residency budget (N_cim = min(N, floor(budget/K)) aligned down to the CIM core width), the remainder goes to the other units (single unit or ratio grid); decode: the CIM part is RESIDENT (pinned across tokens, no weight stream, charged against a global per-DAG budget allocated by saving density); prefill: per-op cap, weights written once per forward (ADR-0009 cooperation revision C1) |

Complexity per op: O(|dims| * sum_{S subset of units, |S|>=2} C(g-1, |S|-1)) candidates, each
priced with |chain|*|S| platform.compute_us calls — e.g. 3 dims x (3 pairs x 7 + 1 triple x 21)
= 126 grid candidates for 3 units at g = 8. Static ratios: one decision per op per DAG,
no DAG-level list scheduling (plan §3.3 "first version"). Prices come ONLY from
`platform.compute_us` / `platform.supports` (never computed here).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from simulator.runtime import tiling

STD_LENGTHS = (128, 256, 512, 1024)      # HeteroInfer §4.2.2 / Fig.14 standard sequence lengths


@dataclass
class Candidate:
    strategy: str
    dim: str | None
    plan: list                            # [(unit, fraction)]
    T_est: float = float("inf")
    merge_bytes: int = 0
    resident: bool = False                # capacity_split in decode: the CIM part is pinned (part 0 / the whole op)
    cim_elems: int = 0                    # capacity_split: K * N_cim weight elements the CIM part holds


@dataclass
class SolverContext:
    """Everything a decision needs besides the DAG + node. `units` = compute units this op may
    use; `host` = the unit that owns the graph (a non-host single placement pays the
    hand-back sync); `static_units` = units that need static shapes (chunked M); `dims` =
    split axes allowed per category (None = all legal); `align` = {unit: (quantum, min)};
    `rows` = tokens in flight (row splits); `allow_chunking` = HeteroInfer activation-centric
    candidates; `allow_grid` = {dim: bool}-style allow-list via `dims`."""
    platform: object
    units: tuple
    host: str
    sync_us: float
    grid: int = 8
    std_lengths: tuple = STD_LENGTHS
    align: dict = field(default_factory=dict)
    rows: int = 1
    dims: dict | None = None               # category -> tuple of allowed dims (None = tiling.LEGAL_DIMS)
    static_units: tuple = ("npu",)
    merge_unit: str | None = None          # None = host
    allow_chunking: bool = True
    n_candidates: int = 0                  # filled by solve()
    # Round-2 revision (ADR-0009 "Revisions 2026-10-23 round 2"): how a NON-HOST single placement is
    # charged for the hand-back boundary to its consumers.
    #   "always"     the per-op greedy rule (HeteroInfer T_NPU^all + T_sync): every non-host single
    #                placement pays one hand-back sync, whatever its consumers do.
    #   "deferred"   each cross-unit boundary is priced ONCE, at its consumer: a consumer the solver
    #                will still decide pays its own sync_in later (so a chain of same-unit ops forms
    #                one run with one sync-in/out pair); a FIXED consumer (`fixed_units`: mem / sampling
    #                / a pinned support block) charges the hand-back now iff it sits elsewhere; a
    #                block with no consumer keeps the hand-back (the paper's rule).
    #   "neighbours" as "deferred", but an already-decided consumer (`hints` = the previous pass's
    #                unit per original op id; a split -> its merge unit = host) also charges the
    #                hand-back when it sits elsewhere: the clean-up pass that undoes an isolated
    #                off-host move the deferred pass made.
    handback: str = "always"
    fixed_units: dict = field(default_factory=dict)
    hints: dict = field(default_factory=dict)
    # Cooperation revision C1 (ADR-0009, 2026-10-26): the capacity-aware weight split. `cim_cap_elems` =
    # the CIM residency budget in weight elements (None = no capacity candidates); `cim_align` = the
    # column quantum of the CIM part (the core width); `cim_resident` (decode) = the CIM part is pinned
    # across tokens and consumes the GLOBAL per-DAG budget: pass 1 (`cim_alloc` None) lets every op
    # request up to the cap and records its saving in `cim_requests`, pass 2 (`cim_alloc` = {nid: N_cim}
    # from scheduler._allocate_cim_budget) offers each op exactly its allocation (absent -> no
    # capacity candidate). Prefill: per-op cap, never resident (weights written once per forward).
    cim_cap_elems: float | None = None
    cim_align: int = 1
    cim_resident: bool = False
    cim_alloc: dict | None = None
    cim_requests: dict = field(default_factory=dict)


def _supports(ctx, unit, chain_nodes):
    sup = getattr(ctx.platform, "supports", None)
    if sup is None:
        raise ValueError("solver: platform has no supports(unit, category) — the pricing contract "
                         "(worker A) is required; use the test FakePlatform/shim until merged")
    return all(sup(unit, n.category) for n in chain_nodes)


def _dram_bytes(ctx, node):
    """Bytes the engine will meter on the shared DRAM pool for `node` on its unit: the INT8 basis
    with the weight part at the unit's `weight_bytes_per_param` (rev F13; a platform without
    the table = 1.0 everywhere); cpu_cache-resident bytes are priced inside the CPU model, not
    here — the same R7 rule the engine applies."""
    if tiling._residency(node) != "dram":
        return 0
    wbpp = getattr(ctx.platform, "weight_bytes_per_param", None) or {}
    return tiling.unit_bytes(node, wbpp.get(node.unit, 1.0))


def _clean(node):
    """A candidate template never carries a previous assignment's sync (`fixed_us`/`sync_us`,
    re-added explicitly by the estimate) or fusion tag (`fused_head`: the solver prices every
    elementwise node as a chain head, pessimistic) — keeps a re-assign idempotent. A scheduler-
    attached per-op overhead (`op_fixed_us`) is kept as the node's fixed term."""
    ex = node.wl.extra
    if ex:
        for k in ("fixed_us", "sync_us", "fused_head", "fused_group"):
            ex.pop(k, None)
        if ex.get("op_fixed_us"):
            ex["fixed_us"] = float(ex["op_fixed_us"])
    return node


def _node_us(ctx, node):
    """Engine-consistent per-node cost: fixed + max(compute, dram bytes at the single-stream rate)
    (`fixed` = the serial pre-op delay the pricing contract returns, rev F1)."""
    node = _clean(node)
    pr = ctx.platform.price(node)
    c, fixed = float(pr["latency_us"]), float(pr.get("fixed_us", 0.0))
    bw = ctx.platform.bw
    b = _dram_bytes(ctx, node)
    m = b / (bw.eff_BW * 1e3) if (b > 0 and bw.eff_BW > 0) else 0.0
    return fixed + max(c, m)


def _producer_unit(dag, d):
    """Unit of the real producer behind dep `d` (looks through an inserted convert node, which
    sits on the consumer's unit: the sync is charged on the convert, so the decision must see
    the boundary either way — keeps re-assign idempotent)."""
    n = dag[d]
    while n.category == "convert" and n.deps:
        n = dag[n.deps[0]]
    return n.unit


def _boundary_in(dag, chain, unit):
    block = set(chain)
    for cid in chain:
        for d in dag[cid].deps:
            if d in block:
                continue
            du = _producer_unit(dag, d)
            if du is None:
                raise ValueError(f"solver: producer {d} of node {cid} is not placed yet "
                                 f"(decide ops in topological order)")
            if du not in (unit, "mem"):
                return True
    return False


def _charges_handback(ctx, dag, chain, unit):
    """Whether a non-host single placement of `chain` on `unit` pays the hand-back sync now
    (see SolverContext.handback)."""
    if ctx.handback == "always":
        return True
    if ctx.handback not in ("deferred", "neighbours"):
        raise ValueError(f"solver: unknown handback mode {ctx.handback!r} (always | deferred | neighbours)")
    block = set(chain)
    consumers = [s for c in chain for s in dag.successors(c) if s not in block]
    if not consumers:
        return True                                   # terminal block: T_u^all + T_sync (paper rule)
    for s in consumers:
        hu = ctx.fixed_units.get(s)
        if hu is None and ctx.handback == "neighbours":
            hu = ctx.hints.get(s)
        if hu is None:
            continue                                  # decided later: the boundary is priced at the consumer
        if hu not in (unit, "mem"):
            return True
    return False


def _estimate_single(ctx, dag, chain, unit, resident=False):
    """`resident` = the whole op is CIM-resident (a capacity_split that fits entirely, decode):
    the template is tagged so its weight bytes drop out of the pool term; a regular template has
    the tag removed (never inherits a stale one)."""
    t = ctx.sync_us if _boundary_in(dag, chain, unit) else 0.0
    for cid in chain:
        tmpl = tiling._copy(dag[cid], unit=unit, pricing_group=dag[cid].pricing_group)
        if resident:
            tmpl.wl.extra["cim_resident"] = True
        else:
            tmpl.wl.extra.pop("cim_resident", None)
        t += _node_us(ctx, tmpl)
    if unit != ctx.host and _charges_handback(ctx, dag, chain, unit):
        t += ctx.sync_us
    return t


def _estimate_split(ctx, dag, tp):
    """Split estimate: parts on the SAME unit run serially (summed per unit, rev F7 — an
    activation-centric plan puts several standard chunks on the NPU), units run concurrently
    (max over units), the aggregate streaming is bounded by the knee at k = #distinct units."""
    bw = ctx.platform.bw
    per_unit, total_bytes = {}, 0
    for u, members in zip(tp.units, tp.parts):
        if u not in per_unit:
            per_unit[u] = ctx.sync_us if _boundary_in(dag, tp.chain, u) else 0.0
        for m in members:
            per_unit[u] += _node_us(ctx, m)
            total_bytes += _dram_bytes(ctx, m)
    k = len(per_unit)
    agg = bw.aggregate_GBs(k) * 1e3
    t_agg = total_bytes / agg if agg > 0 else 0.0
    merge_unit = ctx.merge_unit or ctx.host
    t = max(max(per_unit.values()), t_agg) + ctx.sync_us
    if tp.merge_bytes > 0 and bw.eff_BW > 0:
        t += tp.merge_bytes / (bw.eff_BW * 1e3)
    if merge_unit != ctx.host:
        t += ctx.sync_us
    return t


def _compositions(total, k):
    """All k-tuples of positive ints summing to `total` (ratio grid points)."""
    for cuts in combinations(range(1, total), k - 1):
        prev, out = 0, []
        for c in cuts:
            out.append(c - prev)
            prev = c
        out.append(total - prev)
        yield tuple(out)


def _grid_plans(ctx, units):
    """Ratio-grid plans over every subset (size >= 2) of `units`; the host unit (if present)
    goes LAST so it absorbs the alignment remainder."""
    g = ctx.grid
    for r in range(2, len(units) + 1):
        for subset in combinations(units, r):
            ordered = sorted(subset, key=lambda u: (u == ctx.host, u))
            for comp in _compositions(g, r):
                yield [(u, c / g) for u, c in zip(ordered, comp)]


def _label(dim, node, ctx):
    if dim == "N":
        M = node.wl.M
        return "weight_centric" if (M == 1 or M in ctx.std_lengths) else "hybrid"
    return {"M": "token_split", "K": "k_split", "heads": "head_split",
            "kv": "kv_split", "rows": "row_split"}[dim]


def cim_cols(ctx, nid, K, N):
    """Output columns the CIM part of a capacity_split takes for a K x N matmul: the allocation
    (pass 2) or min(N, floor(cap / K)) (pass 1 / prefill), aligned DOWN to `cim_align` unless the
    whole op fits. 0 = no capacity candidate."""
    if ctx.cim_cap_elems is None or K <= 0 or N <= 0:
        return 0
    if ctx.cim_alloc is not None:
        n = int(ctx.cim_alloc.get(nid, 0))
    else:
        n = int(ctx.cim_cap_elems // K)
    if n >= N:
        return N
    q = max(1, int(ctx.cim_align))
    return (n // q) * q


def _capacity_candidates(ctx, dag, nid, units):
    """The capacity_split candidates of a matmul (revision C1): the CIM holds N_cim columns, the
    remainder goes to ONE other unit or to a ratio-grid plan over the other units (host last).
    Decode (`cim_resident`): an op that fits entirely is offered as a RESIDENT single placement
    (plain cim_only streams its weights every token; the resident one does not); prefill: an op
    that fits entirely is already the cim_only candidate (nothing to add)."""
    node = dag[nid]
    if node.category != "matmul" or "cim" not in units:
        return []
    K, N = node.wl.K, node.wl.N
    n = cim_cols(ctx, nid, K, N)
    if n <= 0:
        return []
    if n >= N:
        if ctx.cim_resident:
            return [Candidate("capacity_split", None, [("cim", 1.0)], resident=True, cim_elems=K * N)]
        return []
    others = [u for u in units if u != "cim"]
    if not others:
        return []
    f = n / N
    out = [Candidate("capacity_split", "N", [("cim", f), (u, 1.0 - f)], resident=ctx.cim_resident,
                     cim_elems=K * n) for u in others]
    if len(others) >= 2:
        for plan in _grid_plans(ctx, others):
            out.append(Candidate("capacity_split", "N", [("cim", f)] + [(u, c * (1.0 - f)) for u, c in plan],
                                 resident=ctx.cim_resident, cim_elems=K * n))
    return out


def candidates(ctx, dag, nid):
    """Every feasible strategy for node `nid` (or its attention block), unpriced."""
    node = dag[nid]
    chain = tiling._block(dag, nid)
    chain_nodes = [dag[c] for c in chain]
    units = [u for u in ctx.units if _supports(ctx, u, chain_nodes)]
    out = [Candidate(f"{u}_only", None, [(u, 1.0)]) for u in units]
    out += _capacity_candidates(ctx, dag, nid, units)
    if len(units) < 2:
        return out, units
    legal = tiling.LEGAL_DIMS.get(node.category, ())
    if node.category == "attention" and "hd" not in node.wl.extra:
        legal = ("rows",)
    allowed = legal if ctx.dims is None else tuple(d for d in legal if d in ctx.dims.get(node.category, ()))
    M = node.wl.M
    if ctx.allow_chunking and node.category == "matmul" and M > 1 and M not in ctx.std_lengths:
        # HeteroInfer activation-centric: floor(M/s) static chunks on a static-graph unit, the
        # remainder on another unit (multi-tensor when floor(M/s) > 1). Independent of the grid.
        for a in [u for u in units if u in ctx.static_units]:
            for b in [u for u in units if u != a]:
                for s in ctx.std_lengths:
                    if s < M and M % s:
                        out.append(Candidate("activation_centric", "M",
                                             [(a, s / M)] * (M // s) + [(b, (M % s) / M)]))
    for dim in allowed:
        if (dim == "rows" and ctx.rows < 2) or (dim == "M" and M < 2):
            continue
        for plan in _grid_plans(ctx, units):
            out.append(Candidate(_label(dim, node, ctx), dim, plan))
    return out, units


def solve(ctx, dag, nid):
    """Price every candidate for `nid` and return (best Candidate, n_candidates). Infeasible
    plans (alignment/minimum violations) are skipped. Deterministic: ties -> first found
    (singles first, host-last ordering inside the grid)."""
    cands, units = candidates(ctx, dag, nid)
    if not cands:
        raise ValueError(f"solver: no unit in {ctx.units} supports node {nid} "
                         f"({dag[nid].category}); cannot place it")
    chain = tiling._block(dag, nid)
    best, n = None, 0
    best_reg, best_cap = None, None            # pass-1 bookkeeping of the capacity split (decode)
    for c in cands:
        if c.dim is None:
            c.T_est = _estimate_single(ctx, dag, chain, c.plan[0][0], resident=c.resident)
        else:
            try:
                tp = tiling.tile_parts(dag, nid, c.dim, c.plan, align=ctx.align, rows=ctx.rows,
                                       resident=((0,) if c.resident else ()), chain=chain)
            except ValueError:
                continue
            c.T_est = _estimate_split(ctx, dag, tp)
            c.merge_bytes = tp.merge_bytes
        n += 1
        if best is None or c.T_est < best.T_est - 1e-9:
            best = c
        if c.strategy == "capacity_split":
            if best_cap is None or c.T_est < best_cap.T_est - 1e-9:
                best_cap = c
        elif best_reg is None or c.T_est < best_reg.T_est - 1e-9:
            best_reg = c
    if ctx.cim_resident and ctx.cim_alloc is None and best_cap is not None and best_reg is not None:
        # pass 1: what pinning this op's CIM part would save per pinned weight element
        node = dag[nid]
        n_cim = best_cap.cim_elems // node.wl.K
        ctx.cim_requests[nid] = {"saving_us": best_reg.T_est - best_cap.T_est, "K": node.wl.K,
                                 "N_cim": n_cim, "elems": best_cap.cim_elems,
                                 "T_regular_us": best_reg.T_est, "T_capacity_us": best_cap.T_est}
    ctx.n_candidates = n
    return best, n
