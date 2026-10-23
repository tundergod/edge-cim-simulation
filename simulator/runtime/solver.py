"""Per-op partition solver (ADR-0009): HeteroInfer §4.4 generalised to any unit set / dim.

For one op (or attention block) the solver enumerates every feasible strategy and picks
the one minimising the estimated critical path:

    single-unit u:   T = sync_in(u) + sum_chain max(compute, bytes/BW) + [sync_out if u != host]
    split (k parts): T = max( max_i [sync_in(u_i) + T_i],  sum_i bytes_i / BW_agg(k) )
                         + sync_merge + merge_bytes/BW + [sync if merge_unit != host]

which is HeteroInfer's  T_total = min(max(T_GPU^p1, T_NPU^p2) + T_sync + T_copy, T_GPU^all,
T_NPU^all + T_sync + T_copy)  with: T_copy = 0 (UMA), T_sync charged per cross-unit
boundary exactly as the engine charges it (tiling.apply_sync), and the shared-bandwidth
knee (resources.SharedBandwidth) bounding the parts' aggregate streaming. Strategies:

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


def _supports(ctx, unit, chain_nodes):
    sup = getattr(ctx.platform, "supports", None)
    if sup is None:
        raise ValueError("solver: platform has no supports(unit, category) — the pricing contract "
                         "(worker A) is required; use the test FakePlatform/shim until merged")
    return all(sup(unit, n.category) for n in chain_nodes)


def _dram_bytes(node):
    """Bytes the engine meters on the shared DRAM pool (cpu_cache-resident bytes are priced
    inside the CPU model, not here — the same R7 rule the engine applies)."""
    return node.bytes_streamed if tiling._residency(node) == "dram" else 0


def _node_us(ctx, node):
    """Engine-consistent per-node cost: max(compute, dram bytes at the single-stream rate)."""
    c = float(ctx.platform.compute_us(node))
    bw = ctx.platform.bw
    b = _dram_bytes(node)
    m = b / (bw.eff_BW * 1e3) if (b > 0 and bw.eff_BW > 0) else 0.0
    return max(c, m)


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


def _estimate_single(ctx, dag, chain, unit):
    t = ctx.sync_us if _boundary_in(dag, chain, unit) else 0.0
    for cid in chain:
        tmpl = tiling._copy(dag[cid], unit=unit, pricing_group=dag[cid].pricing_group)
        t += _node_us(ctx, tmpl)
    if unit != ctx.host:
        t += ctx.sync_us
    return t


def _estimate_split(ctx, dag, tp):
    bw = ctx.platform.bw
    per_part, total_bytes = [], 0
    for u, members in zip(tp.units, tp.parts):
        t = ctx.sync_us if _boundary_in(dag, tp.chain, u) else 0.0
        for m in members:
            t += _node_us(ctx, m)
            total_bytes += _dram_bytes(m)
        per_part.append(t)
    k = len(tp.units)
    agg = bw.aggregate_GBs(k) * 1e3
    t_agg = total_bytes / agg if agg > 0 else 0.0
    merge_unit = ctx.merge_unit or ctx.host
    t = max(max(per_part), t_agg) + ctx.sync_us
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


def candidates(ctx, dag, nid):
    """Every feasible strategy for node `nid` (or its attention block), unpriced."""
    node = dag[nid]
    chain = tiling._block(dag, nid)
    chain_nodes = [dag[c] for c in chain]
    units = [u for u in ctx.units if _supports(ctx, u, chain_nodes)]
    out = [Candidate(f"{u}_only", None, [(u, 1.0)]) for u in units]
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
    for c in cands:
        if c.dim is None:
            c.T_est = _estimate_single(ctx, dag, chain, c.plan[0][0])
        else:
            try:
                tp = tiling.tile_parts(dag, nid, c.dim, c.plan, align=ctx.align, rows=ctx.rows)
            except ValueError:
                continue
            c.T_est = _estimate_split(ctx, dag, tp)
            c.merge_bytes = tp.merge_bytes
        n += 1
        if best is None or c.T_est < best.T_est - 1e-9:
            best = c
    ctx.n_candidates = n
    return best, n
