"""Op tiling — split one op (or one attention block) across units (ADR-0009).

`split_node(dag, nid, dim, plan)` replaces the op `nid` by k sub-nodes (same category,
shrunk Workload, each on its own unit, deps = the original's deps) plus ONE `merge` node
(category "merge") that the original's successors now depend on. `plan` is
`[(unit, fraction), ...]`; `dim` picks the split axis:

| dim     | category          | sub-shape                      | merge bytes                         |
|---------|-------------------|--------------------------------|-------------------------------------|
| N       | matmul            | N_i (output features)          | 0 (concat; HeteroInfer weight-centric) |
| M       | matmul            | M_i (tokens; activation-centric)| 0 (concat)                          |
| K       | matmul            | K_i (contraction)              | k*M*N*4 + M*N*bpe (partial-sum add) |
| heads   | attention block   | heads_i (GQA-group aligned)    | 0 (concat)                          |
| kv      | attention block   | kv_i (flash-decoding chunks)   | k*H*M*(hd+2)*4 + H*M*hd*bpe (LSE)   |
| rows    | norm/rope/ffn/softmax/residual (and non-bmm attention) | rows_i | 0 |

An ATTENTION split operates on the whole block = QK^T rep + the nodes strictly between it
and its S.V partner (scale/mask/softmax) + S.V; every part gets its own copy of the chain
and a FRESH pricing_group (the composite is priced once per sub-pair, R2). FLOPs are
conserved exactly (integer partition of the split axis; the LAST part absorbs the
alignment remainder, so an NPU quantum pads only inside the pricing model, never in the
DAG). Bytes: the non-split operand is REPLICATED per part (N-split re-reads the
activation, M-split re-reads the weight, K-split writes k partial outputs); the merge
node carries the explicit reduction bytes. Sub-node ids are allocated deterministically
(max id + 1, parts first, merge last); a plan with a single part is a no-op (unit set,
no merge node).

Sync model: every cross-unit boundary (producer unit != consumer unit, 'mem' excluded)
costs one `sync_us` — carried by the merge node (`wl.extra["fixed_us"]`) for a gather, or
by the consumer node's `wl.extra["fixed_us"]` otherwise (`apply_sync`). Copy cost is 0
(UMA zero-copy, HeteroInfer §4.3). SYNC_US — the ONE sync table (ADR-0009 rev F8; the SoC
profiles carry no sync block, Platform.sync_us mirrors this): fast = HeteroInfer
predictable-wait + flag polling ("a few microseconds" decode; + "tens of microseconds" GPU
submission delay in prefill) -> 10 / 30 us [LIT, HeteroInfer §4.3]; clfinish = the 400 us
clFinish baseline [LIT, §3.1]; none = 0 (ideal). All SIMULATED: no heterogeneous-sync silicon.
The engine and the solver charge `fixed_us` as a SERIAL pre-op delay (rev F1).

Per-unit weight bytes (ADR-0008 rev F13): a matmul node's `bytes_streamed` is op_profile's INT8
count until a scheduler's `_finish` rescales the weight part to the assigned unit's
`weight_bytes_per_param`; the INT8 basis is remembered in `wl.extra["bytes_int8"]` so the
rescale is idempotent and sub-node byte shares are always derived from the INT8 basis.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from simulator.models.engine import Workload
from simulator.runtime.dag import OpNode, Dag

DIMS = ("N", "M", "K", "heads", "kv", "rows")
_SUPPORT = ("norm", "rope", "ffn", "softmax", "residual")
LEGAL_DIMS = {"matmul": ("N", "M", "K"), "attention": ("heads", "kv"),
              **{c: ("rows",) for c in _SUPPORT}}
SYNC_US = {"fast": {"decode": 10.0, "prefill": 30.0},
           "clfinish": {"decode": 400.0, "prefill": 400.0},
           "none": {"decode": 0.0, "prefill": 0.0}}
SYNC_MODES = tuple(SYNC_US)
ACC_BYTES = 4          # fp32 partial sums / log-sum-exp statistics exchanged at a merge
DEFAULT_ALIGN = {"npu": (32, 32)}   # unit -> (quantum, minimum) on the split axis; NPU-1 systolic tile = 32


def resolve_sync_us(mode, phase, override=None):
    """Per-boundary sync cost (us) for `mode` in SYNC_MODES and `phase` in {decode, prefill};
    `override` (a number) wins when given."""
    if override is not None:
        if override < 0:
            raise ValueError(f"sync_us_override must be >= 0, got {override!r}")
        return float(override)
    if mode not in SYNC_US:
        raise ValueError(f"unknown sync_mode {mode!r} (known: {SYNC_MODES})")
    if phase not in ("decode", "prefill"):
        raise ValueError(f"unknown phase {phase!r}")
    return SYNC_US[mode][phase]


def is_attention_rep(node):
    """True for the QK^T bmm that heads an attention block (the pricing-group rep)."""
    return (node.category == "attention" and "hd" in node.wl.extra
            and node.pricing_group is not None and node.pricing_group == node.id)


def partition(total, fracs, quanta=None, minima=None):
    """Integer partition of `total` by `fracs` (sum 1). Part i (except the last) is rounded
    to a multiple of quanta[i]; the LAST part absorbs the remainder. Every part must be
    >= max(1, minima[i]) or the plan is infeasible (ValueError)."""
    k = len(fracs)
    if k == 0:
        raise ValueError("partition: empty plan")
    if any(f <= 0 for f in fracs) or abs(sum(fracs) - 1.0) > 1e-9:
        raise ValueError(f"partition: fractions must be > 0 and sum to 1, got {fracs}")
    quanta = quanta or [1] * k
    minima = minima or [1] * k
    if k == 1:
        sizes = [int(total)]
    else:
        sizes = []
        for f, q in zip(fracs[:-1], quanta[:-1]):
            sizes.append(int(round(f * total / q)) * q)
        sizes.append(int(total) - sum(sizes))
    for s, m in zip(sizes, minima):
        if s < max(1, m):
            raise ValueError(f"partition: infeasible plan {fracs} on {total} "
                             f"(quanta {quanta}, minima {minima}) -> sizes {sizes}")
    return sizes


def _int_shares(total, fracs):
    """Exact integer shares of `total` by `fracs` (zeros allowed; remainder to the last)."""
    shares = [int(round(f * total)) for f in fracs[:-1]]
    shares.append(int(total) - sum(shares))
    return shares


def base_bytes(node):
    """The node's INT8 (op_profile) byte count: `wl.extra['bytes_int8']` when a scheduler has
    rescaled `bytes_streamed` to a unit's weight bytes (rev F13), else bytes_streamed itself."""
    if node.wl.extra and "bytes_int8" in node.wl.extra:
        return int(node.wl.extra["bytes_int8"])
    return node.bytes_streamed


def unit_bytes(node, wbpp):
    """`bytes_streamed` of a matmul node on a unit streaming `wbpp` bytes per weight parameter:
    the INT8 basis with the K*N weight part rescaled (activation bytes unchanged). Non-matmul
    nodes: the INT8 basis. A CIM-RESIDENT part (`wl.extra['cim_resident']`: the decode
    capacity split, ADR-0009 cooperation revision C1) streams NO weight bytes per token — the
    weights are pinned in the array across tokens, only the activation bytes hit the pool."""
    b = base_bytes(node)
    if node.category != "matmul":
        return b
    if node.wl.extra and (node.wl.extra.get("cim_resident") or node.wl.extra.get("chunk_weights_shared")):
        wbpp = 0.0            # C1 pinned CIM part / C2 chunk c > 0: the weights were streamed by chunk 0
    return int(round(b + node.wl.K * node.wl.N * (float(wbpp) - 1.0)))


def _bpe(node, elems):
    """Bytes per element implied by the node's INT8 byte count (op_profile counts 1 B/element)."""
    return base_bytes(node) / elems if elems > 0 else 0.0


def _copy(node, *, wl=None, bytes_streamed=None, out_elems=None, unit=None, tile=None,
          pricing_group=None):
    """A fresh OpNode copy (fresh Workload + extra dict) with the given overrides; the id
    and deps are kept (remapped later by split_node)."""
    w = wl if wl is not None else replace(node.wl, extra=dict(node.wl.extra))
    return replace(node, wl=w, deps=list(node.deps), in_values=list(node.in_values),
                   bytes_streamed=node.bytes_streamed if bytes_streamed is None else int(bytes_streamed),
                   out_elems=node.out_elems if out_elems is None else int(out_elems),
                   unit=node.unit if unit is None else unit, tile=tile,
                   pricing_group=pricing_group, mem_domain=None)


def _attn_M(node):
    """Query length (rows) of an attention bmm, derived from `out_elems` (QK^T out = H*M*kv,
    S.V out = H*M*hd) so a kv-split chunk's partial output still yields the block's M. NB
    `wl_from_row` DOES set wl.M = Sq for bmms (decode 1, prefill P) and the unit models price
    wl.M; this helper is the out_elems-consistent value tiling uses for byte/FLOP bookkeeping."""
    wl = node.wl
    hd = wl.extra["hd"]
    role = "qk" if node.pricing_group == node.id else "sv"     # holds for originals and part copies
    per_head = (wl.kv if role == "qk" else hd)
    return max(1, node.out_elems // max(1, wl.heads * per_head))


def _block(dag, nid):
    """The ids to split together, in topological order: [nid] for a plain op; for an
    attention QK^T rep, [qk, between..., sv] (fail-loud if the block is not closed)."""
    n = dag[nid]
    if n.category == "attention" and "hd" in n.wl.extra:
        if n.pricing_group != n.id:
            raise ValueError(f"split_node: attention node {nid} is not the QK^T rep of its "
                             f"pricing group ({n.pricing_group}); split the rep")
        partners = [m.id for m in dag.nodes if m.pricing_group == nid and m.id != nid]
        if len(partners) != 1:
            raise ValueError(f"split_node: attention rep {nid} has {len(partners)} S.V partners (expected 1)")
        sv = partners[0]
        fwd = _reach(dag, nid, lambda i: dag.successors(i))
        bwd = _reach(dag, sv, lambda i: dag[i].deps)
        between = (fwd & bwd) - {nid, sv}
        order = [i for i in _topo(dag, between | {nid, sv})]
        for i in order[:-1]:
            if any(s not in order for s in dag.successors(i)):
                raise ValueError(f"split_node: attention block {nid}..{sv} is not closed "
                                 f"(node {i} feeds a node outside the block)")
        return order
    return [nid]


def _reach(dag, start, step):
    seen, stack = set(), [start]
    while stack:
        i = stack.pop()
        for j in step(i):
            if j not in seen:
                seen.add(j)
                stack.append(j)
    return seen


def _topo(dag, ids):
    """Topological order of the sub-DAG induced by `ids` (ties by id: deterministic)."""
    ids = set(ids)
    indeg = {i: sum(1 for d in dag[i].deps if d in ids) for i in ids}
    ready = sorted(i for i, d in indeg.items() if d == 0)
    out = []
    while ready:
        i = ready.pop(0)
        out.append(i)
        for s in dag.successors(i):
            if s in ids:
                indeg[s] -= 1
                if indeg[s] == 0:
                    ready.append(s)
                    ready.sort()
    if len(out) != len(ids):
        raise ValueError("split_node: cycle inside the block")
    return out


def _scale_proportional(node, frac, bshare, *, kv=None):
    """A proportional (row-wise / per-chunk) shrink of a node with no structural shape:
    nbytes/bytes/out_elems scale by `frac` (bytes via the exact integer share `bshare`)."""
    wl = replace(node.wl, nbytes=int(round(node.wl.nbytes * frac)), extra=dict(node.wl.extra),
                 kv=(node.wl.kv if kv is None else kv))
    return _copy(node, wl=wl, bytes_streamed=bshare, out_elems=int(round(node.out_elems * frac)))


def _matmul_part(node, dim, size):
    M, K, N = node.wl.M, node.wl.K, node.wl.N
    bpe = _bpe(node, M * K + K * N + M * N)
    if dim == "N":
        M_, K_, N_ = M, K, size
        out = M * size
    elif dim == "M":
        M_, K_, N_ = size, K, N
        out = size * N
    elif dim == "K":
        M_, K_, N_ = M, size, N
        out = M * N                      # a full partial output per part
    else:
        raise ValueError(f"matmul cannot be split on {dim!r}")
    wl = replace(node.wl, M=M_, K=K_, N=N_, extra=dict(node.wl.extra))
    b = round((M_ * K_ + K_ * N_ + M_ * N_) * bpe)
    wl.extra["bytes_int8"] = int(b)             # the part's own INT8 basis (rev F13)
    return _copy(node, wl=wl, bytes_streamed=b, out_elems=out)


def _attn_bmm_part(node, dim, size, frac, bshare):
    wl = node.wl
    H, hd, kv = wl.heads, wl.extra["hd"], wl.kv
    M = _attn_M(node)
    role = "qk" if node.pricing_group == node.id else "sv"
    if dim == "heads":
        w = replace(wl, heads=size, extra=dict(wl.extra))
        return _copy(node, wl=w, bytes_streamed=bshare, out_elems=int(round(node.out_elems * frac)))
    if dim == "kv":
        bpe = _bpe(node, H * (M * hd + hd * kv + M * kv))
        w = replace(wl, kv=size, extra=dict(wl.extra))
        if role == "qk":
            b = H * (M * hd + hd * size + M * size)
            out = H * M * size
        else:
            b = H * (M * size + size * hd + M * hd)
            out = H * M * hd                      # a full partial output per chunk
        return _copy(node, wl=w, bytes_streamed=round(b * bpe), out_elems=out)
    raise ValueError(f"attention cannot be split on {dim!r}")


def _merge_bytes(dag, chain, dim, k):
    last = dag[chain[-1]]
    if dim == "K":
        wl = last.wl
        bpe = _bpe(last, wl.M * wl.K + wl.K * wl.N + wl.M * wl.N)
        return int(round(k * wl.M * wl.N * ACC_BYTES + wl.M * wl.N * bpe))
    if dim == "kv":
        wl = last.wl
        H, hd, M = wl.heads, wl.extra["hd"], _attn_M(last)
        bpe = _bpe(last, H * (M * wl.kv + wl.kv * hd + M * hd))
        return int(round(k * H * M * (hd + 2) * ACC_BYTES + H * M * hd * bpe))
    return 0


@dataclass
class TilePlan:
    """What a split produces, before ids are assigned: `chain` = original ids of the block
    (topological), `parts[i]` = the i-th part's copies of every chain member (original ids
    kept, unit set), `units`, `sizes` on the split axis, `merge_bytes`, `replicated_bytes`
    (= sum(part bytes) - original bytes: the operand re-read per part)."""
    nid: int
    dim: str
    chain: list
    units: list
    sizes: list
    parts: list
    merge_bytes: int
    replicated_bytes: int


def _normalize_plan(plan):
    plan = [(u, float(f)) for u, f in plan if f > 0]
    if not plan:
        raise ValueError("split_node: empty plan")
    units = [u for u, _ in plan]
    fracs = [f for _, f in plan]
    if abs(sum(fracs) - 1.0) > 1e-9:
        raise ValueError(f"split_node: plan fractions must sum to 1, got {fracs}")
    return units, fracs


def tile_parts(dag, nid, dim, plan, *, align=None, rows=None, resident=(), chain=None):
    """Compute the sub-node templates for splitting `nid` on `dim` by `plan` (no DAG rewrite).
    `align` = {unit: (quantum, minimum)} for N/M/K; the keys "heads"/"kv"/"rows" give the
    axis quantum for attention/row splits (heads quantum = the GQA group). `rows` is the row
    count of a row-split op (tokens in flight; 1 in decode -> unsplittable). `resident` = part
    indices whose matmul part is tagged CIM-resident (`wl.extra['cim_resident']`: pinned
    weights, no per-token weight stream; the decode capacity split) — every other part has
    the tag removed, so a template never inherits a stale one."""
    align = {**DEFAULT_ALIGN, **(align or {})}
    node = dag[nid]
    if dim not in LEGAL_DIMS.get(node.category, ()) and not (dim == "rows" and node.category == "attention"
                                                             and "hd" not in node.wl.extra):
        raise ValueError(f"split_node: dim {dim!r} is not legal for category {node.category!r} "
                         f"(legal: {LEGAL_DIMS.get(node.category, ())})")
    units, fracs = _normalize_plan(plan)
    k = len(units)
    chain = _block(dag, nid) if chain is None else list(chain)   # `chain` = the caller's _block (solver: once per op)
    wl = node.wl
    if dim in ("N", "M", "K"):
        total = {"N": wl.N, "M": wl.M, "K": wl.K}[dim]
        quanta = [align.get(u, (1, 1))[0] for u in units]
        minima = [align.get(u, (1, 1))[1] for u in units]
    elif dim in ("heads", "kv"):
        total = wl.heads if dim == "heads" else wl.kv
        q, m = align.get(dim, (1, 1))
        quanta, minima = [q] * k, [m] * k
    else:                                                   # rows
        if rows is None:
            raise ValueError("split_node: a 'rows' split needs rows= (tokens in flight)")
        total = int(rows)
        quanta, minima = [1] * k, [1] * k
    if total <= 0:
        raise ValueError(f"split_node: node {nid} has no extent on {dim!r}")
    sizes = partition(total, fracs, quanta, minima)
    eff = [s / total for s in sizes]                        # realised fractions after alignment
    parts = []
    for i, (u, f, s) in enumerate(zip(units, eff, sizes)):
        members = []
        for cid in chain:
            c = dag[cid]
            bshare = _int_shares(base_bytes(c), eff)[i]
            tile = {"of": nid, "dim": dim, "part": i, "n": k, "frac": f, "src": cid}
            if c.category == "matmul":
                p = _matmul_part(c, dim, s)
                if i in resident:
                    p.wl.extra["cim_resident"] = True
                else:
                    p.wl.extra.pop("cim_resident", None)
            elif c.category == "attention" and "hd" in c.wl.extra:
                p = _attn_bmm_part(c, dim, s, f, bshare)
            else:
                p = _scale_proportional(c, f, bshare, kv=(s if (dim == "kv" and c.wl.kv) else None))
            p.unit = u
            p.tile = tile
            # fresh pricing_group per sub-pair: the part's QK^T copy is the rep (own ORIGINAL id
            # for now; split_node remaps to the new ids), the S.V copy points at it.
            if c.category == "attention" and "hd" in c.wl.extra:
                p.pricing_group = chain[0]
            members.append(p)
        parts.append(members)
    orig_bytes = sum(base_bytes(dag[c]) for c in chain)
    part_bytes = sum(base_bytes(m) for ms in parts for m in ms)
    return TilePlan(nid=nid, dim=dim, chain=chain, units=units, sizes=sizes, parts=parts,
                    merge_bytes=_merge_bytes(dag, chain, dim, k),
                    replicated_bytes=part_bytes - orig_bytes)


def _residency(node):
    from simulator.runtime.scheduler import _residency as r   # deferred: scheduler imports tiling
    return r(node)


def split_node(dag, nid, dim, plan, *, merge_unit="cpu", sync_us=0.0, align=None, rows=None, resident=()):
    """Return a REBUILT Dag with `nid` (or its attention block) split per `plan` on `dim`
    and a merge node (unit `merge_unit`, fixed_us = `sync_us`, bytes = reduction bytes)
    feeding the original's successors. A single-part plan only sets the unit (no merge).
    Ids: parts first (chain order within a part), merge last, all allocated from max id + 1.
    `resident` = part indices tagged CIM-resident (see tile_parts)."""
    units, fracs = _normalize_plan(plan)
    if len(units) == 1:
        for cid in _block(dag, nid):
            n = dag[cid]
            n.unit = units[0]
            n.mem_domain = _residency(n)
        return dag
    tp = tile_parts(dag, nid, dim, plan, align=align, rows=rows, resident=resident)
    block = set(tp.chain)
    last = tp.chain[-1]
    next_id = max(n.id for n in dag.nodes) + 1
    new_parts, part_last = [], []
    for members in tp.parts:
        idmap = {}
        for m in members:
            idmap[m.id] = next_id
            next_id += 1
        for m in members:
            m.id = idmap[m.id]
            m.out_value = m.id
            m.deps = [idmap.get(d, d) for d in m.deps]
            m.in_values = list(m.deps)
            if m.pricing_group is not None:
                m.pricing_group = idmap[m.pricing_group]
            m.mem_domain = _residency(m)
            new_parts.append(m)
        part_last.append(members[-1].id)
    orig_last = dag[last]
    mid = next_id
    merge = OpNode(id=mid, category="merge",
                   wl=Workload(op="merge", nbytes=tp.merge_bytes, dtype=orig_last.wl.dtype,
                               extra={"fixed_us": float(sync_us), "sync_us": float(sync_us),
                                      "dim": dim, "of": nid, "k": len(units)}),
                   deps=list(part_last), unit=merge_unit, bytes_streamed=tp.merge_bytes,
                   in_values=list(part_last), out_value=mid, precision=orig_last.precision,
                   out_elems=orig_last.out_elems,
                   tile={"of": nid, "dim": dim, "role": "merge", "n": len(units)})
    merge.mem_domain = "dram" if tp.merge_bytes > 0 else "none"
    nodes = []
    inserted = False
    for n in dag.nodes:
        if n.id in block:
            if not inserted:
                nodes.extend(new_parts)
                nodes.append(merge)
                inserted = True
            continue
        if last in n.deps:
            n.deps = [mid if d == last else d for d in n.deps]
            n.in_values = list(n.deps)
        nodes.append(n)
    out = Dag(nodes)
    out.plans = dict(dag.plans)
    out.plans.setdefault(nid, {})
    out.plans[nid].update({"dim": dim, "parts": list(zip(tp.units, [s / sum(tp.sizes) for s in tp.sizes])),
                           "sizes": list(tp.sizes), "block": list(tp.chain),
                           "sub_ids": [m.id for m in new_parts], "merge_id": mid,
                           "merge_bytes": tp.merge_bytes, "replicated_bytes": tp.replicated_bytes})
    return out


def split_dag(dag, plans, **kw):
    """Apply `plans` = {nid: (dim, plan)} in ascending nid order (each split returns a new Dag;
    ids never get renumbered, so later splits see earlier merge nodes)."""
    for nid in sorted(plans):
        dim, plan = plans[nid]
        dag = split_node(dag, nid, dim, plan, **kw)
    return dag


def apply_sync(dag, sync_us):
    """Charge one `sync_us` on every consumer with at least one producer on a different
    unit ('mem' producers/consumers excluded; merge nodes already carry their own sync).
    Recomputed from scratch -> idempotent. Returns the per-DAG sync count."""
    count = 0
    for n in dag.nodes:
        ex = n.wl.extra
        if n.category == "merge":
            ex["fixed_us"] = ex.get("op_fixed_us", 0.0) + ex["sync_us"]
            count += 1 if ex["sync_us"] > 0 else 0
            continue
        boundary = (n.unit != "mem" and any(dag[d].unit not in (n.unit, "mem") for d in n.deps))
        s = float(sync_us) if boundary else 0.0
        fixed = ex.get("op_fixed_us", 0.0) + s
        if s > 0:
            ex["sync_us"] = s
            count += 1
        else:
            ex.pop("sync_us", None)
        if fixed > 0:
            ex["fixed_us"] = fixed
        else:
            ex.pop("fixed_us", None)
    return count


def flops(node):
    """2*M*K*N for a matmul; 2*H*M*hd*kv per attention bmm; 0 otherwise."""
    wl = node.wl
    if node.category == "matmul":
        return 2 * wl.M * wl.K * wl.N
    if node.category == "attention" and "hd" in wl.extra:
        return 2 * wl.heads * _attn_M(node) * wl.extra["hd"] * wl.kv
    return 0
