"""M5 — workload generator: per-token op DAG from the Phase-0.2 op_profile (Phase 2.1).

Wraps `tools/trace_export/op_profile.Model` (length-templates fit from the
Phase-0.1 inventory) — does NOT re-run the PyTorch tracer at sim time. For a
given (model, phase, L) it instantiates the profile rows and emits an ordered
per-forward `Dag`: one node per op-instance (a row's `count` is expanded into
that many nodes, so summing the DAG over a generation reproduces
`Model.profile(P,D)` exactly — the oracle check), wired as a serial data-chain.

`bytes_streamed` (the op_profile byte count) is the per-node shared-memory
traffic the M3 contention model meters; `wl` carries the shape the unit engine
prices. Counts come from the profiler (already ×layers) — never hand-rolled.

Run from the repo root (op_profile reads measurements/ relative to CWD).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "tools" / "trace_export"))
import op_profile  # noqa: E402  (Model, _instantiate, _flops_bytes, _key, _sum_by_key)
import fixture_io  # noqa: E402  (load_fixture, PRECISION_CONTRACT, PREFILL_FIX, DECODE_FIX)

from simulator.models.engine import Workload  # noqa: E402
from simulator.runtime.dag import OpNode, Dag  # noqa: E402

GEMM_OPS = op_profile.GEMM_OPS


def _mkn(sig):
    """(M,K,N) for a matmul/bmm sig — mirrors op_profile._flops_bytes shape logic."""
    op, ins = sig["op"], sig["in_shapes"]
    if op == "aten.mm.default":            # [[M,K],[K,N]]
        (M, K), (_, N) = ins[0], ins[1]
    elif op == "aten.addmm.default":       # [[N],[M,K],[K,N]]
        (M, K), (_, N) = ins[1], ins[2]
    else:                                  # bmm [[B,M,K],[B,K,N]]
        B, M, K = ins[0]
        N = ins[1][2]
        M = B * M                          # fold batch into M for the GEMM cost
    return int(M), int(K), int(N)


def _attn_kv_heads(sig):
    """(kv, heads, hd) for an attention bmm sig. QK^T = [[H,Sq,hd],[H,hd,Skv]],
    S·V = [[H,Sq,Skv],[H,Skv,hd]]. kv = the sequence axis being attended over."""
    ins = sig["in_shapes"]
    H = ins[0][0] if ins and ins[0] else 1
    # kv = the larger of the two inner sequence dims (Skv); hd = head dim
    a, b = ins[0], ins[1]
    kv = max(a[2], b[2])
    hd = min(a[2], b[1]) if len(b) == 3 else a[2]
    return int(kv), int(H), int(hd)


def wl_from_row(row, model):
    """Map an op_profile row (op, in_shapes, out_shape, category) -> Workload.
    dtype = the op's natural precision (matmul int8 = CIM scope; non-GEMM fp16);
    the scheduler later decides which unit prices it."""
    op, cat = row["op"], row["category"]
    nbytes = int(row["bytes"])
    if cat == "matmul":                                   # mm/addmm (q/k/v/o, gate/up/down, lm_head)
        M, K, N = _mkn(row)
        return Workload(op="matmul", M=M, K=K, N=N, dtype="int8", nbytes=nbytes)
    if cat == "attention" and op == "aten.bmm.default":   # QK^T / S·V (the real attention GEMM)
        kv, heads, hd = _attn_kv_heads(row)
        return Workload(op="attention", kv=kv, heads=heads, K=hd, dtype="fp16",
                        nbytes=nbytes, extra={"hd": hd})
    if cat == "kv_cache":                                 # DynamicCache append (memory movement)
        return Workload(op="kv_append", nbytes=nbytes, dtype="int8")
    if cat == "embedding":
        return Workload(op="stream", nbytes=nbytes, dtype="fp16")
    # support / elementwise (norm/rope/ffn/softmax/residual, + attention scale & mask) ->
    # priced by category on CPU; memory term from nbytes. aten op kept for provenance.
    kv = int(row["out_shape"][-1]) if (cat == "softmax" and row.get("out_shape")) else 0
    return Workload(op=cat, kv=kv, nbytes=nbytes, dtype="fp16",
                    extra={"model": model, "category": cat, "aten": op})


_STRUCT_CACHE = {}


def _phase_anchors(phase):
    return fixture_io.PREFILL_FIX if phase == "prefill" else fixture_io.DECODE_FIX


def _check_anchor_structure(a, b, model, phase, lo, hi):
    """Fail-loud: the two committed fixture lengths must be positionally identical in STRUCTURE
    (same node count; same per-node op/category/deps/src) so the lo<->hi shape-fit is aligned.
    A silent zip() over mismatched anchors would misalign every downstream node."""
    if len(a) != len(b):
        raise ValueError(f"{model} {phase}: fixture anchor node counts differ "
                         f"({len(a)} @ {lo} vs {len(b)} @ {hi}) — structure not length-independent")
    for i, (na, nb) in enumerate(zip(a, b)):
        ka = (na["op"], na["category"], na["deps"], na.get("src"))
        kb = (nb["op"], nb["category"], nb["deps"], nb.get("src"))
        if ka != kb:
            raise ValueError(f"{model} {phase} node {i}: anchor structure mismatch "
                             f"({na['op']}/{na['category']} @ {lo} vs {nb['op']}/{nb['category']} @ {hi})")


def _load_structure(model, phase):
    """The phase's value-flow STRUCTURE from the committed trace-truth fixture: the
    ordered compute nodes (op/src/category/deps) + a per-node shape-template fit from
    the two committed lengths (lo<->hi), so build_token_dag can instantiate shapes at
    ANY L. Structure is length-independent (verified: the two lengths are positionally
    identical). Cached per (model, phase)."""
    key = (model, phase)
    if key in _STRUCT_CACHE:
        return _STRUCT_CACHE[key]
    fx = fixture_io.load_fixture(model)
    lo, hi = _phase_anchors(phase)
    a, b = fx[phase][str(lo)], fx[phase][str(hi)]
    _check_anchor_structure(a, b, model, phase, lo, hi)
    struct = []
    for i, (na, nb) in enumerate(zip(a, b)):
        tin = [op_profile._fit_shape(sa, sb, lo, hi)
               for sa, sb in zip(na["in_shapes"], nb["in_shapes"])]
        tout = op_profile._fit_shape(na["out_shape"], nb["out_shape"], lo, hi)
        if any(t is None for t in tin) or (na["out_shape"] is not None and tout is None):
            raise ValueError(f"{model} {phase} node {i} ({na['op']}): shape not fittable "
                             f"from lengths {lo}/{hi} — fixture/anchor mismatch")
        struct.append({"op": na["op"], "category": na["category"],
                       "deps": na["deps"], "tin": tin, "tout": tout})
    _STRUCT_CACHE[key] = struct
    return struct


def build_token_dag(model, phase, L, *, _model_obj=None):
    """Per-forward value-flow op DAG for one token at sequence/kv length L (prefill:
    L=P; decode: L=past). STRUCTURE (order + data-dependency edges) comes from the
    trace-truth fixture (real intra-layer DAG, #54); SHAPES/bytes are instantiated at
    L from the per-node template. Each node produces its own value (out_value == id),
    so in_values == deps. Returns a Dag. `_model_obj` is accepted for caller-signature
    compatibility (structure is fixture-cached, not from the Model)."""
    struct = _load_structure(model, phase)
    nodes = []
    for i, s in enumerate(struct):
        in_shapes = [op_profile._inst_shape(t, L) for t in s["tin"]]
        out_shape = op_profile._inst_shape(s["tout"], L)
        _, by = op_profile._flops_bytes({"op": s["op"], "in_shapes": in_shapes, "out_shape": out_shape})
        row = {"op": s["op"], "in_shapes": in_shapes, "out_shape": out_shape,
               "category": s["category"], "bytes": by}
        wl = wl_from_row(row, model)
        deps = list(s["deps"])
        out_elems = op_profile._prod(out_shape) if out_shape else 0
        nodes.append(OpNode(id=i, category=s["category"], wl=wl, deps=deps, bytes_streamed=by,
                            in_values=list(deps), out_value=i, out_elems=out_elems,
                            precision=fixture_io.PRECISION_CONTRACT[s["category"]]))
    return Dag(nodes)


def category_counts(dag):
    """{category: n_nodes} for a single DAG."""
    out = {}
    for n in dag.nodes:
        out[n.category] = out.get(n.category, 0) + 1
    return out


def _profile_counts_bytes(m, P, D):
    counts, total_bytes = {}, 0
    for r in m.profile(P, D):
        counts[(r["phase"], r["category"])] = counts.get((r["phase"], r["category"]), 0) + r["count"]
        total_bytes += r["bytes"] * r["count"]
    return counts, total_bytes


def _dag_counts_bytes(model, P, D, m):
    """Per-(phase, category) node counts + total streamed bytes over one generation:
    one prefill forward at P + one decode forward per past P..P+D-1 (mirrors profile())."""
    counts, total_bytes = {}, 0
    for phase, Ls in (("prefill", [P]), ("decode", list(range(P, P + D)))):
        for L in Ls:
            for n in build_token_dag(model, phase, L, _model_obj=m).nodes:
                counts[(phase, n.category)] = counts.get((phase, n.category), 0) + 1
                total_bytes += n.bytes_streamed
    return counts, total_bytes


def structural_check(model, phase, L):
    """Structural oracle (R1): the built DAG's topology must reproduce the INDEPENDENTLY-
    loaded trace-truth fixture (fixture_io.load_fixture, NOT build_token_dag's cache) AND
    satisfy architecture-config invariants. Since the DAG structure is fixture-derived, the
    fixture comparison is a faithful-reproduction + length-independence check (L is generally
    NOT a committed fixture length); the genuinely-independent content is the config cross-
    check — softmax count == n_layers, residual joins == 2*n_layers, exactly one embedding —
    from op_profile.Model(model).config. Returns (ok, detail)."""
    fx = fixture_io.load_fixture(model)
    ref = _phase_anchors(phase)[-1]                 # a committed length (topology is length-indep)
    fix = fx[phase][str(ref)]
    dag = build_token_dag(model, phase, L)
    reproduces = len(dag.nodes) == len(fix) and all(
        dag.nodes[i].category == fix[i]["category"] and list(dag.nodes[i].deps) == fix[i]["deps"]
        for i in range(len(fix)))
    cfg = op_profile.Model(model).config
    nL = cfg["n_layers"]
    softmax_n = sum(1 for n in dag.nodes if n.category == "softmax")
    res_joins = sum(1 for n in dag.nodes if n.category == "residual" and len(n.deps) >= 2)
    embeds = sum(1 for n in dag.nodes if n.category == "embedding")
    invariants = (softmax_n == nL and res_joins == 2 * nL and embeds == 1)
    ok = reproduces and invariants
    detail = {"model": model, "phase": phase, "L": L, "reproduces_fixture": reproduces,
              "n_layers": nL, "softmax_n": softmax_n, "residual_joins": res_joins,
              "embeddings": embeds, "invariants_ok": invariants}
    return ok, detail


def oracle_check(model, P, D):
    """Fail-loud oracle: the DAG's per-(phase,category) node counts AND total streamed
    bytes summed over a (P,D) generation must equal Model.profile(P,D) (no dropped or
    double-counted ops; identical category set = semantic coverage + zero orphans; bytes
    match guards wl_from_row). Returns (ok, detail)."""
    m = op_profile.Model(model)                     # construction self-validates templates vs held-out inventory
    exp_c, exp_b = _profile_counts_bytes(m, P, D)
    got_c, got_b = _dag_counts_bytes(model, P, D, m)
    ok = (exp_c == got_c) and (exp_b == got_b)
    detail = {"model": model, "P": P, "D": D, "counts_match": exp_c == got_c,
              "bytes_match": exp_b == got_b, "profile_bytes": exp_b, "dag_bytes": got_b,
              "expected": {f"{k[0]}/{k[1]}": v for k, v in sorted(exp_c.items())},
              "got": {f"{k[0]}/{k[1]}": v for k, v in sorted(got_c.items())}}
    return ok, detail
