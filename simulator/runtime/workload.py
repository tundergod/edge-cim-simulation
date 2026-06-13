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


def _phase_templates(m, phase):
    return m.dec if phase == "decode" else m.pre


def build_token_dag(model, phase, L, *, _model_obj=None):
    """Per-forward op DAG for one token at sequence/kv length L (prefill: L=P;
    decode: L=past). Each profile template -> `count` serially-chained nodes.
    Returns a Dag. `_model_obj` lets callers reuse a loaded Model."""
    m = _model_obj or op_profile.Model(model)
    nodes = []
    nid = 0
    prev = None
    for t in _phase_templates(m, phase):
        sig = op_profile._instantiate(t, L)
        cat = op_profile.categorize(sig)
        if cat is None:
            continue
        fl, by = op_profile._flops_bytes(sig)
        row = {"op": sig["op"], "in_shapes": sig["in_shapes"], "out_shape": sig["out_shape"],
               "category": cat, "bytes": by}
        wl = wl_from_row(row, model)
        for _ in range(t["count"]):
            nodes.append(OpNode(id=nid, category=cat, wl=wl, deps=[prev] if prev is not None else [],
                                bytes_streamed=by))
            prev = nid
            nid += 1
    return Dag(nodes)


def category_counts(dag):
    """{category: n_nodes} for a single DAG."""
    out = {}
    for n in dag.nodes:
        out[n.category] = out.get(n.category, 0) + 1
    return out


def _profile_category_counts(m, P, D):
    out = {}
    for r in m.profile(P, D):
        out[(r["phase"], r["category"])] = out.get((r["phase"], r["category"]), 0) + r["count"]
    return out


def _dag_category_counts(model, P, D, m):
    """Per-(phase, category) node counts over one generation: one prefill forward
    at P + one decode forward per past position P..P+D-1 (mirrors profile())."""
    out = {}
    for n in build_token_dag(model, "prefill", P, _model_obj=m).nodes:
        out[("prefill", n.category)] = out.get(("prefill", n.category), 0) + 1
    for past in range(P, P + D):
        for n in build_token_dag(model, "decode", past, _model_obj=m).nodes:
            out[("decode", n.category)] = out.get(("decode", n.category), 0) + 1
    return out


def oracle_check(model, P, D):
    """Fail-loud oracle: the DAG's per-(phase,category) node counts summed over a
    (P,D) generation must equal Model.profile(P,D) counts (no dropped/double-counted
    ops; category set identical = semantic coverage + zero orphans). Returns (ok, detail)."""
    m = op_profile.Model(model)                     # construction self-validates templates vs held-out inventory
    exp = _profile_category_counts(m, P, D)
    got = _dag_category_counts(model, P, D, m)
    ok = exp == got
    detail = {"model": model, "P": P, "D": D, "match": ok,
              "expected": {f"{k[0]}/{k[1]}": v for k, v in sorted(exp.items())},
              "got": {f"{k[0]}/{k[1]}": v for k, v in sorted(got.items())}}
    return ok, detail
