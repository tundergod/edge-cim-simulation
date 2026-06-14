"""Phase 2.1 — M5 oracle check (simulator/runtime/workload.py).

The per-token DAG, summed over a (P,D) generation, must reproduce
Model.profile(P,D) per-(phase,category) counts for every model (no dropped or
double-counted ops; identical category set = semantic coverage + zero orphans).
Model() construction also self-validates the length-templates vs held-out
inventory. Run from the repo root (op_profile reads measurements/ via CWD).

    .venv/bin/pytest tests/test_workload.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.runtime.workload import oracle_check, build_token_dag, wl_from_row  # noqa: E402
from simulator.runtime.dag import wl_is_sane  # noqa: E402

MODELS = ["llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b", "qwen2.5-7b"]


def test_oracle_counts_and_bytes_match_all_models():
    # cross-artifact check: the fixture-derived value-flow DAG, summed over a (P,D)
    # generation, must reproduce op_profile per-(phase,category) COUNTS and total
    # streamed BYTES for every model (the fixture and op_profile are independent
    # derivations of the same trace — agreement is the consistency gate).
    for m in MODELS:
        ok, detail = oracle_check(m, 128, 4)
        assert ok, f"{m}: DAG counts/bytes != profile: {detail}"
        assert detail["counts_match"] and detail["bytes_match"]


def test_decode_dag_value_flow_sane():
    dag = build_token_dag("llama-3.1-8b", "decode", 256)
    assert len(dag) > 0 and dag.is_acyclic()
    assert all(wl_is_sane(n.wl) for n in dag.nodes)
    assert all(n.bytes_streamed >= 0 for n in dag.nodes)
    # value-flow DAG removes the 2.1 serial-chain crutch: there are real roots
    # (embedding + rope-freq) and the graph is NOT a single chain.
    assert len(dag.roots()) >= 1


def test_value_flow_has_fanout_and_joins():
    # Q/K/V fanout (a node feeding >=3 successors) + residual joins (>=2 deps) —
    # impossible under the old serial chain.
    from collections import Counter
    dag = build_token_dag("llama-3.2-1b", "decode", 512)
    joins = [n for n in dag.nodes if len(n.deps) >= 2]
    assert joins, "no join nodes — still a serial chain?"
    fanout = Counter(d for n in dag.nodes for d in n.deps)
    assert max(fanout.values()) >= 3, "no >=3 fanout (Q/K/V projections)"


def test_attention_chain_crosses_softmax_category():
    # QK^T(attn) -> scale(attn) -> mask(attn) -> softmax(SOFTMAX) -> S·V(attn):
    # softmax is its own category (S1-2), consuming an attention node and feeding one.
    dag = build_token_dag("llama-3.2-1b", "decode", 512)
    softmaxes = [n for n in dag.nodes if n.category == "softmax"]
    assert softmaxes, "no softmax nodes"
    for s in softmaxes:
        assert any(dag[d].category == "attention" for d in s.deps), "softmax not fed by attention"
        succ_cats = {dag[x].category for x in dag.successors(s.id)}
        assert "attention" in succ_cats, "softmax does not feed S·V (attention)"


def test_value_flow_no_dangling_value():
    dag = build_token_dag("llama-3.1-8b", "decode", 200)
    ids = {n.id for n in dag.nodes}
    for n in dag.nodes:
        assert n.out_value == n.id                 # each node produces its own value
        assert n.in_values == n.deps               # node-granular value-flow
        for v in n.in_values:
            assert v in ids and v < n.id           # references an earlier produced value


def test_precision_from_contract():
    import fixture_io
    dag = build_token_dag("llama-3.2-1b", "decode", 512)
    for n in dag.nodes:
        assert n.precision == fixture_io.PRECISION_CONTRACT[n.category]


def test_anchor_structure_mismatch_fails_loud():
    # _load_structure zips the two committed fixture lengths positionally; a length or
    # per-node (op/category/deps/src) mismatch must fail loud, not silently misalign.
    from simulator.runtime.workload import _check_anchor_structure
    good = [{"op": "aten.mm.default", "category": "matmul", "deps": [], "src": "S"}]
    _check_anchor_structure(good, good, "m", "decode", 512, 1024)        # identical -> ok
    try:
        _check_anchor_structure(good, good + good, "m", "decode", 512, 1024)   # length mismatch
    except ValueError:
        pass
    else:
        raise AssertionError("anchor length mismatch not caught")
    bad = [{"op": "aten.add.Tensor", "category": "norm", "deps": [0], "src": "S"}]
    try:
        _check_anchor_structure(good, bad, "m", "decode", 512, 1024)      # per-node mismatch
    except ValueError:
        return
    raise AssertionError("anchor per-node structure mismatch not caught")


def test_categories_are_known():
    from simulator.runtime.dag import CATEGORIES
    dag = build_token_dag("llama-3.2-1b", "decode", 128)
    assert {n.category for n in dag.nodes} <= set(CATEGORIES)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"\n{len(fns)} workload tests passed.")
