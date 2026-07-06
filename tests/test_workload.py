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


def test_attn_head_dim_correct_for_both_bmms():
    # _attn_kv_heads must derive hd == config head_dim for BOTH attention bmms:
    # QK^T ([H,Sq,hd]·[H,hd,Skv]) and S·V ([H,Sq,Skv]·[H,Skv,hd]). A naive
    # min/contracted-axis pick returns Skv (=kv) for S·V instead of head dim.
    import op_profile
    import fixture_io
    from simulator.runtime.workload import _attn_kv_heads
    for m in MODELS:
        cfg = op_profile.Model(m).config
        hd_cfg = cfg.get("head_dim") or cfg["hidden_size"] // cfg["n_heads"]
        fx = fixture_io.load_fixture(m)
        for phase in ("prefill", "decode"):
            for L, nodes in fx[phase].items():
                attn_bmms = [n for n in nodes
                             if n["category"] == "attention" and n["op"] == "aten.bmm.default"]
                assert attn_bmms, f"{m} {phase}/{L}: no attention bmm in fixture"
                # bmms appear in QK^T, S·V order per attention block
                for j, n in enumerate(attn_bmms):
                    role = "qk" if j % 2 == 0 else "sv"
                    _, _, hd = _attn_kv_heads(n, role)
                    assert hd == hd_cfg, (f"{m} {phase}/{L} {n['in_shapes']} role={role}: "
                                          f"hd={hd} != config head_dim {hd_cfg}")


def test_attn_kv_hd_correct_for_short_context():
    # Regression for the magnitude-based kv/hd swap: when the kv/sequence axis is
    # SMALLER than head_dim (Skv < hd, i.e. prefill_len < head_dim-1), the old
    # max()/inequality heuristic returned kv=head_dim, hd=Skv. Build a decode DAG at
    # L < head_dim-1 and assert each attention bmm reports kv == L+1 (past+current)
    # and extra['hd'] == config head_dim. Fixture lengths (256/512/1024) all have
    # kv>>hd so they cannot catch this.
    import op_profile
    for m in MODELS:
        cfg = op_profile.Model(m).config
        hd_cfg = cfg.get("head_dim") or cfg["hidden_size"] // cfg["n_heads"]
        L = hd_cfg - 5                      # short context: kv (=L+1) < hd
        assert L > 0
        dag = build_token_dag(m, "decode", L)
        bmms = [n for n in dag.nodes
                if n.category == "attention" and "hd" in n.wl.extra]
        assert bmms, f"{m}: no attention bmm nodes at L={L}"
        for n in bmms:
            assert n.wl.kv == L + 1, f"{m} decode L={L}: kv={n.wl.kv} != L+1={L+1}"
            assert n.wl.extra["hd"] == hd_cfg, (
                f"{m} decode L={L}: hd={n.wl.extra['hd']} != config head_dim {hd_cfg}")


def test_odd_attention_bmm_count_fails_loud():
    # the pending_qk parity guard (build_token_dag): a QK^T bmm with no S·V partner means the
    # by-role kv/hd tagging (qk/sv) assumed paired bmms and silently relied on an even count.
    # Prime _STRUCT_CACHE with a minimal HERMETIC one-node struct (a single attention
    # aten.bmm.default, no partner) so _load_structure returns it without touching real
    # fixtures — the struct shape matches what _load_structure normally produces: a list of
    # {op, category, deps, tin, tout} where tin/tout are op_profile shape "templates" (a plain
    # int list here, i.e. length-independent, since L doesn't matter for this guard).
    from simulator.runtime import workload
    model, phase = "fake-odd-attn-model", "decode"
    key = (model, phase)
    assert key not in workload._STRUCT_CACHE, "test model name collided with a cached struct"
    struct = [{"op": "aten.bmm.default", "category": "attention", "deps": [],
              "tin": [[1, 1, 64], [1, 64, 128]], "tout": [1, 1, 128]}]
    workload._STRUCT_CACHE[key] = struct
    try:
        workload.build_token_dag(model, phase, 10)
    except ValueError as e:
        assert "odd attention-bmm count" in str(e)
    else:
        raise AssertionError("odd attention-bmm count (dangling QK^T) not rejected")
    finally:
        del workload._STRUCT_CACHE[key]


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
