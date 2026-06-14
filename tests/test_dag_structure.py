"""Phase 2.2a Step C — structural oracle (simulator/runtime/workload.structural_check).

Validates the built per-token DAG's TOPOLOGY against the INDEPENDENTLY-loaded
trace-truth fixture (R1), plus architecture-config invariants. The DAG structure is
fixture-derived, so the fixture comparison is a faithful-reproduction + length-
independence consistency check (built at L != the committed fixture lengths); the
genuinely-independent content is the config cross-check (softmax==n_layers, residual
joins==2*n_layers, one embedding — from op_profile.Model.config, not the fixture) and
the attention cross-category/cross-unit chain integrity (S1-2).

    .venv/bin/pytest tests/test_dag_structure.py
"""
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "trace_export"))
from simulator.runtime.workload import build_token_dag, structural_check  # noqa: E402
from simulator.runtime.scheduler import all_cim_assign  # noqa: E402
import op_profile  # noqa: E402

MODELS = ["llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b", "qwen2.5-7b"]


def _children(dag):
    ch = defaultdict(list)
    for n in dag.nodes:
        for d in n.deps:
            ch[d].append(n)
    return ch


def test_structural_oracle_all_models_both_phases():
    # built at L NOT among the committed fixture lengths (256/1024 prefill, 512/1024
    # decode) -> also validates length-independence of the topology.
    for m in MODELS:
        for phase, L in (("decode", 300), ("prefill", 700)):
            ok, detail = structural_check(m, phase, L)
            assert ok, f"{m} {phase}@{L}: {detail}"


def test_qkv_fanout():
    # the input-norm output feeds >= 3 matmul siblings (Q/K/V projections)
    dag = build_token_dag("llama-3.2-1b", "decode", 512)
    ch = _children(dag)
    qkv = [p for p, kids in ch.items() if sum(1 for c in kids if c.category == "matmul") >= 3]
    assert qkv, "no node feeding >=3 matmul children (Q/K/V fanout)"


def test_gate_up_siblings():
    # the post-attention-norm output feeds >= 2 matmul siblings (gate/up projections)
    dag = build_token_dag("llama-3.2-1b", "decode", 512)
    ch = _children(dag)
    gu = [p for p, kids in ch.items() if sum(1 for c in kids if c.category == "matmul") >= 2]
    assert gu, "no node feeding >=2 matmul children (gate/up siblings)"


def test_attention_chain_crosses_category_and_unit():
    # QK^T(attn,cim) -> scale(attn) -> mask(attn) -> softmax(SOFTMAX,cpu) -> S·V(attn,cim):
    # softmax is a SEPARATE category on a DIFFERENT unit (S1-2), fed by and feeding attention.
    dag = all_cim_assign(build_token_dag("llama-3.2-1b", "decode", 512))
    softmaxes = [n for n in dag.nodes if n.category == "softmax"]
    assert softmaxes
    for s in softmaxes:
        assert s.unit == "cpu", "AllCim softmax must be on CPU (cross-unit from cim attention)"
        deps = [dag[d] for d in s.deps]
        assert any(d.category == "attention" and d.unit == "cim" for d in deps), \
            "softmax not fed by a cim attention node (mask-add)"
        # the mask-add is itself fed by an attention node (scale) -> multi-step attention chain
        mask = next(d for d in deps if d.category == "attention")
        assert any(dag[x].category == "attention" for x in mask.deps), "no scale before mask in chain"
        succ = [dag[x] for x in dag.successors(s.id)]
        assert any(x.category == "attention" and x.unit == "cim" for x in succ), \
            "softmax does not feed a cim attention node (S·V)"


def test_residual_joins_match_layer_count():
    dag = build_token_dag("llama-3.2-1b", "decode", 512)
    n_layers = op_profile.Model("llama-3.2-1b").config["n_layers"]
    joins = [n for n in dag.nodes if n.category == "residual" and len(n.deps) >= 2]
    assert len(joins) == 2 * n_layers, f"{len(joins)} residual joins != 2*{n_layers}"


def test_global_nodes_count_and_position():
    dag = build_token_dag("llama-3.2-1b", "decode", 512)
    cfg = op_profile.Model("llama-3.2-1b").config
    embeds = [n for n in dag.nodes if n.category == "embedding"]
    assert len(embeds) == 1 and not embeds[0].deps, "embedding must be a single root node"
    # lm_head = the last matmul, output width = vocab
    matmuls = [n for n in dag.nodes if n.category == "matmul"]
    lm_head = max(matmuls, key=lambda n: n.id)
    assert lm_head.wl.N == cfg["vocab"], "last matmul is not lm_head (N != vocab)"
    assert lm_head.id == max(n.id for n in dag.nodes), "lm_head is not the final node"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"\n{len(fns)} structural-oracle tests passed.")
