"""Phase 2.2a Step A — trace-truth fixture (tools/trace_export/trace_fixture.py).

The fixture is the INDEPENDENT value-flow ground truth the structural oracle
(Step C) validates the M5 DAG against (R6 anti-self-confirmation: it comes from a
fresh PyTorch eager trace, not from build_token_dag's template). Covers:

- R5  value-flow integrity: every `in_value` resolves to an earlier `out_value`
      or an explicit external (weight/const/cache); alias/view -> `alias_of`;
      multi-output ops keep every value; unresolved edge fails loud.
- S1-1 adversarial id-reuse: CPython recycles id() of GC'd tensors (verified:
      20 seq FakeTensors -> 6 distinct ids without strong refs); the recorder's
      keep-list must hold every tensor so producer ids never recycle into a
      fabricated edge.
- R4  every op carries `trace_dtype` (eager reality) AND `sim_precision`
      (ADR-0004c unit-native placement); the two differ (#8).
- R6  committed fixture compute-op counts == op_profile counts at each length.

    .venv/bin/pytest tests/test_trace_fixture.py
"""
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "trace_export"))

import trace_fixture as tf  # noqa: E402

MODELS = {"llama-3.2-1b": "meta-llama/Llama-3.2-1B",
          "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
          "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
          "qwen2.5-7b": "Qwen/Qwen2.5-7B"}


# ---------- R4 / S2-3: precision contract ----------

def test_precision_contract_maps_categories():
    pc = tf.PRECISION_CONTRACT
    assert pc["matmul"] == "int8"           # CIM native (ADR-0004c)
    assert pc["attention"] == "fp16"        # Mali GPU native
    # support ops are non-quantized (fp16/fp32), never int8
    for cat in ("softmax", "norm", "rope", "ffn", "residual"):
        assert pc[cat] in ("fp16", "fp32")
    # every value is a recognised precision tier
    assert set(pc.values()) <= {"int8", "fp16", "fp32"}


# ---------- S1-1: keep-list prevents id-reuse ----------

def test_keeplist_prevents_id_reuse_under_gc():
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    fake = FakeTensorMode(allow_non_fake_inputs=True)
    ids = []
    with fake, tf.EdgeRecorder() as rec:
        for _ in range(30):
            t = torch.zeros(4, 4)        # one fresh tensor the recorder must retain
            ids.append(id(t))
            del t
            gc.collect()                 # would recycle id() if the recorder didn't hold a ref
    # the recorder's strong-ref keep-list kept every tensor alive -> no id recycled
    assert len(set(ids)) == 30, f"id() recycled ({len(set(ids))}/30) -> keep-list not holding refs"
    # and no two recorded outputs share a value-id (monotonic, injective)
    all_out = [v for r in rec.records for v in r["out_values"]]
    assert len(all_out) == len(set(all_out)), "value-id collision across recorded outputs"


# ---------- R5: alias / multi-output / resolution ----------

def test_alias_view_records_alias_of():
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    fake = FakeTensorMode(allow_non_fake_inputs=True)
    with fake, tf.EdgeRecorder() as rec:
        a = torch.zeros(4, 8)
        b = a.transpose(0, 1)            # view: output aliases a's storage
        del a, b
    aliasing = [r for r in rec.records if r["alias_of"]]
    assert aliasing, "no alias_of recorded for a view op"
    for r in aliasing:
        for out_v, src_v in r["alias_of"].items():
            assert src_v in r["in_values"], "alias_of must point at an input value-id"
            assert out_v in r["out_values"]


def test_multi_output_op_keeps_every_value():
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    fake = FakeTensorMode(allow_non_fake_inputs=True)
    with fake, tf.EdgeRecorder() as rec:
        x = torch.zeros(4, 4)
        torch.sort(x)                    # returns (values, indices) -> 2 tensor outputs
    sorts = [r for r in rec.records if "sort" in r["op"]]
    assert sorts, "sort op not captured"
    assert len(sorts[0]["out_values"]) == 2, "multi-output op dropped a value"


def test_validate_value_flow_passes_on_real_trace():
    recs = tf.trace_phase("llama-tiny", _tiny_cfg(), "decode", 8)
    assert tf.validate_value_flow(recs) is True
    # every in_value is produced earlier or external — restated independently
    produced, ext = set(), set()
    for r in recs:
        ext |= set(r["external_in"])
        for v in r["in_values"]:
            assert v in produced or v in ext, f"{r['op']}: in_value {v} unresolved"
        produced |= set(r["out_values"])


def test_validate_value_flow_fails_loud_on_dangling_edge():
    bad = [
        {"op": "a", "in_values": [], "out_values": [0], "external_in": [], "alias_of": {}},
        {"op": "b", "in_values": [99], "out_values": [1], "external_in": [], "alias_of": {}},
    ]
    try:
        tf.validate_value_flow(bad)
    except ValueError:
        return
    raise AssertionError("dangling in_value (no producer, not external) not rejected")


# ---------- R1 contraction + R4 fields ----------

def test_compute_subgraph_deps_are_earlier_compute_nodes():
    recs = tf.trace_phase("llama-tiny", _tiny_cfg(), "decode", 8)
    sub = tf.compute_subgraph(recs)
    assert sub, "empty compute subgraph"
    for i, node in enumerate(sub):
        assert node["category"] is not None
        for d in node["deps"]:
            assert 0 <= d < i, f"node {i} dep {d} is not an earlier compute node"


def test_compute_subgraph_nodes_have_dtype_and_precision():
    recs = tf.trace_phase("llama-tiny", _tiny_cfg(), "decode", 8)
    for node in tf.compute_subgraph(recs):
        assert node["trace_dtype"], "missing trace_dtype"
        assert node["sim_precision"] == tf.PRECISION_CONTRACT[node["category"]]
        # #8: eager trace dtype is NOT the simulated INT8 placement
    matmuls = [n for n in tf.compute_subgraph(recs) if n["category"] == "matmul"]
    assert matmuls and all(n["sim_precision"] == "int8" for n in matmuls)
    assert all(n["trace_dtype"] != "int8" for n in matmuls), "trace_dtype should be eager (not int8)"


# ---------- R6: committed fixtures consistent with op_profile ----------

def test_committed_fixture_counts_match_op_profile():
    import op_profile
    for key in MODELS:
        fx = tf.load_fixture(key)
        m = op_profile.Model(key)
        grid = m.grid_profile()
        for phase, lens in (("prefill", tf.PREFILL_FIX), ("decode", tf.DECODE_FIX)):
            for L in lens:
                fix_counts = _cat_counts(fx[phase][str(L)])
                prof_counts = _profile_cat_counts(grid[phase][L])
                assert fix_counts == prof_counts, (
                    f"{key} {phase}@{L}: fixture counts {fix_counts} != op_profile {prof_counts}")


# ---------- helpers ----------

def _cat_counts(nodes):
    out = {}
    for n in nodes:
        out[n["category"]] = out.get(n["category"], 0) + 1
    return out


def _profile_cat_counts(rows):
    out = {}
    for r in rows:
        out[r["category"]] = out.get(r["category"], 0) + r["count"]
    return out


def _tiny_cfg():
    """A small but structurally-real eager Llama config (no HF download)."""
    from transformers import LlamaConfig
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                      head_dim=16)
    cfg._attn_implementation = "eager"
    return cfg


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"\n{len(fns)} trace-fixture tests passed.")
