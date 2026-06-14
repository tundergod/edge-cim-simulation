"""Phase 2.2b Step B — value-based conversion-op (simulator/runtime/precision.py).

`insert_conversions` puts a `convert` OpNode on each value-flow edge (p->c) where
EXACTLY ONE endpoint is on the GPU (the fp16 attention island) AND the two sides
differ in sim_precision (int8<->fp16) — the dequant/requant of the project's mixed-
precision config. It is keyed on the (unit-pair, precision) of the edge, NOT on the
precision delta alone, so AllCim (no GPU node) inserts ZERO conversions and its L4 is
untouched. Returns a REBUILT Dag (fresh successor index); idempotent. Conversion bytes
come from the produced value's element count (`out_elems`), and a convert is priced as
a pure memory-bound cast (compute = 0).

    .venv/bin/pytest tests/test_precision.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "trace_export"))
import op_profile  # noqa: E402
from simulator.runtime.workload import build_token_dag  # noqa: E402
from simulator.runtime.scheduler import all_cim_assign  # noqa: E402
from simulator.runtime.precision import insert_conversions  # noqa: E402

MODELS = ["llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b", "qwen2.5-7b"]


def _cimhetero_place(dag):
    """CimHetero placement (matmul->cim/int8, attention BMM->gpu/fp16, scale/mask + support
    ->cpu, kv_cache/embedding->mem). The BMM is distinguished from scale/mask by wl.extra['hd']."""
    for n in dag.nodes:
        if n.category == "matmul":
            n.unit = "cim"
        elif n.category == "attention":
            n.unit = "gpu" if "hd" in n.wl.extra else "cpu"   # QK^T/S·V bmm vs scale/mask
        elif n.category in ("softmax", "norm", "rope", "ffn", "residual"):
            n.unit = "cpu"
        else:
            n.unit = "mem"   # kv_cache, embedding
    return dag


def _expected_edges(dag):
    """Independent oracle: edges that should get a convert (re-implements the rule)."""
    by_id = {n.id: n for n in dag.nodes}
    exp = set()
    for c in dag.nodes:
        for d in c.deps:
            p = by_id[d]
            if (p.category != "convert" and c.category != "convert"
                    and p.precision != c.precision
                    and ((p.unit == "gpu") + (c.unit == "gpu")) == 1):
                exp.add((d, c.id))
    return exp


def test_allcim_inserts_zero_conversions():
    dag = all_cim_assign(build_token_dag("llama-3.2-1b", "decode", 512))
    out = insert_conversions(dag)
    assert sum(1 for n in out.nodes if n.category == "convert") == 0


def test_allcim_cim_to_cpu_edge_not_converted():
    # the load-bearing guard: a cim(int8 matmul)->cpu(fp16 support) edge HAS a precision
    # delta but NO gpu endpoint, so it must NOT convert (rule keys off the GPU unit-pair,
    # not the precision delta — else AllCim would get conversions and break the hard gate).
    dag = all_cim_assign(build_token_dag("llama-3.2-1b", "decode", 512))
    by_id = {n.id: n for n in dag.nodes}
    cim_to_cpu = [(by_id[d], c) for c in dag.nodes for d in c.deps
                  if by_id[d].unit == "cim" and c.unit == "cpu"
                  and by_id[d].precision != c.precision]
    assert cim_to_cpu, "expected at least one cim(int8)->cpu(fp16) edge in AllCim"
    assert sum(1 for n in insert_conversions(dag).nodes if n.category == "convert") == 0


def test_cimhetero_conversions_match_oracle_all_models():
    for m in MODELS:
        for phase in ("decode", "prefill"):
            dag = _cimhetero_place(build_token_dag(m, phase, 512))
            exp = _expected_edges(dag)
            out = insert_conversions(dag)
            n_conv = sum(1 for n in out.nodes if n.category == "convert")
            assert n_conv == len(exp), f"{m} {phase}: {n_conv} converts != oracle {len(exp)}"
            assert n_conv > 0


def test_cimhetero_decode_three_per_layer():
    # the real int8<->fp16 GPU boundaries per decode layer: K dequant + V dequant (kv_cache
    # int8 -> bmm fp16) + S·V requant (gpu fp16 -> O-proj int8) = 3 x n_layers.
    m = "llama-3.2-1b"
    nL = op_profile.Model(m).config["n_layers"]
    out = insert_conversions(_cimhetero_place(build_token_dag(m, "decode", 512)))
    assert sum(1 for n in out.nodes if n.category == "convert") == 3 * nL


def test_convert_node_properties_and_rebuilt_dag():
    dag = _cimhetero_place(build_token_dag("llama-3.2-1b", "decode", 512))
    out = insert_conversions(dag)
    convs = [n for n in out.nodes if n.category == "convert"]
    assert convs
    by_id = {n.id: n for n in out.nodes}
    for cv in convs:
        assert cv.unit in ("cim", "cpu", "gpu", "mem") and cv.mem_domain in ("dram", "cpu_cache", "none")
        assert len(cv.deps) == 1 and cv.in_values == cv.deps and cv.out_value == cv.id
        p = by_id[cv.deps[0]]
        assert cv.bytes_streamed == p.out_elems * 3   # int8(1) read + fp16(2) write
    assert out.is_acyclic()
    # rebuilt successor index is consistent with the new deps
    for n in out.nodes:
        for s in out.successors(n.id):
            assert n.id in by_id[s].deps


def test_insert_conversions_idempotent():
    dag = _cimhetero_place(build_token_dag("llama-3.2-1b", "decode", 512))
    once = insert_conversions(dag)
    n1 = sum(1 for n in once.nodes if n.category == "convert")
    twice = insert_conversions(once)
    n2 = sum(1 for n in twice.nodes if n.category == "convert")
    assert n1 == n2 and n1 > 0           # re-running does not double-insert


def test_convert_priced_as_pure_memory():
    from simulator.runtime.platform import Platform
    dag = _cimhetero_place(build_token_dag("llama-3.2-1b", "decode", 512))
    out = insert_conversions(dag)
    plat = Platform("llama-3.2-1b")
    cv = next(n for n in out.nodes if n.category == "convert")
    p = plat.price(cv)
    assert p["latency_us"] == 0.0 and p["source_model"] == "convert"   # cost is bytes only


def test_build_token_dag_sets_out_elems():
    dag = build_token_dag("llama-3.2-1b", "decode", 512)
    compute = [n for n in dag.nodes if n.category in ("matmul", "attention", "kv_cache")]
    assert compute and all(n.out_elems > 0 for n in compute)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"\n{len(fns)} precision tests passed.")
