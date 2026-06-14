"""Phase 2.2b Step C — group-aware GPU-attention composite pricing (R2).

`m4_gpu.attn_bmm_us` prices the QK^T+S·V pair as one combined cost. When attention is
on the GPU (CimHetero), the two bmm nodes of an attention block share a `pricing_group`
(= the QK^T rep's id) so the COMPUTE is priced ONCE: the rep returns attn_bmm_us, the
other returns 0. The invariant is on COMPUTE (price/compute_us), not engine token latency
— each bmm still carries its own real K/V-cache memory stream. scale/mask/softmax are NOT
priced as a bmm. The 2.2a fail-loud is kept for an ungrouped GPU bmm / a non-bmm subtype.

    .venv/bin/pytest tests/test_pricing.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "trace_export"))
from simulator.runtime.workload import build_token_dag  # noqa: E402
from simulator.runtime.platform import Platform  # noqa: E402
from simulator.runtime.dag import OpNode  # noqa: E402
from simulator.models.engine import Workload  # noqa: E402


def _cimhetero_place(dag):
    for n in dag.nodes:
        if n.category == "matmul":
            n.unit = "cim"
        elif n.category == "attention":
            n.unit = "gpu" if "hd" in n.wl.extra else "cpu"
        elif n.category in ("softmax", "norm", "rope", "ffn", "residual"):
            n.unit = "cpu"
        else:
            n.unit = "mem"
    return dag


def test_gpu_attention_composite_priced_once():
    dag = _cimhetero_place(build_token_dag("llama-3.2-1b", "decode", 512))
    plat = Platform("llama-3.2-1b")
    groups = {}
    for n in dag.nodes:
        if n.category == "attention" and n.unit == "gpu":
            groups.setdefault(n.pricing_group, []).append(n)
    assert groups and all(g is not None for g in groups)
    for pg, members in groups.items():
        assert len(members) == 2                       # QK^T + S·V
        total = sum(plat.price(n)["latency_us"] for n in members)
        rep = next(n for n in members if n.id == pg)
        once = plat.gpu.attn_bmm_us(rep.wl.kv or 1, heads=rep.wl.heads, layers=1)
        assert abs(total - once) < 1e-9                # composite COMPUTE priced exactly once


def test_scale_mask_softmax_not_priced_as_bmm():
    dag = _cimhetero_place(build_token_dag("llama-3.2-1b", "decode", 512))
    plat = Platform("llama-3.2-1b")
    nonbmm = [n for n in dag.nodes if (n.category == "attention" and "hd" not in n.wl.extra)
              or n.category == "softmax"]
    assert nonbmm
    for n in nonbmm:
        assert plat.price(n)["source_model"] != "m4_gpu"


def test_ungrouped_or_nonbmm_gpu_attention_fails_loud():
    plat = Platform("llama-3.2-1b")
    scale = OpNode(id=0, category="attention", unit="gpu",
                   wl=Workload(op="attention", extra={"aten": "aten.mul.Tensor"}))
    bmm_nogroup = OpNode(id=1, category="attention", unit="gpu", pricing_group=None,
                         wl=Workload(op="attention", kv=512, heads=32, K=64, extra={"hd": 64}))
    for bad in (scale, bmm_nogroup):
        try:
            plat.price(bad)
        except NotImplementedError:
            continue
        raise AssertionError(f"GPU attention {bad.wl.extra} not fail-loud")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"\n{len(fns)} pricing tests passed.")
