"""Phase 2.4 hardening — M4 CPU roofline (simulator/models/m4_cpu.py): the
op_us()/predict() element-count paths, the dtype fail-loud guard, and the L3
cache-tier branch of _tier_bw.

    .venv/bin/pytest tests/test_m4_cpu.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.models.m4_cpu import CpuModel, _tier_bw  # noqa: E402
from simulator.models.engine import Workload  # noqa: E402
from simulator.specs.loader import load_spec  # noqa: E402


def _cpu():
    return CpuModel(load_spec("cpu_rk3588"))


def test_op_us_named_model_path_returns_finite_latency():
    # op_us() is the convenience wrapper: it always puts extra={'model':..., 'dtype':...},
    # taking predict()'s "model"-in-extra element-count branch.
    lat = _cpu().op_us("rmsnorm", "llama-3.2-1b")
    assert 0 < lat < float("inf")


def test_predict_default_non_model_path_returns_finite_latency():
    # the "default" path: predict() called directly on a Workload with explicit size vars
    # and no 'model' key in extra -> _n_elem_from_wl (NOT the named-model convenience path).
    cpu = _cpu()
    wl = Workload(op="residual", K=2048, dtype="fp32")
    assert "model" not in wl.extra
    r = cpu.predict(wl)
    assert 0 < r["latency_us"] < float("inf")
    assert r["bound"] in ("compute", "memory")


def test_predict_rejects_unsupported_dtype():
    # Workload's default dtype is 'int8' (GEMM default); the CPU model only supports
    # fp32|fp16 -> a Workload built without an explicit fp32/fp16 dtype must fail loud.
    cpu = _cpu()
    wl = Workload(op="residual", K=2048)   # dtype defaults to 'int8'
    try:
        cpu.predict(wl)
    except ValueError:
        return
    raise AssertionError("CpuModel.predict accepted an unsupported dtype")


def test_l3_cache_tier_branch_exercised():
    # a large working set (vocab-scale element count) exceeds L2 (512 KiB/core) -> L3.
    spec = load_spec("cpu_rk3588")
    n_elem = 200_000
    bpe = 4
    single_copy = n_elem * bpe
    assert single_copy > spec["cache"]["l2_KiB_per_core"] * 1024   # confirms this hits L3, not L2
    assert _tier_bw(spec, single_copy) == spec["cache_bw_GBs"]["l3_shared_per_core"]

    cpu = _cpu()
    wl = Workload(op="sampling_argmax", N=n_elem, dtype="fp32")
    r = cpu.predict(wl)
    assert 0 < r["latency_us"] < float("inf")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"\n{len(fns)} m4 cpu tests passed.")
