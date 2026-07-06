"""Phase 2.4 hardening — smoke tests for three swappable unit engines that had no
dedicated test file: SramTier (m1_cim_spm), GpuRooflineModel (m4_gpu_roofline),
NpuModel (m4_npu). One representative predict() call per engine, spec-bound via
load_spec, asserting a finite positive latency (the frozen UnitEngine contract).

    .venv/bin/pytest tests/test_unit_smoke.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.models.m1_cim_spm import SramTier  # noqa: E402
from simulator.models.m4_gpu_roofline import GpuRooflineModel  # noqa: E402
from simulator.models.m4_npu import NpuModel  # noqa: E402
from simulator.models.engine import Workload  # noqa: E402
from simulator.specs.loader import load_spec  # noqa: E402


def test_sram_tier_predict_resident_working_set():
    sram = SramTier(load_spec("sram_metis_aipu"))
    r = sram.predict(Workload(op="stream", nbytes=1000))   # << 32 MiB L2 -> resident branch
    assert 0 < r["latency_us"] < float("inf")
    assert r["bound"] == "memory"


def test_gpu_roofline_predict_matmul():
    gpu = GpuRooflineModel(load_spec("gpu_mali_g610"))
    r = gpu.predict(Workload(op="matmul", M=1, K=128, N=128, dtype="fp16"))
    assert 0 < r["latency_us"] < float("inf")
    assert r["bound"] in ("compute", "memory")


def test_npu_predict_matmul():
    npu = NpuModel(load_spec("npu_rknpu2"))
    r = npu.predict(Workload(op="matmul", M=1, K=128, N=128, dtype="int8"))
    assert 0 < r["latency_us"] < float("inf")
    assert r["bound"] in ("compute", "memory", "floor")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"\n{len(fns)} unit smoke tests passed.")
