"""`heteroinfer_layer` registry entry (the paper's "Hetero-layer" baseline, 2026-10-26): HeteroInfer's per-op solver
restricted to a whole-op gpu/npu choice (no tensor split, no activation chunking).

    .venv/bin/pytest tests/test_heteroinfer_layer.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from simulator.runtime.config import SimConfig, _KNOWN_SCHEDULERS  # noqa: E402
from simulator.runtime.platform import Platform  # noqa: E402
from simulator.runtime.runner import run  # noqa: E402
from simulator.runtime.scheduler import SCHEDULERS, HeteroInferScheduler, plans_summary  # noqa: E402
from simulator.runtime.workload import build_token_dag  # noqa: E402

MODEL, PROFILE = "llama-3.2-1b", "sd8gen3_cim"


class Cfg:
    model = MODEL
    units = {"cim": True, "gpu": True, "npu": True, "cpu": True}
    sync_mode = "fast"
    sync_us_override = None
    partition_grid = 8
    precision_boundary_placement = "consumer"
    host = None
    allow_cpu_compute = None


def test_registered_and_whole_op_only():
    assert "heteroinfer_layer" in SCHEDULERS and "heteroinfer_layer" in _KNOWN_SCHEDULERS
    s = SCHEDULERS["heteroinfer_layer"]
    assert isinstance(s, HeteroInferScheduler) and s.name == "heteroinfer_layer"
    assert s.matmul_dims == () and s.std_lengths == () and s.host == "gpu" and s.support_unit == "gpu"
    plat = Platform(MODEL, soc_profile=PROFILE)
    for phase, L in (("decode", 512), ("prefill", 256), ("prefill", 300)):
        dag = s.assign(build_token_dag(MODEL, phase, L), Cfg(), platform=plat)
        strategies = plans_summary(dag)
        assert set(strategies) <= {"gpu_only", "npu_only", "gpu_fixed", "mem_fixed", "cpu_fixed"}, (phase, strategies)
        assert not any(n.category == "merge" for n in dag.nodes)          # no tensor split, no chunking
        assert {n.unit for n in dag.nodes if n.category == "matmul"} <= {"gpu", "npu"}
        assert all(n.unit == "gpu" for n in dag.nodes if n.category in ("norm", "rope", "ffn", "softmax", "residual", "attention"))
        assert dag.plan_source == "solver:gpu"


def test_layer_is_never_faster_than_tensor_and_runner_wires_it():
    def cfg(policy):
        return SimConfig.from_dict({"workload": {"model": MODEL, "context": 1024, "prefill_len": 256, "decode_len": 8},
                                    "platform": {"soc_profile": PROFILE, "units": {"gpu": True, "npu": True, "cim": True, "cpu": True}},
                                    "scheduler": {"policy": policy}})
    layer, tensor, gpu = run(cfg("heteroinfer_layer")), run(cfg("heteroinfer")), run(cfg("gpu_only"))
    assert layer["plan_source"] == "solver:gpu" and layer["merge_count_per_decode_token"] == 0
    assert layer["tok_s"] <= tensor["tok_s"] * 1.001 and layer["prefill_tok_s"] <= tensor["prefill_tok_s"] * 1.001
    assert layer["tok_s"] >= gpu["tok_s"] * 0.999      # the solver may keep everything on the GPU, never worse
    assert set(layer["plans_summary"]["decode"]) <= {"gpu_only", "npu_only", "gpu_fixed", "mem_fixed", "cpu_fixed"}
