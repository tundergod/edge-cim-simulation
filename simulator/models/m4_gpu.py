"""M4 — Mali-G610 GPU timing model (Phase 1 fit). params/m4_gpu.json.

The GPU's role in the CIM-centric design is the ATTENTION OFFLOAD (native
activation x activation bmm); its matmul throughput is a self-written unoptimised
OpenCL kernel -> absolute GEMM throughput is treated as a LOWER BOUND, only the
shape-trend is fit.

attn_bmm_us(kv): single-head decode attention (QK^T + S.V), linear in kv length,
fit from measurements/aetina/mali_matmul.json (attn group, f16). This is the offload
reference used in the end-to-end recompose (step 9).
"""
import json
from pathlib import Path

_PARAMS = Path(__file__).parent / "params" / "m4_gpu.json"


class MaliGpuModel:
    def __init__(self, params=None):
        p = params if params is not None else json.loads(_PARAMS.read_text())
        self.attn_a_us = p["attn_bmm_a_us"]       # intercept (us), single-head qkT+sv
        self.attn_b_us = p["attn_bmm_b_us_per_kv"]  # slope (us per kv token)
        self.gemm_gflops_lb = p["gemm_gflops_saturated_lowerbound"]

    def attn_bmm_us(self, kv, heads=1, layers=1):
        """Single-head decode attention (QK^T+S.V) us, x heads x layers if given."""
        return (self.attn_a_us + self.attn_b_us * kv) * heads * layers

    def gemm_lat_us(self, M, K, N):
        """GEMM latency (us) = FLOPs / saturated throughput. LOWER BOUND (unoptimised kernel)."""
        return 2.0 * M * K * N / (self.gemm_gflops_lb * 1e9) * 1e6
