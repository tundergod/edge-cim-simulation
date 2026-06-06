"""M4 — Mali-G610 GPU ANALYTIC ROOFLINE slot (Phase 1.2, swappable model engine).

Coexists with the Phase-1.1 micro-benchmark model (m4_gpu.py / MaliGpuModel) — that one
stays the PRIMARY GPU source; this is the swap slot for model-swap (D4): a self-contained
roofline that takes ONLY a spec at construction and conforms to the frozen UnitEngine
contract {latency_us, bound, provenance}.

    latency_us = max(compute_us, memory_us)
      compute_us = 2*M*K*N / (eff_compute * fp16_peak_gflops)        # FLOPs / effective ceiling
      memory_us  = nbytes  / mem_eff_BW_GBs                          # bytes / effective BW
      bytes (if not given) = (K*N + M*K + M*N) * bytes_per_elem      # weight + act-in + act-out

HONESTY (D4, non-negotiable):
  - Both axes are CALIBRATED to mali_matmul.json (FP16): eff_compute = saturated f16
    GFLOP/s (ksweep, 5 pts) / fp16 peak; mem_eff_BW = lstsq fit to the FP16 decode-GEMV
    points. This is FP16 only.
  - INT8 GPU GEMM has ZERO data (the Mali matmul kernel is FP32/FP16). predict() on an
    int8 workload still uses the FP16-calibrated ceilings -> flagged in provenance.
  - The whole model is a SHAPE-TREND fit, NOT a strict lower bound: an unoptimised OpenCL
    kernel and only 5 saturation points -> NOT a transferable calibration. predicted is mostly
    <= measured (~2/3; frac_pred_le_measured~0.53) but ~1/3 over-predict by up to +5%;
    provenance says 'simulated (roofline shape-trend)'.
  - FP32 peak 512 GFLOP/s in the spec is an assumption (may underestimate 2-4x); this
    model calibrates against FP16, so it does not rely on that assumption.
  - ksweep_saturation_M is a DEAD param in the spec (kept, not deleted, per audit);
    unused here.
"""
from simulator.models.engine import UnitEngine

_BYTES_PER_ELEM = {"fp32": 4, "fp16": 2, "int8": 1}


class GpuRooflineModel(UnitEngine):
    """Mali-G610 analytic roofline (FP16-calibrated, shape-trend; NOT a strict lower bound). Spec = gpu_mali_g610."""

    def __init__(self, spec, engine="analytic"):
        super().__init__(spec, engine)
        fit = spec["roofline_fit"]                          # calibrated block (fit_gpu_roofline.py)
        self.fp16_peak = spec["fp16_peak_gflops"]           # 1024 (assumption)
        self.eff_compute = fit["eff_compute_fp16"]          # saturated f16 / fp16 peak
        self.ceil_gflops = self.fp16_peak * self.eff_compute  # effective compute ceiling (~20 GFLOP/s)
        self.mem_eff_BW_GBs = fit["mem_eff_BW_GBs"]         # FP16 decode-GEMV lstsq fit (~1.26)

    def predict(self, wl):
        bpe = _BYTES_PER_ELEM.get(wl.dtype, 2)
        nbytes = wl.nbytes or (wl.K * wl.N + wl.M * wl.K + wl.M * wl.N) * bpe
        compute_us = 2.0 * wl.M * wl.K * wl.N / (self.ceil_gflops * 1e9) * 1e6
        memory_us = nbytes / (self.mem_eff_BW_GBs * 1e9) * 1e6
        if compute_us >= memory_us:
            latency_us, bound = compute_us, "compute"
        else:
            latency_us, bound = memory_us, "memory"
        int8_flag = " (dtype=int8 has ZERO GPU data; FP16 ceilings used)" if wl.dtype == "int8" else ""
        prov = (f"simulated (roofline shape-trend, FP16-calibrated to mali_matmul.json; "
                f"ceil={self.ceil_gflops:.2f} GFLOP/s, BW={self.mem_eff_BW_GBs:.2f} GB/s; "
                f"shape-trend fit, NOT a strict lower bound, NOT transferable){int8_flag}")
        return {"latency_us": latency_us, "bound": bound, "provenance": prov}
