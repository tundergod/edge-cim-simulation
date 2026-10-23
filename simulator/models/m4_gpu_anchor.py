"""M4 — HeteroInfer-ANCHORED mobile-GPU roofline (Snapdragon 8 Gen 3 / Adreno-750-like),
the GPU slot of SoC profile `sd8gen3_cim` (ADR-0008).

HONESTY: EVERY number here is `borrowed` — fitted to the latencies HeteroInfer (SOSP'25,
docs: cim-mobile-soc/docs/heteroinfer-design-reference.md §3b Table 3, §2.1 Fig.1) PUBLISHED
for its own W4A16 OpenCL kernels on an 8 Gen 3. None of it is our silicon; the fit is a
reproduction of six published anchors (tools/analysis/fit_hetero_anchors.py writes the
`anchor_fit` block of simulator/specs/gpu_adreno750_sd8gen3.json and the residual table in
validation/reports/hetero/anchor_fit.json). Shapes follow the paper's semantics: weight
[N_out, K_in], activation [K_in, M] (M = sequence rows; M=1 = decode GEMV).

Model (latency us; W_bytes = K*N*weight_bytes_per_param, 0.5 for the paper's INT4 weights):
  GEMV regime (M == 1):   t = gemm_floor_us + W_bytes / bw_GBs
      (memory-bound matvec; the floor absorbs the per-kernel dequant/launch cost the three
       decode anchors imply — a constant-BW roofline cannot do better than ~15% on them because
       the implied effective BW rises with N_out: 16 -> 20 -> 31 GB/s).
  GEMM regime (M > 1):    t = gemm_floor_us + 2*M*K*N / C_eff(M),  C_eff(M) = C_max * M/(M+M_half)
      (Fig.1 "TFLOPS rises ~linearly with rows then plateaus at ~1 TFLOPS"; affine in M.)
  Bridging 1 < M < ~7 (no anchors between M=1 and M=128): min(M * GEMV, GEMM) — ASSUMPTION
      (the better of looping the matvec kernel or one GEMM launch).
  Attention (act x act bmm, QK^T + S.V composite, batched over heads) — EXTRAPOLATED: the paper
      has NO attention anchor (checklist item 9). Decode (M==1): 2*launch + bytes/bw (memory-bound
      like a GEMV, two kernels). Prefill (M>1): 2*launch + max(FLOPs/C_eff(heads*M), bytes/bw).
  Elementwise support ops: priced by the Platform as the per-kernel launch floor (bytes are
      metered by the M3 pool); `kernel_launch_us` is an ASSUMPTION from §4.3 ("tens of us"
      submission overhead) — the fitted gemm_floor is a GEMV-kernel cost, NOT a launch floor
      (a 320 us floor on every small kernel would contradict the paper's 14 tok/s Llama-8B decode).
"""
from simulator.models.engine import UnitEngine, Workload


class GpuAnchorRooflineModel(UnitEngine):
    """Table-3-anchored 2-regime roofline (borrowed; sd8gen3_cim). Spec = gpu_adreno750_sd8gen3."""

    def __init__(self, spec, engine="analytic"):
        super().__init__(spec, engine)
        if engine != "analytic":
            raise ValueError(f"GpuAnchorRooflineModel has only the analytic backend (got engine={engine!r})")
        fit = spec["anchor_fit"]
        self.floor_us = float(fit["gemm_floor_us"])
        self.bw_GBs = float(fit["bw_GBs"])
        self.c_max_tflops = float(fit["c_max_tflops"])
        self.m_half = float(fit["m_half"])
        self.wbpp = float(fit["weight_bytes_per_param"])
        self.launch_us = float(spec["kernel_launch_us"])
        self.calib_dtype = spec.get("calibration_dtype", "w4a16")

    # ---- regimes -----------------------------------------------------------
    def gemv_us(self, K, N):
        return self.floor_us + K * N * self.wbpp / (self.bw_GBs * 1e3)

    def c_eff_tflops(self, rows):
        return self.c_max_tflops * rows / (rows + self.m_half)

    def gemm_us(self, M, K, N):
        return self.floor_us + 2.0 * M * K * N / (self.c_eff_tflops(M) * 1e12) * 1e6

    def matmul_us(self, M, K, N):
        """(latency_us, regime). M<=1 GEMV; M>1 min(M*GEMV, GEMM) (= GEMM beyond the crossover)."""
        if M <= 1:
            return self.gemv_us(K, N), "gemv"
        loop, gemm = M * self.gemv_us(K, N), self.gemm_us(M, K, N)
        return (loop, "gemv-loop") if loop < gemm else (gemm, "gemm")

    def attention_us(self, M, hd, kv, heads, nbytes_bmm):
        """QK^T + S.V composite (two kernels), heads batched. nbytes_bmm = bytes of ONE bmm
        (0 -> derived from the shape at fp16)."""
        if not nbytes_bmm:
            nbytes_bmm = (M * hd + hd * kv + M * kv) * heads * 2
        mem_us = 2.0 * nbytes_bmm / (self.bw_GBs * 1e3)
        if M <= 1:
            return 2.0 * self.launch_us + mem_us, "memory"
        flops = 2.0 * 2.0 * heads * M * hd * kv
        comp_us = flops / (self.c_eff_tflops(heads * M) * 1e12) * 1e6
        bound = "compute" if comp_us >= mem_us else "memory"
        return 2.0 * self.launch_us + max(comp_us, mem_us), bound

    # ---- frozen engine contract ---------------------------------------------
    def predict(self, wl: Workload) -> dict:
        dflag = "" if wl.dtype in ("fp16", "int4", self.calib_dtype) else (
            f" (dtype={wl.dtype} requested; priced with the {self.calib_dtype}-calibrated kernel, "
            f"{self.wbpp} B/weight — no such anchor in the paper)")
        if wl.op in ("attention", "attn_bmm"):
            hd = wl.extra.get("hd", wl.K or 128)
            lat, bound = self.attention_us(max(1, wl.M), hd, max(1, wl.kv), max(1, wl.heads), wl.nbytes)
            prov = ("extrapolated: GPU attention QK^T+S.V composite on the Table-3-anchored roofline "
                    "(no attention anchor in HeteroInfer; two kernels x launch %.0f us + %s-bound; "
                    "borrowed sd8gen3)%s" % (self.launch_us, bound, dflag))
            return {"latency_us": float(lat), "bound": bound, "provenance": prov}
        if wl.K <= 0 or wl.N <= 0:
            raise ValueError(f"GpuAnchorRooflineModel GEMM needs K,N > 0 (got K={wl.K}, N={wl.N})")
        lat, regime = self.matmul_us(max(1, wl.M), wl.K, wl.N)
        bound = "memory" if regime != "gemm" else "compute"
        prov = ("borrowed: GPU %s regime, HeteroInfer Table 3 anchored 2-regime roofline "
                "(floor=%.0f us, bw=%.1f GB/s, C_max=%.2f TFLOPS, M_half=%.0f; max anchor residual "
                "%.1f%%; no silicon of ours)%s"
                % (regime, self.floor_us, self.bw_GBs, self.c_max_tflops, self.m_half,
                   100 * float(self.spec["anchor_fit"].get("max_abs_rel_residual", 0.0)), dflag))
        return {"latency_us": float(lat), "bound": bound, "provenance": prov}
