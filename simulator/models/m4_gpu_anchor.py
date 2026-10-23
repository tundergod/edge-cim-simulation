"""M4 — HeteroInfer-ANCHORED mobile-GPU kernel-time model (Snapdragon 8 Gen 3 / Adreno-750-like),
the GPU slot of SoC profile `sd8gen3_cim` (ADR-0008, revised F2/F3).

HONESTY: EVERY number here is `borrowed` — fitted to the latencies HeteroInfer (SOSP'25,
docs: cim-mobile-soc/docs/heteroinfer-design-reference.md §3b Table 3, §2.1 Fig.1) PUBLISHED
for its own W4A16 OpenCL kernels on an 8 Gen 3. None of it is our silicon; the fit is a
reproduction of six published anchors (tools/analysis/fit_hetero_anchors.py writes the
`anchor_fit` block of simulator/specs/gpu_adreno750_sd8gen3.json and the residual table in
validation/reports/hetero/anchor_fit.json). Shapes follow the paper's semantics: weight
[N_out, K_in], activation [K_in, M] (M = sequence rows; M=1 = decode GEMV).

KERNEL TIME ONLY (F2): the model prices what the kernel needs once its operands are available;
the weight/activation DRAM bytes are metered by the runtime's shared pool (engine
max(kernel, bytes/BW)), never here — so a weight-centric split part gains the multi-stream
headroom of HeteroInfer §3.3. The anchors are fitted under that decomposition
(pred = max(kernel(M,K,N), K*N*weight_bytes_per_param / pool_GBs)); all six are kernel-bound.

Model (latency us):
  GEMV regime (M == 1):   t = gemv_floor_us + K * (N + n_half) / rate
      work-proportional dequant+MAC cost at `rate` (Gparam/s) with an OCCUPANCY term: a matvec has
      only N outputs = N independent work-items, so a narrow N under-fills the GPU; (N + n_half)/N
      is the occupancy penalty the three decode anchors imply (their effective rate rises with
      N_out: 4096 -> 28672). `gemv_floor_us` is bounded to the §4.3 "tens of us" launch band
      (<= 60 us) by the fit, so a small matvec costs a launch + its work, not a 320 us constant.
  GEMM regime (M > 1):    t = gemm_floor_us + 2*M*K*N / C_eff(M),  C_eff(M) = C_max * M/(M+M_half)
      (Fig.1 "TFLOPS rises ~linearly with rows then plateaus at ~1 TFLOPS"; affine in M.) Its own
      floor (the GEMM kernel's tiling/launch cost, fitted) is what keeps a 1/8-wide GEMM slice from
      being free in the solver (Table 3 "NPU-only" at M=128).
  Bridging 1 < M < ~7 (no anchors between M=1 and M=128): min(M * GEMV, GEMM) — ASSUMPTION
      (the better of looping the matvec kernel or one GEMM launch).
  Attention (act x act bmm, QK^T + S.V composite, batched over heads) — EXTRAPOLATED: the paper
      has NO attention anchor (checklist item 9). Two kernels x `kernel_launch_us` + (M > 1 only)
      FLOPs / C_eff(heads*M); the bmm bytes are pool-metered.
  Elementwise support ops: priced by the Platform as the per-fused-kernel launch floor (bytes
      metered by the M3 pool); `kernel_launch_us` is an ASSUMPTION from §4.3 ("tens of us").
"""
from simulator.models.engine import UnitEngine, Workload


class GpuAnchorRooflineModel(UnitEngine):
    """Table-3-anchored 2-regime kernel-time model (borrowed; sd8gen3_cim). Spec = gpu_adreno750_sd8gen3."""

    def __init__(self, spec, engine="analytic"):
        super().__init__(spec, engine)
        if engine != "analytic":
            raise ValueError(f"GpuAnchorRooflineModel has only the analytic backend (got engine={engine!r})")
        fit = spec["anchor_fit"]
        self.gemv_floor_us = float(fit["gemv_floor_us"])
        self.rate = float(fit["gemv_rate_gparams_s"])        # Gparam/s (dequant + MAC throughput at full occupancy)
        self.n_half = float(fit["n_half"])                   # occupancy half-width (output columns)
        self.gemm_floor_us = float(fit["gemm_floor_us"])
        self.c_max_tflops = float(fit["c_max_tflops"])
        self.m_half = float(fit["m_half"])
        self.launch_us = float(spec["kernel_launch_us"])
        self.calib_dtype = spec.get("calibration_dtype", "w4a16")

    # ---- regimes -----------------------------------------------------------
    def gemv_us(self, K, N):
        return self.gemv_floor_us + K * (N + self.n_half) / (self.rate * 1e3)

    def c_eff_tflops(self, rows):
        return self.c_max_tflops * rows / (rows + self.m_half)

    def gemm_us(self, M, K, N):
        return self.gemm_floor_us + 2.0 * M * K * N / (self.c_eff_tflops(M) * 1e12) * 1e6

    def matmul_us(self, M, K, N):
        """(latency_us, regime). M<=1 GEMV; M>1 min(M*GEMV, GEMM) (= GEMM beyond the crossover)."""
        if M <= 1:
            return self.gemv_us(K, N), "gemv"
        loop, gemm = M * self.gemv_us(K, N), self.gemm_us(M, K, N)
        return (loop, "gemv-loop") if loop < gemm else (gemm, "gemm")

    def attention_us(self, M, hd, kv, heads):
        """QK^T + S.V composite (two kernels), heads batched; compute only (bytes pool-metered)."""
        if M <= 1:
            return 2.0 * self.launch_us, "floor"
        flops = 2.0 * 2.0 * heads * M * hd * kv
        return 2.0 * self.launch_us + flops / (self.c_eff_tflops(heads * M) * 1e12) * 1e6, "compute"

    # ---- frozen engine contract ---------------------------------------------
    def predict(self, wl: Workload) -> dict:
        dflag = "" if wl.dtype in ("fp16", "int4", self.calib_dtype) else (
            f" (dtype={wl.dtype} requested; priced with the {self.calib_dtype}-calibrated kernel"
            f" — no such anchor in the paper)")
        if wl.op in ("attention", "attn_bmm"):
            hd = wl.extra.get("hd", wl.K or 128)
            lat, bound = self.attention_us(max(1, wl.M), hd, max(1, wl.kv), max(1, wl.heads))
            prov = ("extrapolated: GPU attention QK^T+S.V composite on the Table-3-anchored kernel model "
                    "(no attention anchor in HeteroInfer; two kernels x launch %.0f us, %s; bytes "
                    "pool-metered; borrowed sd8gen3)%s" % (self.launch_us, bound, dflag))
            return {"latency_us": float(lat), "bound": bound, "provenance": prov}
        if wl.K <= 0 or wl.N <= 0:
            raise ValueError(f"GpuAnchorRooflineModel GEMM needs K,N > 0 (got K={wl.K}, N={wl.N})")
        lat, regime = self.matmul_us(max(1, wl.M), wl.K, wl.N)
        if wl.N < 4096:        # the occupancy term is fitted on N_out in {4096, 28672}; narrower outputs extrapolate it
            dflag += " (EXTRAPOLATED: N_out=%d below the smallest Table-3 anchor 4096, occupancy term unanchored)" % wl.N
        prov = ("borrowed: GPU %s regime, HeteroInfer Table 3 anchored 2-regime kernel time "
                "(gemv floor=%.0f us, rate=%.1f Gparam/s, n_half=%.0f; gemm floor=%.0f us, C_max=%.2f TFLOPS, "
                "M_half=%.0f; max anchor residual %.1f%%; DRAM bytes pool-metered; no silicon of ours)%s"
                % (regime, self.gemv_floor_us, self.rate, self.n_half, self.gemm_floor_us, self.c_max_tflops,
                   self.m_half, 100 * float(self.spec["anchor_fit"].get("max_abs_rel_residual", 0.0)), dflag))
        return {"latency_us": float(lat), "bound": "compute", "provenance": prov}
