# Phase 0.3 — real-board characterization findings (CIM-centric)

Two machines, collect-what-you-can. **aetina** (Metis Alpha CIM + Mali-G610 GPU + RK3588 A76
CPU; SDK v1.3.1 in docker) and **metiscard** (production Metis Card 16 GiB, the L4 LLM anchor).
All figures regenerable from committed JSON (`tools/plotting/phase03_figs.py`,
`tools/analysis/cim_analysis.py`); raw per-shape data in `measurements/aetina,metis_card/`.

## Headline results

1. **CIM excels at weight-stationary matmul but is gated by a per-call DMA floor.** On the
   1×1-conv proxy, decode GEMV **device** throughput is 110–230 GOP/s (INT8), but the **per-call
   host↔device DMA floor is ~911 µs (p95 1112 µs)** — so end-to-end *system* latency of a small
   decode GEMV is ~0.9 ms regardless of its tiny compute (dev 18–165 µs). This on-PCIe floor (A1d.5,
   Fig 3) is the structural CIM cost the simulator must model: CIM wins on compute, loses on the
   round trip.

2. **CIM cannot keep attention stationary → a ~31–46 ms/token attention penalty → offload.** The
   conv-proxy attention *compute floor* is tiny (~20 µs single-head), but it ignores the per-decode-step
   reload of the growing KV-cache into the crossbar. Composing in the measured reload
   (`T_attn = L·(kv_bytes/BW + DMA_floor)`, over all L layers) gives **~31–46 ms/token (≥99 % reload,
   rising with kv)** — ~**97–376× slower than GPU-native FP16 attention (80–500 µs, Mali)** (C4, Fig 7). This is an
   Alpha-topology **upper-bound estimate**, but the order of magnitude is the point: dynamic
   activation×activation attention violates CIM's weight-stationary premise and must go to GPU/NPU.
   This is the empirical basis for the CIM-centric "design around CIM, offload attention" thesis.

3. **Micro→end-to-end bridge validates (10 % hold-out error).** Fitting the effective decode
   bandwidth on **1B+3B only** (streamed weight bytes = projections + lm_head; the input embedding is
   a decode gather, not streamed) and predicting the held-out **8B** decode tok/s (ADR-0006 hold-out)
   gives **2.44 vs measured 2.70 tok/s (10 % error)** (C5, Fig 6). Implied bandwidth
   **16.2 / 20.5 / 20.3 GB/s** (1B/3B/8B) shows a clear size
   trend — smaller models stream their weights less efficiently — bracketing the documented ~24 GB/s
   production wall. So op-level structure + Phase-0.2 counts compose into the real LLM's throughput
   (the L1→L4 link), and the residual error is genuine size-dependence, not a fit artifact.

## CIM micro-characterization (A1 / A1d)

| axis | result |
|---|---|
| **device envelope** | conv weight **K·N ≤ ~6 M params** allocatable (probed: 6.3 M OK, 8.4 M+ fail `zeMemAllocDevice`). Larger ops run as 2048-wide composite tiles (4 cores × 512×512, effective 2048 output width — *not* a single 2048×2048 array; see Phase 1 / AIPU ISSCC 2024); latency = n_tiles × tile, and the 911 µs floor × n_tiles compounds (the real CIM tiling cost). |
| **A1d.2 channel staircase** | decode dev latency rises with output channels N: 9.8 µs @64 → 24.7 @1024 → 41.2 @2048 → 82.4 @3072 (Fig 4). Roughly linear in N within a tile. |
| **A1d.3 (M,K,N) aspect** | at equal MAC (4.19 M): wide [1024→4096] 227, tall [4096→1024] 227, **square [2048→2048] 204 GOP/s** — mild (~10 %) penalty for square vs extreme aspect. |
| **A1d.4 l2 vs ddr** | **ratio 1.00–1.01× (no effect)** — Alpha has *no on-card DRAM* (`ddr` = host LPDDR over PCIe), so `dpu_constants_home` is not a meaningful residency axis here. Confirms the plan caveat: do **not** extrapolate an l2/ddr gap to the production card. |
| **A1d.6 GQA-narrow waste** | the narrow kv-projection (N=512) reaches only **112 GOP/s vs 204 for the wide gate/up (1B)** — ~55 % utilization; GQA's narrow output dimension underfills the crossbar. |

## Support + offload units

- **CPU A6 (RK3588 A76, cores 4-7):** non-GEMM support ops timed FP16/FP32 (cov <1 %): rmsnorm 157 µs,
  swiglu 558 µs, softmax@kv1024 1603 µs, argmax-sampling 986 µs (8B, FP16; numpy fp16 is emulated →
  upper bound). Feeds the simulator CPU model (M4).
- **GPU A5 (Mali-G610, OpenCL):** self-written tiled GEMM (FP16 + FP32); FP16 > FP32 throughout. Native
  attention bmm 80–500 µs (the offload reference in Fig 7). The GEMM kernel is unoptimized (~20 GFLOP/s
  on the square sweep) → treat absolute GPU matmul throughput as a **lower bound**; the CIM-vs-GPU
  *attention* contrast (3 orders of magnitude) is robust to kernel quality.
- **NPU A4 (RKNPU2):** **not collected** (ONNX→`.rknn` conversion pipeline not run this pass) — a
  collect-what-you-can gap; the attention-offload argument stands on the GPU comparison alone.

## Production Metis Card (B, the L4 anchor)

`axllm --show-stats` INT8 decode tok/s (median, ctx 1024, 3 prompts): **1B 13.07→14.77, 3B 6.38→6.99,
8B 2.70→2.92** (1-core→4-core). tok/s ∝ 1/params (weight-streaming wall). **4c/1c speedup 1.13→1.10→1.08
shrinks with model size** — the size-invariance drift ADR-0006 flags, and exactly why C5 is a
1B/3B→8B hold-out rather than a self-fit. `core_temp` tracer is unsupported on this PCIe-Rev1 board →
thermal readout is a confirmed **Phase 0.4 gap** (Alpha RK3588 thermal zones remain available).

## Artifacts

- `measurements/aetina/{metis_alpha_matmul(.json + _raw), cpu_ops, mali_matmul, cim_attention_composed}.json`
- `measurements/metis_card/{vendor_llm_int8, twopillar_prediction}.json`
- `characterization/aetina/{run_metis_cim.py, run_cpu_ops.py, run_mali_matmul/}`, `characterization/metis_card/run_vendor_llm.py`
- `tools/analysis/cim_analysis.py`, `tools/plotting/phase03_figs.py`, `docs/figures/phase0.3/*.{png,pdf,svg}`

## Gaps (collect-what-you-can)

- RKNPU2 (A4) native matmul/attention not collected; A2/A3 PCIe-contention sweep folded into the A1d.5
  per-call floor (dev-vs-system) rather than a separate concurrency sweep; large-M (≥2048) prefill
  conv-proxy tiles fail device allocation (LongBench M≈11.8k stays analytic, Phase 1); l2/ddr is a
  null axis on Alpha (above). The C4 composed attention is an Alpha-topology upper-bound estimate.
  Tiled-op GOP/s for non-2048-aligned dims (Qwen H=3584, F=18944) is biased **low** — true OPs
  over padded-tile latency (the 2048-grid over-covers K·N by ~1.24×); the 1B/3B/8B dims are
  2048-multiples and unaffected, so no headline rests on the Qwen throughput figure.
