---
type: source
title: "Accelerating LLMs using an Efficient GEMM Library and Target-Aware Optimizations on Real-World PIM Devices"
created: 2026-05-30
updated: 2026-05-30
tags: [processing-in-memory, upmem, gemm, llm-inference, tiled-gemm, tvm, compiler-optimization, quantization, dpu, real-hardware]
raw_path: raw/papers/pim-llm-pgemmlib-cgo2025.pdf
source_kind: paper
ingest_level: full
authors: [Hyeoncheol Kim, Taehoon Kim, Taehyeong Park, Donghyeon Kim, Yongseung Yu, Hanjun Kim, Yongjun Park]
venue: "CGO 2025"
year: 2025
---

# Accelerating LLMs using an Efficient GEMM Library and Target-Aware Optimizations on Real-World PIM Devices

## TL;DR

PIM-LLM is an end-to-end framework for LLM inference on real UPMEM PIM hardware, built around PGEMMlib — the first tiled GEMM library targeting multi-DPU UPMEM. GEMM accounts for 99% of LLaMA inference time on CPU, but naively distributing GEMM across DRAM banks creates transfer costs that overwhelm compute gains; PGEMMlib's Tile-Selector analytically finds near-optimal tiling to minimize host↔DPU transfers while maximizing DPU parallelism. Integrated into TVM with four PIM-aware compiler passes, PIM-LLM achieves up to 45.75x over a TVM+CPU baseline on LLaMA-7b with 2048 DPUs.

## Motivation

- GEMM dominates transformer inference: 99% of LLaMA's total inference time (§2.3, Fig. 3a); 96.5% average across BERT/GPT/LLaMA.
- UPMEM is the only commercially available general-purpose DRAM-based PIM, but lacks any GEMM library — only GEMV (matrix-vector) was previously characterized (Gómez-Luna et al. [21]; cites "Benchmarking a Real Processing-in-Memory System" — no page yet).
- Naively broadcasting B to all DPUs (BaseGEMM) incurs C2D transfer proportional to the full B matrix per DPU; at 2048 DPUs, transfer-to-kernel ratio becomes dominant for small GEMMs (Fig. 3c), eliminating PIM's bandwidth advantage.
- CPU/GPU GEMM tiling assumes shared/cache memory; UPMEM has isolated banks (MRAM 64 MB/DPU + WRAM 64 KB scratchpad) with no inter-DPU communication and strict memory contiguity requirements for UPMEM's parallel transfer API — classical tiling inapplicable.
- PIM resource allocation/deallocation overhead averages 75% of total execution time when not managed across GEMM calls (Fig. 8d).

## Method

### UPMEM Architecture Recap

A single UPMEM module has 128 DPUs (2 ranks × 8 chips × 8 DPUs). Each DPU is directly connected to a 64 MB MRAM bank via DMA. The DPU runs at 450 MHz with a 64-bit integer pipeline; FP is software-emulated. Inter-DPU communication does not exist; data passes exclusively via the host CPU using C2D (CPU-to-DPU) and D2C (DPU-to-CPU) bulk-transfer operations that require contiguous host-side memory.

### PGEMMlib — Tiled GEMM Library

PGEMMlib operates on GEMM(b, M, K, N) with partition factors (f_M, f_K, f_N) along the M, K, N dimensions. Four APIs:

| API | Tiling | C2D size/DPU | D2C size/DPU | Key property |
|-----|--------|-------------|-------------|--------------|
| **BaseGEMM** | None; each DPU gets a row-slice of A, full B | M/nDPUs × K | M/nDPUs × N | Maximum parallelism, redundant B broadcast |
| **NtileGEMM** | N-tiling: B split along columns; C split along N | M × K/nDPUs (×f_N) | reduced | Reduces C2D; best when N large |
| **KtileGEMM** | K-tiling: A and B split along K; partial results accumulated on host | M × K/f_K | M × N (×f_K partial) | Reduces C2D; increases D2C; trade-off depends on f_K |
| **MixedGEMM** | Combined N- and K-tiling | proportionally split | proportionally split | For mid-range GEMM shapes |

For batch GEMM (BtchedGEMM), DPUs are evenly partitioned across batch elements, executing multiple SingleGEMMs in parallel, reusing tiles already in MRAM.

A and B must be row-major and column-major, respectively, for optimal UPMEM contiguous transfer. Weight matrices are typically adjustable to column-major offline.

### Tile-Selector — Analytical Tiling Optimizer

Given (b, M, K, N, nDPUs), Tile-Selector returns (f_M, f_K, f_N) minimizing:

```
T_Total = T_Kernel + T_Transfer
T_Kernel ≈ nDOT_DPU / γ    (nDOT_DPU = b × f_K × M × N / nDPUs)
T_Transfer = C2D_size / C2D_bandwidth + D2C_size / D2C_bandwidth
```

Key facts: (1) kernel execution time depends only on Tile_K (WRAM alignment constraint: Tile_K × element_size ≥ 64 bytes); (2) γ (dot-products per DPU per ms) is profiled once per system via a PerfTable lookup indexed by Tile_K; (3) bandwidths for C2D and D2C are precomputed in lookup tables at multiples of 128 DPUs. Algorithm 1 enumerates all divisors of M, K, N and selects the minimizing configuration — one-time overhead, done at compile time. Tile-Selector achieves 96% of oracle performance on average; finds optimal in 18/60 cases and within top-5% in 52/60.

### PIM-LLM — TVM Integration and Four Compiler Passes

PIM-LLM imports models (PyTorch/ONNX/TF) into TVM's Relay IR, applies graph pattern matching to identify GEMM subgraphs, calls Tile-Selector per operator, and emits a runtime module that dispatches to PGEMMlib. CPU code is generated via LLVM; 8-bit quantized model (int8, TVM int8 quant) for both CPU and PIM.

Four target-aware optimization passes:

1. **Build-time Memory Layout Adjustment** — at compile time, graph analysis identifies weight operands (no runtime data dependencies) whose layout is not column-major; transposes them offline into column-major, shifting the runtime overhead to build time. Effective for BERT-large and LLaMA-7b (runtime layout adjustment was 27% avg / 53% worst-case of inference time; Fig. 10c).

2. **PIM Resource Pooling** — instead of allocating/deallocating DPUs per GEMM call, the pool is sized to the maximum nDPUs needed across all GEMM ops (determined statically from graph analysis) and allocated once at module creation; runtime merely retrieves DPUs from the pool. Eliminates the 75%-average allocation overhead (Fig. 8d).

3. **CPU/PIM Cooperation** — Tile-Selector is run for both CPU and PIM for each GEMM; the operator is statically assigned to whichever is faster. Small GEMMs (especially in attention layers) execute on CPU. Fig. 14 shows scheduling per transformer block: BERT-small offloads most GEMMs to CPU; LLaMA-7b (large) offloads nearly all to PIM. Comparison gate: P_BEST (PIM) vs P_CPU (CPU) → if P_BEST ≤ P_CPU, assign to CPU.

4. **QKV Generation Fusion** — transformer attention computes Q = input × W_Q, K = input × W_K, V = input × W_V as three independent GEMMs. TVM executes them sequentially (only intra-operator parallelism). PIM-LLM identifies groups of three GEMM ops from the same source node, concatenates their B matrices along the batch dimension into a single larger GEMM, dispatches as one fused GEMM, then splits the result. Reduces kernel launch overhead; QKV generation is 17.3% avg / 31% (LLaMA-7b) of total GEMM time (Fig. 8b). Effective for BERT-large (72 QKV groups) and LLaMA-7b (96 QKV groups); GPT2's layout prevents applicability.

### System Configuration

Intel Xeon Gold 5222 (3.8 GHz) × 2, 256 GB DDR4; 8GB DDR4 PIM Modules × 20 (160 GB PIM); 64 MB MRAM/DPU, 450 MHz, 128 EA (8GB) per module. UPMEM Driver 2021.4. Tiling in C + OpenMP; LLMs imported via PyTorch → torch.jit trace → TVM Relay IR. GCC 7.5.0, TVM v0.15.dev0, LLVM 15.0.6.

## Results

### PGEMMlib Evaluation

- NtileGEMM: shape (1, 8192, 3072, 768) achieves 17% of BaseGEMM execution time at f_N=128 (best); larger GEMMs benefit more from N-tiling because they are compute-bound.
- KtileGEMM: non-monotone in f_K — performance initially improves (C2D reduction dominates) then degrades (D2C increase + kernel time growth at low Tile_K); Fig. 11.
- SelectedGEMM (Tile-Selector output): 1.20× / 2.24× over BaseGEMM at 256 / 2048 DPUs; 18.40× scalability vs. 8.16× for BaseGEMM at rank 32 for shape (1, 1024, 1024, 1024).

### End-to-End LLM Inference (Figure 13, 2048 DPUs, seq len 128)

Normalized to BaseGEMM baseline. Key configurations:

| Config | Description | Geomean over baseline |
|--------|-------------|----------------------|
| TVM | Highly-optimized CPU (TVM codegen) | ~3.65× |
| TS | Tile-Selector only | 4.17× |
| TS+ADJ | +memory layout adjustment | 5.85× |
| TS+ADJ+QKV | +QKV fusion | 6.24× |
| PIM-LLM | +CPU/PIM cooperation | 10.09× |

LLaMA-7b specifically: 14.65× vs baseline; **45.75× vs TVM**. LLaMA-3b: 14.21× vs baseline; 10.00× vs TVM.

Scalability (Fig. 16): PIM-LLM improves consistently with nDPUs. At seq len 128: average 1.64× / 2.03× / 2.45× / 2.76× over TVM at 256 / 512 / 1024 / 2048 DPUs. At seq len 512: 1.95× / 2.55× / 3.39× / 4.15× over TVM.

Build-time memory layout adjustment yields up to 3.97× improvement for LLaMA-7b; 1.52× geomean (excluding GPT2, which has pre-compatible layout).

### CPU/PIM Cooperation Effectiveness

For small models (BERT-tiny, BERT-mini), all GEMMs execute on CPU — PIM overhead is not worth it. For large models (LLaMA-7b), nearly all GEMMs offload to PIM. PIM-LLM's selective scheduling is what enables it to outperform naïve PIM-only strategies across model sizes.

## Contributions

1. **PGEMMlib**: first tiled GEMM library for UPMEM, with four APIs (Base/N/K/Mixed GEMM + batched variant) specifically designed around UPMEM's no-shared-memory, contiguous-transfer architecture.
2. **Tile-Selector**: analytical model-based tiling selector that accounts for both kernel execution time (via PerfTable lookup) and data transfer time (via bandwidth LUT); one-time profiling overhead.
3. **PIM-LLM**: end-to-end LLM compiler framework (TVM/Relay integration) with four PIM-aware passes targeting UPMEM's architectural characteristics.
4. First in-depth analysis of GEMM execution patterns with various tiling techniques on a real UPMEM system at scale (up to 2048 DPUs).

## Limitations

- **UPMEM-only**: no inter-DPU communication means K-tiling requires full partial-result transfer via host; distributed GEMM algorithms for shared-memory architectures (cuBLAS, CUTLASS) cannot be directly applied. Generalizing to Samsung HBM-PIM or SK Hynix AiM requires re-deriving transfer models.
- **Integer-only DPUs**: 8-bit quantization required throughout; FP32/FP16 inference is software-emulated and evaluated only implicitly via int8. Sub-byte (INT4) is not explored, and UPMEM's minimum transfer granularity (64-byte aligned) limits efficiency at very narrow bitwidths.
- **Single-batch only**: evaluation at batch size b=1 (seq len 128/512). Batched LLM serving (prefill-dominant, larger b) is not characterized — PIM is likely less competitive in the prefill phase where GEMM is compute-bound.
- **Small-GEMM inefficiency exposed**: BERT-tiny and BERT-mini show no PIM advantage; CPU/PIM cooperation masks this, but PIM hardware remains idle for small-model inference.
- **QKV fusion inapplicable to GPT2**: layout incompatibility prevents fusion; its generality is model-topology-dependent.
- **No energy/power measurement**: performance-per-watt comparison vs CPU/GPU absent.
- **Tile-Selector accuracy degrades for mixed workloads**: PerfTable assumes homogeneous DPU load; irregular GEMMs (non-power-of-2 shapes) require bandwidth interpolation between lookup entries.

## D1–D9 Review-Lens Table

| Dim | Assessment |
|-----|-----------|
| D1 SOTA baseline fairness | Baselines include both BaseGEMM (naïve PIM) and TVM-optimized CPU; Oracle tiling included for Tile-Selector accuracy. GPT2-specific TVM is highly optimized, making PIM gains modest — this is reported honestly. Missing: GPU (cuBLAS) comparison at equivalent memory capacity and energy normalization. |
| D2 Novelty boundary | First GEMM library and tiling framework for UPMEM; first end-to-end LLM optimization on UPMEM hardware. Tiling for isolated-bank PIM is meaningfully different from CPU/GPU tiling. Prior UPMEM work covered only GEMV. Analytical model combining PerfTable + BW-LUT is straightforward engineering but well-executed. |
| D3 Evaluation completeness | 10 LLM variants (BERT ×6, GPT2 ×2, LLaMA 3b+7b), two sequence lengths, four DPU counts (256–2048). Ablation covers all four optimization passes individually. Per-GEMM shape analysis is thorough. Gap: no multi-batch evaluation; no energy measurement; no comparison to GPU. |
| D4 Platform credibility | **Strong**: runs on real Intel-based UPMEM server (Table 1: 20 PIM modules, 128 DPU/module, 450 MHz); 160 GB PIM memory. This is among the few LLM-PIM papers with actual hardware evaluation (not simulation). UPMEM SDK version and GCC version reported. |
| D5 Motivation | Tightly argued: GEMM-dominance (99% LLaMA time), BaseGEMM transfer-dominated behavior (Fig. 3b–c), layout-adjustment runtime overhead (27% avg) are all measured, not assumed. Data-transfer overhead exceeds PIM compute gains is an important negative result that motivates the whole system. |
| D6 Mechanism cost quantification | Each optimization pass benchmarked individually with ablation (Fig. 13, 15, 16, Table 4). Resource pooling overhead (75%), layout adjustment overhead (27%/53%), QKV fraction (17–31%) all quantified. Transfer bandwidth lookup tables and PerfTable construction described precisely. |
| D7 Venue fit (CGO) | Good fit: paper is fundamentally about compiler integration (TVM/Relay), code generation for a novel PIM ISA, and analytical tiling — all classic CGO topics. The LLM application is a driver, not the intellectual core. |
| D8 Self-consistency | Results are internally consistent: Tile-Selector improvements correlate with model size (larger GEMMs → more PIM-beneficial tiling); CPU/PIM cooperation correctly handles small-GEMM case. PIM-LLM is strictly better than all sub-configurations in every model. No contradictions found. |
| D9 Significance / transferable impact | **High for UPMEM MoE/LLM work**: directly the closest methodological precedent for [[moe-upmem-inference]]. Tile-Selector's analytical model (PerfTable + BW-LUT) is portable to other multi-DPU PIM systems lacking shared memory. The four compiler passes are largely architecture-agnostic. 45.75× over TVM on LLaMA-7b is a credible real-hardware result. |

## Connections

- [[processing-in-memory-llm]] — PIM-LLM is the first real UPMEM hardware result for end-to-end LLM inference; fills the "UPMEM-based" gap in the substrate taxonomy table
- [[pim-case-study-atc2021]] — directly extends: that paper characterized UPMEM structural limits (no inter-DPU comms, data-copy overhead, bank-isolation) at the system level; PIM-LLM's design (contiguous-transfer tiling, resource pooling, CPU/PIM cooperation) is a direct engineering response to each limitation
- [[moe-upmem-inference]] — **closest methodological precedent** for our driving UPMEM MoE idea: PGEMMlib's tiling APIs, Tile-Selector's analytical transfer model, and the CPU/PIM cooperation scheduling approach are directly applicable; MoE adds sparse-routing over multiple expert-mapped DPU groups on top of this GEMM substrate
- [[llm-test-time-memory]] — adjacent UPMEM LLM inference idea; shares the UPMEM hardware platform and the memory-bandwidth bottleneck motivation
- [[llm-serving]] — end-to-end LLM serving context; PIM-LLM targets single-batch decode, orthogonal to batched serving systems
- [[on-device-llm-inference]] — 8-bit quantized single-batch decode on UPMEM is effectively on-device/edge inference; shares resource constraint profile
- [[mixture-of-experts]] — PIM-LLM's per-DPU weight partitioning and CPU/PIM cooperation logic are the architectural primitives on which MoE expert routing over UPMEM would be built
- [[compute-in-memory]] · [[in-memory-computing]] · [[memory-centric-computing]] — substrate concepts
- [[cent-asplos2025]] — GDDR-PIM LLM serving; contrasts with UPMEM (DRAM-NB vs GDDR-PIM; CENT is datacenter-batched, PIM-LLM is single-batch)
- [[neupims-asplos2024]] — HBM-PIM LLM serving; contrasts: NeuPIMs uses NPU+HBM-PIM heterogeneity; PIM-LLM uses UPMEM-only with CPU fallback
- [[cambricon-llm-micro2024]] — flash-PIM LLM; different substrate (NAND) and capacity tier
- [[cxl-pnm-lpddr-hpca2024]] — LPDDR5X-PNM; contrasts memory-class PIM form factors
- [[hpim-arxiv2025]] · [[lincoln-hpca2025]] · [[papi-asplos2025]] · [[ianus-asplos2024]] · [[specpim-asplos2024]] — HBM-PIM sibling papers in the PIM-LLM cohort
- [[lp-spec-arxiv2025]] — LPDDR5-PIM speculative decoding; real-hardware single-batch PIM inference, closest platform-credibility peer
- [[repa-kvcache-pim-asplos2026]] — sibling ASPLOS paper: reconfigurable PIM jointly accelerating KV-cache offloading + attention compute; head-partitioned locality + sub-batch GPU/PIM pipeline; ASPLOS 2026.
- [[pim-dl-asplos2024]] — predecessor LUT-NN-on-DRAM-PIM system for DNN inference (BERT/ViT); MI-LLM (the LUT-LLM sibling of PIM-LLM GEMM approach) cites it as methodological predecessor.

Cites (no page yet):
- Gómez-Luna et al. [21] "Benchmarking a Real Processing-in-Memory System" (ACCESS 2022) — prior UPMEM GEMV characterization; the only prior UPMEM compute benchmark in the LLM context
- PIM-GPT [58] (Wu et al., npj Unconventional Computing 2024) — hybrid PIM+ASIC GPT inference; offloads GEMV to PIM, non-GEMV to ASIC
- NeuPIMs [24] / TransPIM [64] / StepStone [11] — cited as related PIM-LLM frameworks using HBM-PIM or modified DRAM
- Das et al. [14] (SOCC 2022) — prior CNN GEMM offload to UPMEM; cited as inefficient due to no tiling
