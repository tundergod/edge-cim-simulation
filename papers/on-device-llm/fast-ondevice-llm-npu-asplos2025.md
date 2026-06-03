---
type: source
title: "Fast On-Device LLM Inference with NPU Acceleration"
created: 2026-05-18
updated: 2026-05-18
tags: [on-device-llm-inference, npu, mobile-inference, edge-ai, asplos]
raw_path: raw/papers/fast-ondevice-llm-npu-asplos2025.pdf
source_kind: paper
ingest_level: full
authors: [Daliang Xu, et al.]
venue: ASPLOS
year: 2025
---

# Fast On-Device LLM Inference with NPU Acceleration

## TL;DR

The first LLM inference system that efficiently leverages the mobile NPU for acceleration on commercial off-the-shelf (COTS) smartphones. Three co-designed techniques address NPU compilation constraints and scheduling inefficiencies: chunk-sharing computation graphs eliminate dynamic-shape overhead, shadow outlier execution offloads sparse outlier channels to CPU/GPU, and out-of-order subgraph execution hides NPU memory-copy stalls. Achieves 22.4× prefill speedup and 30.7× energy savings versus CPU-only baseline; >1000 tokens/sec prefill for billion-parameter models.

## Key Claims

- Mobile NPUs are highly energy-efficient but prior LLM inference systems ignore them, because NPUs require static computation graphs and can't handle dynamic shapes (variable sequence lengths), sparse outlier channels, or sequential subgraph dependencies.
- **Chunk-sharing graphs**: split prompt into fixed-size chunks; subgraphs for different chunks share compiled NPU kernels — eliminating per-sequence recompilation without sacrificing dynamic-length handling.
- **Shadow outlier execution**: LLMs have sparse "outlier" channels with extreme activation magnitudes that break NPU quantization. A small "shadow" model handles only outlier channels on CPU/GPU; the NPU handles the remaining dense channels. The two paths execute in parallel and their results are combined.
- **Out-of-order subgraph execution**: NPU subgraph scheduling has hidden memory-copy overhead between NPU and CPU memory. Reorder subgraph execution to overlap copy with compute, reducing stall time.

## Method

The three techniques compose into a unified NPU+CPU/GPU heterogeneous execution runtime. The chunk-sharing graph compiler produces static NPU graphs reusable across positions within a chunk. Shadow outlier execution is a channel-level decomposition: outlier detection offline → shadow model construction → runtime output merging. OoO scheduling inserts explicit memory-copy operations and reorders them to maximize overlap.

Target platform: commercial Android smartphones with COTS NPUs (snapdragon-class, based on context).

## Results

- **22.4× prefill speedup** and **30.7× energy savings** vs CPU-only baseline.
- **>1000 tokens/sec prefill** for billion-scale models on COTS NPU.
- Baselines include CPU-only llama.cpp and GPU-assisted approaches.

## Contributions

1. First full LLM inference system exploiting mobile NPU; demonstrates that COTS NPUs are viable for on-device LLM prefill at scale.
2. Chunk-sharing computation graph for NPU-compatible dynamic-length inference.
3. Shadow outlier execution for outlier-channel accuracy preservation.
4. Out-of-order subgraph execution for NPU stall elimination.

## Limitations / Open Questions

- Decode phase speedup not quantified separately (NPUs are less effective for single-token decode).
- Outlier channel detection and shadow model construction require offline profiling; model-to-model portability not analyzed.
- Evaluation limited to prefill; long-context decoding performance needs further study.
- NPU-side KV cache management not addressed; contrast with [[kvswap-ondevice-2025]] (disk offloading).

## D1–D9 Review Lens

| Dim | Assessment |
|-----|-----------|
| **D1** | Adequate — CPU-only and GPU-assisted baselines; llama.cpp included. Missing: comparison with [[powerinfer2-smartphone-2024]] (neuron-cluster approach) and [[edgemoe-2023]]. |
| **D2** | Strong — NPU utilization for LLM inference is a genuinely novel direction; each technique addresses a specific NPU constraint clearly. |
| **D3** | Limited in available pages — prefill-focused; decode throughput and long-context KV behavior not covered. |
| **D4** | Strong — real COTS smartphone NPU hardware. |
| **D5** | Strong — mobile NPU energy efficiency vs CPU is well-characterized (30.7× energy gap); commercial deployment motivation is clear. |
| **D6** | Partial — shadow model overhead (channel count, extra CPU compute) not quantified in visible pages. |
| **D7** | ASPLOS 2025 — computer architecture + programming systems. NPU system co-design is a good fit. |
| **D8** | Clear diagram-based presentation of three techniques. |
| **D9** | High — mobile NPU unlocks a new tier of on-device LLM acceleration; 1000+ tokens/sec prefill is a meaningful threshold for production usability. |

## Connections

- [[on-device-llm-inference]] — core contribution to on-device LLM acceleration.
- [[edge-ai]] — mobile-first LLM deployment.
- [[powerinfer2-smartphone-2024]] — complementary: neuron-cluster abstraction for CPU+NPU split vs this paper's chunk-sharing + shadow outlier approach.
- [[edgemoe-2023]] — EdgeMoE for MoE LLMs on edge; both address heterogeneous mobile hardware.
- [[kvswap-ondevice-2025]] — addresses KV cache offloading to disk for long-context on-device inference; complements this paper's prefill acceleration.
- [[llm-in-a-flash-apple-2023]] — LLM in Flash: flash-based inference; addresses the storage/memory boundary rather than NPU execution.
- [[llm-image-generation-mobile]] — research idea for edge LLM inference; NPU acceleration is a key tool.
- [[characterizing-mobile-soc-heterogeneous-llm-inference-sosp2025]] — HeteroInfer (SOSP'25) is a competing on-device engine from the same cycle; targets the GPU+NPU joint-execution gap that NPU-only engines like this one leave open.
