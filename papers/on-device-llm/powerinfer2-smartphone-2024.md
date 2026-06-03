---
type: source
title: "PowerInfer-2: Fast Large Language Model Inference on a Smartphone"
created: 2026-05-18
updated: 2026-05-18
tags: [on-device-llm-inference, mobile-inference, sparsity, npu, smartphone, edge-ai]
raw_path: raw/papers/powerinfer2-smartphone-2024.pdf
source_kind: paper
ingest_level: full
authors: [Xue, Song, Mi, et al.]
venue: arXiv
year: 2024
---

# PowerInfer-2: Fast Large Language Model Inference on a Smartphone

## TL;DR

PowerInfer-2 enables billion-parameter LLM inference on commodity smartphones. The core abstraction is a *neuron cluster*: dense clusters of frequently co-activated neurons are mapped to the NPU (which excels at dense computation), while sparse clusters (infrequently activated) remain on the CPU. I/O-Aware Orchestration pipelines cluster-level computation with storage I/O. Sparsity-Aware Adaptation dynamically adjusts the NPU/CPU allocation based on batch size. Achieves 27.8× speedup over llama.cpp and runs a 47B LLM on smartphone for the first time (11.68 tok/s for TurboSparse-Mixtral-47B on OnePlus 12).

## Key Claims

- Smartphone NPUs have high TOPS but require dense, regular computation; CPU handles sparse/irregular but is slow. Neuron-cluster abstraction bridges the gap.
- **Neuron clusters**: group neurons by co-activation frequency. Dense clusters (consistently activated together) → NPU; sparse clusters (activated intermittently) → CPU.
- **I/O-Aware Orchestration**: pipeline cluster-level computation with storage I/O (reading weight blocks from flash/UFS). Schedule next-cluster I/O during current-cluster compute.
- **Sparsity-Aware Adaptation**: dynamically adjusts NPU vs CPU allocation per batch size — larger batches have higher activation density, shifting more clusters to NPU.
- MoE LLMs (TurboSparse-Mixtral-47B) are a natural target: experts = natural cluster boundaries.

## Method

**Profiling phase**: run representative prompts to profile neuron co-activation patterns; cluster neurons hierarchically by activation correlation.

**Runtime**: decode step → predict which clusters activate (using a small predictor or MoE router) → schedule dense clusters on NPU + sparse on CPU → overlap with storage I/O for next-token's clusters.

**Sparsity-Aware Adaptation**: as batch size increases, the effective density of activations increases; reassign borderline clusters from CPU to NPU.

Evaluated on OnePlus 12 (Snapdragon 8 Gen 3 with NPU) and other flagship Android devices.

## Results

- **27.8× speedup** over llama.cpp on the same smartphone.
- First smartphone to run **47B LLM** (TurboSparse-Mixtral-47B): **11.68 tokens/sec** on OnePlus 12.
- Significant speedup over llama.cpp across 7B, 13B, and 47B models.

## Contributions

1. Neuron cluster abstraction that maps LLM sparsity to smartphone hardware (NPU vs CPU).
2. I/O-Aware Orchestration: overlapping cluster compute with storage I/O.
3. Sparsity-Aware Adaptation: dynamic NPU/CPU allocation based on runtime activation density.
4. First demonstration of a 47B-parameter LLM running on a smartphone.

## Limitations / Open Questions

- Neuron cluster profiling requires offline characterization; new models or tasks may need re-profiling.
- Cluster-to-hardware mapping is static at profile time; runtime distribution shifts could degrade performance.
- 11.68 tok/s for 47B is a throughput milestone but not production-quality for latency-sensitive applications.
- No energy or battery life analysis.

## D1–D9 Review Lens

| Dim | Assessment |
|-----|-----------|
| **D1** | Solid — llama.cpp is the de facto smartphone LLM baseline; OnePlus 12 hardware used consistently. |
| **D2** | Clear — neuron cluster + NPU/CPU heterogeneity is a principled decomposition; distinguishes from [[fast-ondevice-llm-npu-asplos2025]] (chunk-sharing + shadow outlier). |
| **D3** | Good — multiple model sizes (7B/13B/47B), real device evaluation, speedup metrics. Missing: accuracy preservation, long-context, energy. |
| **D4** | Excellent — real smartphone hardware (OnePlus 12, Snapdragon 8 Gen 3). |
| **D5** | Strong — smartphone LLM deployment at >10B scale is a real deployment barrier; 47B as the target sets a concrete milestone. |
| **D6** | Partial — storage I/O overlap cost and NPU/CPU split overhead not separately measured. |
| **D7** | arXiv preprint; targets MobiSys/ASPLOS/MLSys. |
| **D8** | Clear hardware platform specification; 27.8× headline number is traceable. |
| **D9** | Very high — 47B on smartphone is a landmark; practical implication is local execution of frontier-tier models without cloud. |

## Connections

- [[on-device-llm-inference]] — smartphone-scale LLM inference; compare with [[fast-ondevice-llm-npu-asplos2025]] and [[edgemoe-2023]].
- [[edge-ai]] — mobile/consumer device AI.
- [[fast-ondevice-llm-npu-asplos2025]] — NPU utilization for on-device LLM; different technique (chunk-sharing + shadow) vs neuron cluster.
- [[edgemoe-2023]] — MoE expert management on edge; PowerInfer-2 addresses both dense and MoE LLMs.
- [[llm-in-a-flash-apple-2023]] — flash-based weight loading; complementary (Apple Silicon vs Snapdragon).
- [[solid-state-drives]] — UFS/flash as model weight store on smartphone.
- [[llm-image-generation-mobile]] — research idea for mobile LLM; PowerInfer-2 is the key prior work for MoE on smartphone.
- [[characterizing-mobile-soc-heterogeneous-llm-inference-sosp2025]] — HeteroInfer (SOSP'25) reports 1.32× over PowerInfer-2 while preserving FLOAT accuracy (PowerInfer-2's sparse computation degrades memory access patterns).
