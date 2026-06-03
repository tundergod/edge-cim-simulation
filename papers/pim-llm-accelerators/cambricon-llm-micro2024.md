---
type: source
title: "Cambricon-LLM: A Chiplet-Based Hybrid Architecture for On-Device Inference of 70B LLM"
created: 2026-05-22
updated: 2026-05-22
raw_path: raw/papers/cambricon-llm-micro2024.pdf
source_kind: paper
ingest_level: full
authors: [Zhongkai Yu, Shengwen Liang, Tianyun Ma, Yunke Cai, Ziyuan Nan, Di Huang, Xinkai Song, Yifan Hao, Jie Zhang, Tian Zhi, Yongwei Zhao, Zidong Du, Xing Hu, Qi Guo, Tianshi Chen]
venue: MICRO
year: 2024
tags: [llm-inference, in-flash-computing, isc, llm-pim, edge-llm, quantization-int8, chiplet, on-device-llm, nand-flash, cambricon]
---

# Cambricon-LLM: A Chiplet-Based Hybrid Architecture for On-Device Inference of 70B LLM

## TL;DR

Cambricon-LLM is a chiplet-based hybrid NPU + dedicated NAND flash architecture targeting **single-batch on-device inference of 70B-class LLMs**. The NAND flash chip is augmented with on-die computation (in-flash GEMV) and an ultra-lightweight on-die ECC unit; the NPU handles compute-bound matrix-matrix work; an optimized tiling strategy minimizes NPU↔flash data movement. The system delivers 3.44 tok/s on 70B (INT8) and 36.34 tok/s on 7B — 22×–45× faster than UFS-flash-offloading baselines and beyond what prior in-storage computing (OptimStore, BeaconGNN) achieves for LLM single-batch reduction ratio.

## Key claims

- **Single-batch LLM decode arithmetic intensity ≈ 2 ops/byte** under INT8 (§I, Fig. 1a) — 30–100× lower than DLRM/BERT/VGG and >100× lower than Apple A16 / A100 / Jetson Orin's roofline knee. This is the headline memory-wall framing for the edge.
- **Flash-offloading is fundamentally bandwidth-bound** (§I): offloading 70B INT8 to UFS 4.0 (~4 GB/s) yields a theoretical max of 0.06 tok/s — orders below the 3–10 tok/s real-time threshold.
- **Reduction ratio of LLM single-batch GEMV ≈ 4096:1** (input weight matrix to output vector, §II.B Fig. 1b) — 100× larger than the operators OptimStore/BeaconGNN/GenStore/RecSSD were designed for, so prior on-die ISC achieves <10% flash-channel utilization on LLM workloads.
- **Cambricon-LLM throughput**: 3.44 tok/s for 70B INT8 single-batch, 36.34 tok/s for 7B (§I abstract).
- **On-die ECC is essential** (§I, §II): without protection, flash bit-errors degrade LLM accuracy by >70%. The proposed lightweight on-die ECC targets outlier weights specifically.
- **Validation via SSDsim simulator** (§I) — channel + chip configurations swept; physical chiplet not taped out.

## Motivation

Personal/on-device LLM agents demand both huge capacity (70B INT8 = 70 GB) and real-time decode (3–10 tok/s). Smartphone DRAM is far short of the capacity, and offloading to commodity flash (UFS 4.0 at ~4 GB/s) cannot deliver the required bandwidth. Existing on-die in-storage compute prototypes were designed for higher-arithmetic-intensity workloads (graph mining, DLRM embeddings) and cannot harness flash's internal parallelism for the extreme reduction ratio of LLM GEMV.

## Method

- **Chiplet topology**: NPU die + dedicated NAND flash die connected by die-to-die (D2D) link, bypassing UFS 4.0's interface bandwidth ceiling.
- **In-flash on-die compute**: GEMV and matrix-vector reduction primitives execute inside the flash die, fully exposing inter-channel parallelism.
- **Tiling strategy**: distributes the LLM inference workload between NPU and flash to balance compute and bandwidth — flash does the bandwidth-heavy decode GEMV, NPU does matrix-matrix prefill and special functions.
- **Ultra-lightweight on-die ECC**: protects the outlier weights whose bit-errors disproportionately damage LLM accuracy.

## Results

- 70B INT8: 3.44 tok/s; 7B INT8: 36.34 tok/s.
- 22×–45× over UFS-flash-offloading baselines.
- LLM accuracy preserved (no >70% accuracy collapse seen in unprotected flash) thanks to on-die ECC.

## Contributions

1. First chiplet-based NPU + ISC-flash hybrid targeting single-batch on-device LLM.
2. Hardware-aware tiling strategy that exposes flash's internal parallelism while minimizing D2D traffic.
3. On-die ECC algorithm + lightweight unit specifically for LLM-outlier weight protection.

## Limitations / open questions

- Simulator-only (SSDsim); no taped-out chiplet measurement — D4 vulnerability.
- Single-batch only; batched serving would change the GEMV-vs-GEMM balance and may shift the NPU/flash partition.
- INT8 only — sub-byte (INT4 etc.) datapath in flash is not explored; would amplify the reduction ratio further.
- 70B model coverage is the headline but the open-question is whether smaller models (1B–13B) still benefit when DRAM alone can hold them.

## D1–D9 review lens

| # | Dimension | Reading |
|---|---|---|
| D1 | Baselines | Strong vs flash-offloading and prior on-die ISC (OptimStore, BeaconGNN); LPDDR-CXL-PNM or HBM-PIM (server-class) cited as adjacent but not directly compared. |
| D2 | Novelty | Clear delta: prior ISC tackled lower reduction ratios and assumed reliable flash; combining chiplet topology + tiling + ECC is non-trivial. |
| D3 | Evaluation | Two model sizes (7B, 70B); INT8 only; broader Llama-2/3 family or MoE coverage would strengthen. |
| D4 | Platform | Simulator-only. D4 is the dominant attack surface. |
| D5 | Motivation | Compelling — personal-LLM memory wall is a real and well-framed problem. |
| D6 | Mechanism cost | ECC and D2D overheads quantified; on-die compute area discussed via SSDsim, not silicon. |
| D7 | Venue | MICRO-natural; chip-architecture audience. |
| D8 | Consistency | Internal numbers track between abstract, §I, and §II. |
| D9 | Significance | High — answers "what does an LLM-flash-PIM architecture look like" as a reference point. |

## Connections

**Surprises / observations from Phase 2**

- The 4096:1 reduction ratio of LLM single-batch GEMV is presented as a **disadvantage** for prior on-die ISC (OptimStore, BeaconGNN cannot use it well) but is structurally *the same property* that enables flash-PIM efficiency — channel-parallel weight load + per-die reduction. The framing inverts how the ISC community has historically read LLM workloads.
- The 70 GB capacity argument is the **clearest articulation** in the vault's PIM-LLM cohort of *why DRAM-NB designs (NeuPIMs, IANUS) cannot reach 70B on-device* — they need ~70 GB of HBM-PIM, which doesn't fit a phone. This justifies the flash substrate without ever saying "flash beats DRAM."

**Concepts / entities / projects / ideas**

- [[processing-in-memory-llm]] · [[on-device-llm-inference]] · [[llm-weight-quantization]] (uses INT8; sub-byte open) · [[in-storage-computing]] · [[computational-storage]] · [[memory-centric-computing]]
- [[metis-llm-investigation-desktop-2026-05-19]] — Cambricon-LLM is the alternative substrate for the same 70B problem that's bandwidth-wall-limited on Metis at 24 GB/s; Cambricon-LLM achieves 3.44 tok/s via flash channel parallelism.
- [[metis-aipu-nn-v2-2026-05-21]] — appendix M entry #30 (bold, "core comparable"); single most-cited LLM-PIM SoC.
- [[cnn-dnn-edge-memory-wall-metis-embedded]] — appendix LLM cross-reference: this is the literature backdrop for the LLM-closure claim.
- [[metis-cxl-cim-memory-system]] — flash tier of the four-tier CIM memory hierarchy is what this paper proposes.
- [[llm-image-generation-mobile]] — edge / mobile LLM serving substrate alternative.
- [[neupims-asplos2024]] · [[ianus-asplos2024]] · [[cent-asplos2025]] — sibling LLM-PIM substrates (HBM, unified-memory NPU+PIM, CXL+GDDR-PIM).
- [[hpim-arxiv2025]] — related heterogeneous PIM design (SRAM-PIM + HBM-PIM vs Cambricon-LLM's NPU + flash-PIM).
- [[lincoln-hpca2025]] — closest sibling: same 70B+ flash-PIM problem, but LPDDR-interface (not custom D2D chiplet); enables consumer-SoC deployment.
- [[awq-lin-2024]] · [[gptq-frantar-2023]] — sub-byte weight quantization would extend Cambricon-LLM's reduction-ratio advantage further (open follow-up).
- [[mi-llm-multiplier-free-pim-tc2026]] — MI-LLM is the closest real-hardware sibling: multiplier-free LUT inference on UPMEM near-bank PIM vs Cambricon-LLM's chiplet NPU+NAND flash; both target real-hardware on-device LLM at 70B scale; IEEE TC 2026.
- [[sieve-moe-pim-arxiv2026]] — Sieve extends the memory-centric LLM substrate idea to MoE with dynamic GPU/PIM expert partitioning; Cambricon-LLM's single-batch FFN offload framing is the non-MoE counterpart.
- [[pimphony-lolpim-longcontext-hpca2026]] — PIMphony/LoL-PIM targets long-context KV decode on DRAM-PIM; the long-context capacity challenge Cambricon-LLM faces with NAND flash is the same problem PIMphony solves at the DRAM tier; HPCA 2026.
- [[pim-dl-asplos2024]] — sibling LUT-NN-on-DRAM-PIM paper (UPMEM): applies LUT-based DNN inference to the same commodity near-memory compute substrate; DNN-scoped method that MI-LLM extends to LLMs; ASPLOS 2024.
