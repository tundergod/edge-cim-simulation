---
type: concept
title: "Processing-in-Memory for LLM Inference"
created: 2026-05-22
updated: 2026-05-22
tags: [llm-pim, pim, processing-in-memory, dram-nb-pim, hbm-pim, gddr-pim, flash-pim, cxl-pim, memory-wall]
parents: [memory-centric-computing, in-memory-computing]
aliases: [llm-pim, pim-llm, llm-on-pim]
---

# Processing-in-Memory for LLM Inference

LLM inference is **fundamentally bandwidth-limited** in the decode phase — arithmetic intensity is ≈2 ops/byte for INT8 single-batch, two orders of magnitude below the roofline knee of any commodity accelerator (Cambricon-LLM Fig. 1a, [[metis-llm-investigation-desktop-2026-05-19]] §B). PIM moves compute *into* the memory substrate, exposing the much higher *internal* memory bandwidth (~10×–100× of the external interface) for the bandwidth-bound GEMV operations that dominate decode.

## Substrate taxonomy

| Substrate | Internal BW | Capacity | Latency | Examples in vault |
|-----------|-------------|----------|---------|--------------------|
| **HBM-PIM (DRAM-NB)** | ~5–16 TB/s/stack | ~tens of GB | sub-µs | [[neupims-asplos2024]], [[ianus-asplos2024]], [[specpim-asplos2024]], [[papi-asplos2025]], [[hpim-arxiv2025]] |
| **GDDR-PIM (e.g., SK hynix AiM)** | ~16 TB/s | ~tens of GB | sub-µs | [[cent-asplos2025]] |
| **CXL-attached PIM** | internal × stack + CXL fabric | TB-scale | µs–10s of µs | [[cent-asplos2025]] (GDDR-PIM), [[cxl-pnm-lpddr-hpca2024]] (LPDDR5X-PNM) |
| **DIMM-PIM (DDR4 DIMM form factor)** | ~13 TB/s aggregate (16ch × 2 DIMM) | TB-scale (2 TB in L3 eval config) | PCIe-bound host↔GPU | [[l3-dimm-pim-longcontext-arxiv2025]] (MHA-decode offload; simulation); structurally analogous to UPMEM |
| **LPDDR-PIM (mobile)** | LPDDR-internal per-bank | tens of GB | ns–µs | [[lp-spec-arxiv2025]] (LPDDR5-PIM + GEMM aug, mobile speculative) |
| **Flash / NAND in-storage compute** | per-die GB/s × parallel dies | TB-scale | ms-class | [[cambricon-llm-micro2024]], [[lincoln-hpca2025]] (LPDDR-interfaced) |
| **SRAM-CIM** | ~TB/s on-die | ~tens of MB | ns | [[system-axelera-metis-card]] (vendor-closed for LLM), [[hpim-arxiv2025]] (SRAM-PIM-for-attention component of heterogeneous design) |

## Design dimensions

| Dimension | Choices |
|-----------|---------|
| **Workload partition** | All-PIM (CENT) · NPU(GEMM) + PIM(GEMV) (NeuPIMs, IANUS, AttAcc) · NPU + ISC-flash (Cambricon-LLM) |
| **Memory organization** | Partitioned (most early work) · Unified NPU+PIM memory (IANUS) |
| **PIM concurrency** | Blocked mode (commercial AiM, HBM-PIM) · Dual row buffer (NeuPIMs) · PIM Access Scheduling (IANUS) |
| **Fabric** | On-package D2D (Cambricon-LLM) · HBM stack (NeuPIMs, IANUS, AttAcc) · CXL (CENT) |
| **Substrate** | DRAM-NB · DRAM-NM (PNM) · Flash · SRAM |
| **Target serving regime** | Single-batch / edge (Cambricon-LLM, CENT) · Batched / data-center (NeuPIMs, IANUS, AttAcc) |

## Recurring barriers in the literature

1. **Blocked-mode PIM**: commercial AiM/HBM-PIM cannot do PIM ops and normal accesses concurrently → NPU↔PIM serialization. NeuPIMs and IANUS both attack this (dual row buffer + PAS).
2. **Memory duplication**: partitioned-memory designs duplicate the ~90% shared parameters between NPU and PIM (IANUS §1).
3. **GEMV ≠ GEMM**: PIM is bandwidth-rich and compute-poor — great for decode MHA, weak for prefill GEMM. Hybrid NPU+PIM is the dominant 2024 design pattern.
4. **D4 platform credibility**: nearly all 2024–2025 LLM-PIM papers are simulator-only. IANUS's FPGA + commercial PIM + commercial NPU prototype is the strongest D4 in this cohort.
5. **Sub-byte support**: most academic prototypes are INT8; INT4/INT3 datapaths in PIM are open. See [[llm-weight-quantization]].

## Cross-platform reference: Axelera Metis (digital SRAM-CIM)

Axelera Metis is a *commercial digital SRAM-CIM* card. Its measured LLM-decode bandwidth is 24.23 GB/s on the production-class card ([[metis-llm-investigation-desktop-2026-05-19]] §B) — orders below the HBM-PIM / GDDR-PIM / Flash-PIM internal bandwidths in this cohort. This is partly *not a fundamental SRAM-CIM limitation* (on-die SRAM is multi-TB/s) but rather a result of Metis weights living in on-card LPDDR4, not in the SRAM-CIM array, plus the closed precompile pipeline forbidding sub-byte datapaths and dynamic shapes. The lesson the vault carries forward (see [[cnn-dnn-edge-memory-wall-metis-embedded]]) is that **CNN/DNN — not LLM — is the workload class natural to commercial digital SRAM-CIM at the edge form factor.**

## Connections

[[memory-centric-computing]] · [[in-memory-computing]] · [[compute-in-memory]] · [[computational-memory-hierarchy]] · [[neupims-asplos2024]] · [[ianus-asplos2024]] · [[cent-asplos2025]] · [[specpim-asplos2024]] · [[papi-asplos2025]] · [[hpim-arxiv2025]] · [[lp-spec-arxiv2025]] · [[cxl-pnm-lpddr-hpca2024]] · [[lincoln-hpca2025]] · [[cambricon-llm-micro2024]] · [[llm-weight-quantization]] · [[speculative-decoding]] · [[llm-serving]] · [[metis-llm-investigation-desktop-2026-05-19]] · [[metis-aipu-nn-v2-2026-05-21]] · [[metis-cxl-cim-memory-system]] · [[cxl-pim-storage-vs-memory-upmem]] · [[long-context-llm-cxl-optimization]] · [[YYYY-dac-cxl-pim]] · [[pim-case-study-atc2021]] · [[pim-llm-pgemmlib-cgo2025]] · [[l3-dimm-pim-longcontext-arxiv2025]] (DDR4 DIMM-PIM substrate; MHA-decode offload + in-flight re-layout + adaptive sub-batch scheduling; simulation, arXiv 2025) · [[mi-llm-multiplier-free-pim-tc2026]] (multiplier-free LUT LLM inference on real UPMEM near-bank PIM; IEEE TC 2026) · [[sieve-moe-pim-arxiv2026]] (dynamic token-distribution-aware GPU/PIM expert partitioning for MoE on HBM-PIM) · [[duplex-moe-pim-isca2024]] (Op/B-based hot/cold expert split across xPU + Logic-PIM; ISCA 2024) · [[context-aware-moe-cxl-ndp-arxiv2025]] (prefill-routing oracle for one-shot expert placement on CXL-NDP) · [[pimphony-lolpim-longcontext-hpca2026]] (dynamic KV-cache mgmt + token-centric partitioning for long-context decode on DRAM-PIM; HPCA 2026) · [[starc-sparse-attention-pim-arxiv2025]] (sparsity-aware KV clustering/remapping aligned to PIM row granularity) · [[repa-kvcache-pim-asplos2026]] (reconfigurable PIM jointly accelerating KV-cache offloading + attention compute; head-partitioned locality + sub-batch GPU/PIM pipeline; ASPLOS 2026, simulation) · [[pim-dl-asplos2024]] (LUT-NN on real commodity DRAM-PIM via eLUT-NN + Auto-Tuner; DNN-scoped foundation of the LUT-on-PIM thread; ASPLOS 2024) · [[op-b-aware-scheduling]] (arithmetic-intensity Op/B device selection: high-Op/B → compute-rich processor, low-Op/B → near-memory PIM) · [[dynamic-pim-memory-management]] (runtime VA→PA translation + lazy allocation on PIM to break the static-address limit for KV-cache) · [[sparsity-aware-kv-remapping]] (KV-cache layout matched to PIM row granularity so sparse-attention access becomes row-aligned block access)
