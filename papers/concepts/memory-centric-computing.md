---
type: concept
title: "Memory-centric Computing"
created: 2026-05-07
updated: 2026-05-07
tags: [memory-systems, post-von-neumann]
aliases: [memory centric computing]
---

Memory-centric computing shifts system design around data placement and memory-side execution, addressing the memory wall by reducing movement between processors, memory, and storage.

Related proposals: [[18-21-design-and-optimization-of-non-volatile-one-memory-architecture-nstc]], [[26-29-enabling-high-performance-data-centric-computing-on-skyrmionic-racetrack-memories-nstc-bmbf]], [[26-29-integrating-computational-memory-hierarchies-for-efficient-memory-centric-computing-nstc-bmbf]], [[26-29-agrimind-efficient-multimodal-learning-for-interactive-intelligent-agricultural-deployment-nstc-bmbf]].

## LLM-specific memory-centric architectures

LLM decode is bandwidth-limited at AI ≈ 2 ops/byte. The PIM/CIM literature has produced a rich design space for memory-centric LLM inference; see [[processing-in-memory-llm]] for the substrate taxonomy. Key 2024–2025 papers in the vault:

- **[[cambricon-llm-micro2024]]** — Chiplet NPU + in-flash compute for on-device 70B (MICRO 2024).
- **[[neupims-asplos2024]]** — NPU + HBM-PIM with dual row buffer + sub-batch interleaving (ASPLOS 2024).
- **[[ianus-asplos2024]]** — Unified NPU+PIM memory + PIM Access Scheduling; FPGA proof-of-concept (ASPLOS 2024).
- **[[cent-asplos2025]]** — GPU-free CXL-fabric all-PIM LLM serving (ASPLOS 2025).
- **[[cxl-pnm-lpddr-hpca2024]]** — LPDDR5X-based CXL-PNM appliance: 512 GB / 1.1 TB/s; beats 8-GPU on latency/throughput/energy/cost (HPCA 2024).
- **[[papi-asplos2025]]** — Heterogeneous GPU+PIM (FC-PIM HBM + Attn-PIM PCIe/CXL) with online dynamic kernel scheduling; 1.8× over A100+AttAcc (ASPLOS 2025).
- **[[hpim-arxiv2025]]** — Heterogeneous SRAM-PIM + HBM-PIM with intra-token pipeline overlap; 22.8×/1.50×/5.76× vs A100/IANUS/CXL-PNM (arXiv 2025, simulator-only).
- **[[lincoln-hpca2025]]** — LPDDR-interfaced compute-enabled flash for real-time 50–100B LLM on consumer device (HPCA 2025).
- **[[lp-spec-arxiv2025]]** — LPDDR5-PIM + GEMM augmentation for mobile speculative LLM inference (arXiv 2025).
- **[[titans-google-2025]]** — three-tier memory framing (short-term attention / long-term neural memory / persistent) reopens memory co-design beyond KV cache; test-time memory writes are a candidate substrate for near-memory acceleration ([[llm-test-time-memory]]).

These map directly onto the 9-paper "core comparable" bold set of the 71-paper PIM/CIM heatmap in [[metis-aipu-nn-v2-2026-05-21]] §5 Appendix M (#30 Cambricon-LLM, #34 NeuPIMs, #36 IANUS, #62 CENT). The vault carries forward the conclusion that **commercial digital SRAM-CIM (Axelera Metis) sits in a different memory-wall regime** — LPDDR4-bound at 24 GB/s ([[metis-llm-investigation-desktop-2026-05-19]]) — and is not viable for LLM-architecture research without vendor cooperation. See [[cnn-dnn-edge-memory-wall-metis-embedded]] for the reframing.

## Connections

[[in-memory-computing]] · [[compute-in-memory]] · [[computational-memory-hierarchy]] · [[processing-in-memory-llm]] · [[computational-storage]] · [[on-device-llm-inference]] · [[metis-aipu-nn-v2-2026-05-21]] · [[metis-cxl-cim-memory-system]] · [[characterizing-mobile-soc-heterogeneous-llm-inference-sosp2025]] · [[titans-google-2025]] · [[llm-test-time-memory]] · [[hpim-arxiv2025]] · [[papi-asplos2025]] · [[cxl-pnm-lpddr-hpca2024]] · [[lincoln-hpca2025]] · [[lp-spec-arxiv2025]] · [[l3-dimm-pim-longcontext-arxiv2025]] (DDR4 DIMM-PIM; plug-and-play modular capacity + bandwidth scaling for long-context LLM; simulation, arXiv 2025) · [[mi-llm-multiplier-free-pim-tc2026]] (multiplier-free LUT LLM inference on real UPMEM near-bank PIM; real-hardware baseline; IEEE TC 2026) · [[sieve-moe-pim-arxiv2026]] (dynamic token-distribution-aware GPU/PIM expert partitioning for MoE; HBM-PIM simulation) · [[duplex-moe-pim-isca2024]] (Op/B-based hot/cold expert split across xPU + Logic-PIM; ISCA 2024, simulation) · [[context-aware-moe-cxl-ndp-arxiv2025]] (prefill-routing oracle for one-shot expert placement on CXL-NDP; simulation) · [[pimphony-lolpim-longcontext-hpca2026]] (dynamic KV-cache mgmt + token-centric partitioning for long-context decode on DRAM-PIM; HPCA 2026, simulation) · [[starc-sparse-attention-pim-arxiv2025]] (sparsity-aware KV clustering/remapping aligned to PIM row granularity; simulation) · [[repa-kvcache-pim-asplos2026]] (reconfigurable PIM jointly accelerating KV-cache offloading + attention compute; head-partitioned locality + sub-batch GPU/PIM pipeline; ASPLOS 2026, simulation) · [[pim-dl-asplos2024]] (LUT-NN on real commodity DRAM-PIM via eLUT-NN + Auto-Tuner; DNN-scoped foundation of the LUT-on-PIM thread; ASPLOS 2024)
