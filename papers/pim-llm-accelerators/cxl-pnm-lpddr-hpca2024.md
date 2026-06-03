---
type: source
title: "An LPDDR-based CXL-PNM Platform for TCO-efficient Inference of Transformer-based Large Language Models"
created: 2026-05-25
updated: 2026-05-25
raw_path: https://ieeexplore.ieee.org/document/10476443/
source_kind: paper
ingest_level: weak
authors: [Sang Soo Park, Kyungsoo Kim, Jinin So, Jin Jung, Jonggeon Lee, Kyoungwan Woo, Nayeon Kim, Younghyun Lee, Hyungyo Kim, Yongsuk Kwon, Jinhyun Kim, Jieun Lee, Yeongon Cho, Yongmin Tai, Jeonghyeon Cho, Hoyoung Song, Jung Ho Ahn, Nam Sung Kim]
venue: HPCA
year: 2024
tags: [cxl, pnm, lpddr, llm-inference, transformer, tco, memory-bandwidth, near-memory-processing, gpu-free, datacenter]
---

# An LPDDR-based CXL-PNM Platform for TCO-efficient Inference of Transformer-based Large Language Models

## TL;DR

This paper presents a CXL-PNM (Processing Near Memory) appliance built on LPDDR5X memory that attacks the GPU memory-wall and TCO problem for transformer LLM inference at datacenter scale. The key substrate insight is that LPDDR5X under a module form factor delivers 512 GB capacity and 1.1 TB/s bandwidth — 16× more capacity and 10× more bandwidth than GDDR6/DDR5 CXL alternatives — making it a competitive near-memory compute substrate without GPU-class cost. An 8-device CXL-PNM appliance beats an 8-GPU appliance by 23% latency, 31% throughput, and 2.8× energy efficiency at 30% lower hardware cost.

## Key claims

- **Memory bottleneck framing** (abstract/§1): GPT-3.5-class models require >300 GB and ~1.4 TFLOPs for inference — exceeding single-GPU memory capacity and forcing expensive multi-GPU deployments for a workload whose intensity (~2 ops/byte at decode) is memory-bound, not compute-bound.
- **LPDDR5X CXL substrate advantage** (§2): 512 GB / 1.1 TB/s per module under the same form-factor constraint as GDDR6 and DDR5 CXL modules — 16× capacity gain, 10× bandwidth gain vs. those alternatives.
- **CXL-PNM controller** (§3): Integrated CXL controller + LLM inference accelerator on-module; avoids HBM-PIM and AxDIMM limitations (limited capacity and proprietary interfaces respectively).
- **Software stack** (§4): Python-transparent CXL-PNM runtime; LLM programs require no source changes to use PNM offload.
- **System-level TCO result** (§5): 8-device CXL-PNM appliance vs. 8-GPU appliance — 23% lower latency, 31% higher throughput, 2.8× higher energy efficiency, 30% lower hardware cost.

## Why it might matter

This is a top-venue (HPCA 2024) real-silicon-credible alternative to GPU-only datacenter LLM serving that uses LPDDR5X as the near-memory substrate — directly relevant to the vault's CXL-PIM/PNM design space. For the planned **CIM-on-mobile-SoC simulator work**, this paper establishes that LPDDR-class substrates can be positioned as CXL-attached near-memory compute platforms at datacenter scale, giving a framing contrast: our work is the *edge / embedded* analog (LPDDR4 on Metis SoC, no CXL) while CXL-PNM is the *datacenter* analog (LPDDR5X over CXL). The claim that LPDDR5X out-bandwidth GDDR6/DDR5 CXL modules by 10× is a concrete substrate justification that a CIM simulator paper targeting LPDDR-class platforms should cite and position against.

The comparison with [[cent-asplos2025]] (GDDR6-PIM, GPU-free, CXL fabric, ASPLOS 2025) is direct: both papers argue GPUs are wrong for memory-bound LLM decode; CXL-PNM uses LPDDR5X near-memory while CENT uses GDDR6-PIM. The HPCA 2024 date makes CXL-PNM the earlier sibling — CENT may be partly a response or parallel design point.

**relevance: high** — the LPDDR substrate angle and CXL-PNM framing are unique in the vault; no other source covers LPDDR5X as a CXL near-memory compute platform.

## Connections

- [[processing-in-memory-llm]] — CXL-PNM is a PNM instantiation using LPDDR5X as the memory substrate; contributes a distinct design point (LPDDR5X vs. HBM-PIM vs. GDDR6-PIM) to the substrate taxonomy.
- [[memory-centric-computing]] — squarely in the memory-centric LLM inference cluster; the 10× BW/16× capacity argument is a direct memory-wall argument.
- [[llm-serving]] — TCO-focused serving system paper; 23%/31%/2.8× numbers are LLM serving metrics.
- [[computational-memory-hierarchy]] — the CXL tier with near-memory compute is a computational memory hierarchy instantiation.
- [[cent-asplos2025]] — closest sibling: also GPU-free, also CXL-based, also TCO-motivated; CENT uses GDDR6-PIM while this paper uses LPDDR5X-PNM. Compare bandwidth arguments: CENT claims ~16 TB/s internal GDDR6-PIM BW vs. this paper's 1.1 TB/s external LPDDR5X module BW — different measurement points, both vs. baseline GPU.
- [[neupims-asplos2024]] — alternative substrate (HBM-PIM + NPU), same decode-is-memory-bound thesis; differs by augmenting GPU/NPU rather than replacing it.
- [[ianus-asplos2024]] — also ASPLOS 2024, also NPU+HBM-PIM; three papers (NeuPIMs, IANUS, CXL-PNM) converge on 2024 as the year PIM/PNM entered credible LLM serving territory.
- [[YYYY-dac-cxl-pim]] — vault's own paper benchmarking PIM vs. CXL-PIM trade-offs; CXL-PNM is one of the CXL-PIM design points that benchmarking study contextualizes.
- [[long-context-llm-cxl-optimization]] — this idea's CXL optional layer is exactly the datacenter-tier substrate CXL-PNM demonstrates; cite as a credible existence proof that LPDDR+CXL is a viable serving substrate even without GPUs.
- Introduces LPDDR5X as CXL near-memory compute substrate — no dedicated concept page yet.
- Introduces Python-transparent CXL-PNM software stack — no dedicated page yet.
- [[cim-centric-llm-mobile-soc]] — datacenter-tier complement: CXL-PNM is server-class LPDDR5X CXL near-memory compute; we are mobile-class SRAM-CIM via PCIe + MMIO. Shared narrative on "GPU-free LLM serving via memory-centric compute".
- [[context-aware-moe-cxl-ndp-arxiv2025]] — Context-aware MoE extends the CXL-NDP substrate used here to MoE workloads via prefill-routing oracle for one-shot expert placement; direct successor work on the same CXL-PNM/NDP platform.
