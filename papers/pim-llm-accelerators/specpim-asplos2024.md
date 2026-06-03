---
type: source
title: "SpecPIM: Accelerating Speculative Inference on PIM-Enabled System via Architecture-Dataflow Co-Exploration"
created: 2026-05-25
updated: 2026-05-25
raw_path: https://doi.org/10.1145/3620666.3651352
source_kind: paper
ingest_level: weak
authors: [Cong Li, Zhe Zhou, Size Zheng, Jiaxi Zhang, Yun Liang, Guangyu Sun]
venue: ASPLOS
year: 2024
tags: [pim, llm-inference, speculative-decoding, dse, architecture-dataflow, draft-model, target-model, memory-wall, hbm-pim]
---

# SpecPIM: Accelerating Speculative Inference on PIM-Enabled System via Architecture-Dataflow Co-Exploration

## TL;DR

SpecPIM targets the mismatch between speculative inference's variable compute profile and PIM hardware's fixed resource allocation. The draft language model (DLM) and target language model (TLM) have fundamentally different computation patterns, so SpecPIM builds a joint architecture-and-dataflow design space exploration (DSE) framework that independently satisfies each model's resource demands while fully utilizing system bandwidth. Evaluated against GPU-based and existing PIM-based LLM accelerators, SpecPIM achieves 1.52×/2.02× geomean speedup and 6.67×/2.68× geomean energy efficiency improvement respectively.

## Key claims

*Note: working from abstract and search-retrieved metadata; no open-access full text found as of ingest date.*

- **Variable resource heterogeneity is the core obstacle**: speculative inference introduces two distinct models (DLM for draft generation, TLM for verification) with disparate AI profiles — one is bandwidth-bound GEMV-heavy, the other is compute-bound GEMM-heavy; a single fixed PIM allocation serves neither well.
- **Architecture design space exploration**: constructs a hardware design space that can satisfy each model's different resource demands on a PIM-enabled system.
- **Dataflow design space exploration**: dedicates a separate dataflow design space to maximize utilization of available hardware resources for each model's execution phase.
- **DSE framework**: proposes an automated co-exploration framework that finds optimal design configurations for different target scenarios (latency vs. energy objectives).
- **Performance** (abstract): 1.52× geomean speedup and 6.67× energy efficiency over GPU speculative inference; 2.02× speedup and 2.68× energy efficiency over existing PIM-based LLM accelerators.
- Venue: ASPLOS 2024 (Volume 3, pp. 950–965); authors from Peking University (School of Integrated Circuits + EECS).

## Why it might matter

Speculative decoding is the primary software lever for batch=1 LLM decode, yet all other PIM-LLM papers in the vault ([[neupims-asplos2024]], [[ianus-asplos2024]], [[cent-asplos2025]]) target standard autoregressive decode. SpecPIM is the **only vault source** that co-designs PIM architecture specifically for speculative inference, making it a direct bridge between the [[speculative-decoding]] and [[processing-in-memory-llm]] concept pages. The architecture-dataflow DSE angle is directly analogous to techniques needed for any planned CIM+LLM work — the heterogeneous two-model compute profile is structurally identical to a heterogeneous CIM+NPU pipeline. `relevance: high`

## Connections

- [[processing-in-memory-llm]] — SpecPIM is the speculative-decoding branch of PIM-LLM; substrate is HBM-PIM class (DRAM-NB); introduces speculative-inference PIM — no prior vault page covers this angle.
- [[speculative-decoding]] — SpecPIM applies the DLM+TLM paradigm on PIM hardware; contrasts with purely software approaches ([[medusa-cai-2024]], [[eagle-li-2024]]).
- [[memory-centric-computing]] — SpecPIM fits the memory-centric LLM inference cluster.
- [[in-memory-computing]] — PIM substrate; same compute-near-memory principle.
- [[llm-serving]] — speculative inference is a serving-layer acceleration technique; SpecPIM pushes it to the hardware level.
- [[neupims-asplos2024]] — sibling ASPLOS 2024 NPU+HBM-PIM paper; targets batched autoregressive decode, not speculative inference; SpecPIM's architecture DSE is conceptually adjacent to NeuPIMs' dual-row-buffer + sub-batch interleaving.
- [[ianus-asplos2024]] — sibling ASPLOS 2024 unified-memory NPU+PIM paper; autoregressive decode; the unified-memory vs partitioned-memory axis is orthogonal to SpecPIM's draft/target model axis.
- [[medusa-cai-2024]] — software speculative decoding baseline (multi-head, no separate draft model); SpecPIM uses classic DLM/TLM setup.
- [[eagle-li-2024]] — software speculative decoding baseline (feature-level drafting); architecture-free counterpart to SpecPIM.
- [[cim-centric-llm-mobile-soc]] — sibling PIM+LLM design; SpecPIM's architecture-dataflow DSE methodology is a useful reference for the CIM-centric system's mapping/scheduling phase (M6).
- [[mi-llm-multiplier-free-pim-tc2026]] — real-hardware UPMEM LLM inference (multiplier-free LUT); SpecPIM's simulated HBM-PIM approach contrasts with MI-LLM's real-hardware UPMEM execution; IEEE TC 2026.
- [[sieve-moe-pim-arxiv2026]] — Sieve extends PIM-based LLM inference to MoE with dynamic expert partitioning; SpecPIM's DSE framework is architecturally relevant to Sieve's GPU/PIM dispatch optimization.
- [[duplex-moe-pim-isca2024]] — Duplex targets MoE hot/cold expert split on xPU+Logic-PIM; SpecPIM's architecture-dataflow co-exploration methodology is relevant to Duplex's static/dynamic Op/B-based assignment.
