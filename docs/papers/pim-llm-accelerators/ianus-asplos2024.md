---
type: source
title: "IANUS: Integrated Accelerator based on NPU-PIM Unified Memory System"
created: 2026-05-22
updated: 2026-05-22
raw_path: raw/papers/ianus-asplos2024.pdf
source_kind: paper
ingest_level: full
authors: [Minseok Seo, Xuan Truong Nguyen, Seok Joong Hwang, Yongkee Kwon, Guhyun Kim, Chanwook Park, Ilkon Kim, Jaehan Park, Jeongbin Kim, Woojae Shin, Jongsoon Won, Haerang Choi, Kyuyoung Kim, Daehan Kwon, Chunseok Jeong, Sangheon Lee, Yongseok Choi, Wooseok Byun, Seungcheol Baek, Hyuk-Jae Lee, John Kim]
venue: ASPLOS
year: 2024
tags: [llm-inference, llm-pim, npu-pim-heterogeneous, unified-memory, end-to-end-llm, gpt-2, fpga-prototype, sk-hynix]
---

# IANUS: Integrated Accelerator based on NPU-PIM Unified Memory System

## TL;DR

IANUS integrates a commercial NPU with a commercial PIM (SK hynix AiM/HBM-PIM-class) into a **unified main memory system** — the PIM memory simultaneously serves as the NPU's main memory. This removes the data-duplication / cross-engine movement that earlier "accelerator-style PIM" attempts inherited from partitioned memory designs. A new **PIM Access Scheduling (PAS)** runtime schedules concurrent PIM ops and normal NPU memory accesses across the shared memory. IANUS achieves 6.2× and 3.2× end-to-end GPT-2 speedup vs A100 GPU and DFX (FPGA SOTA), respectively. A proof-of-concept FPGA prototype with commercial PIM + NPU + FPGA-based PIM controller demonstrates feasibility.

## Key claims

- **End-to-end LLM has diverse AIs** (§1, Fig. 2): complex vector ops, MHA (GEMV), FC (GEMM) — no single engine fits all. Pure-NPU is FC-good but MHA-bound; pure-PIM is MHA-good but FC-bad; DFX (FPGA) is GEN-stage-tuned but SUMM-stage-weak.
- **Identifies ~90% shared parameters between NPU and PIM in LLM** (§1) → motivates *unified* (not partitioned) memory rather than duplicate copies.
- **PIM Access Scheduling (PAS)** (§1): orchestrates concurrent PIM ops and NPU normal memory accesses on shared memory while respecting DRAM timing — the key runtime contribution.
- **Performance** (abstract, §1): 6.2× over A100, 3.2× over DFX, on GPT-2 end-to-end.
- **FPGA proof-of-concept** (§1): commercial PIM + NPU + FPGA PIM-controller — moves beyond simulator-only credibility.

## Motivation

Earlier PIM-LLM systems treated PIM as an "accelerator" with its own dedicated memory and copied shared parameters between NPU memory and PIM memory. For LLMs where ~90% of weights are shared, this duplication wastes capacity and forces movement. The unified-memory design eliminates duplication but introduces a new contention surface (PIM ops vs. NPU normal accesses), which PAS resolves.

## Method

- Unified memory architecture where the PIM die's memory is the NPU's main memory.
- **PIM Access Scheduling (PAS)** maps workload across NPU/PIM honoring (a) resource contention on the shared memory and (b) parallel-execution opportunity between engines.
- Detailed cycle simulation of end-to-end GPT-2 inference.
- FPGA proof-of-concept with commercial PIM (SK hynix-class), commercial NPU, and FPGA PIM controller.

## Results

- 6.2× over A100, 3.2× over DFX on GPT-2.
- FPGA prototype demonstrates that unified-memory NPU+PIM is implementable with commercial components.

## Contributions

1. First end-to-end LLM accelerator with NPU+PIM **unified** (not partitioned) memory; eliminates ~90% weight duplication overhead.
2. PIM Access Scheduling (PAS) runtime for concurrent NPU memory accesses + PIM ops on shared memory.
3. FPGA proof-of-concept with three commercial / off-the-shelf components — strongest D4 (platform credibility) among 2024 LLM-PIM papers.

## Limitations / open questions

- GPT-2 only — 2024-vintage LLMs (Llama-2/3, MoE) absent.
- A100 baseline is dated by 2026; H100/B200 would be a fairer comparison.
- FPGA prototype runs at reduced clocks — performance projections rely on simulation.
- Unclear how PAS scales to multi-NPU / multi-PIM-die settings.

## D1–D9 review lens

| # | Dimension | Reading |
|---|---|---|
| D1 | Baselines | A100 + DFX both reported; HBM-PIM/AiM as PIM-only baseline. |
| D2 | Novelty | Unified-memory partitioning is the clear delta over AttAcc / NeuPIMs / HBM-PIM. |
| D3 | Evaluation | GPT-2 only — narrow workload coverage. |
| D4 | Platform | FPGA + commercial PIM + commercial NPU — strongest in the 2024 LLM-PIM cohort. |
| D5 | Motivation | Strong — data duplication in partitioned-memory PIM is a real overhead. |
| D6 | Mechanism cost | PAS scheduler overhead quantified in simulator. |
| D7 | Venue | ASPLOS-natural. |
| D8 | Consistency | Coherent. |
| D9 | Significance | High — establishes unified-memory as a design point. |

## Connections

**Phase 2 observations**

- IANUS's headline insight — **~90% of LLM parameters are shared between NPU and PIM** — is the cleanest framing of why partitioned-memory PIM designs are wasteful for LLM. This is the *novel* observation that distinguishes IANUS from AttAcc / NeuPIMs (both partitioned-memory). The follow-on PAS scheduler is the operational consequence.
- The FPGA prototype with commercial PIM + commercial NPU is the **strongest D4 (platform credibility)** in the 2024 LLM-PIM cohort. All other LLM-PIM papers in the vault are simulator-only (Cambricon-LLM SSDsim; NeuPIMs ONNXim+DRAMsim3; CENT custom simulator; AttAcc HBM model). This makes IANUS a candidate counter-example when a CIM/PIM paper draft is criticized for being simulator-only.
- The author list spans SNU + SK hynix + KAIST + SAPEON — an unusually broad cross-industry collaboration; SK hynix is the AiM PIM chip vendor.

**Concepts / entities / projects / ideas**

- [[processing-in-memory-llm]] · [[memory-centric-computing]] · [[llm-serving]] · [[in-memory-computing]]
- [[neupims-asplos2024]] — sibling 2024 ASPLOS NPU+PIM paper; partitioned vs unified memory is the orthogonal axis.
- [[cent-asplos2025]] — successor in the same line; CENT abandons GPU, IANUS abandons partitioning.
- [[cambricon-llm-micro2024]] — flash-substrate sibling.
- [[metis-aipu-nn-v2-2026-05-21]] — appendix M (IANUS is in the broader 71-paper list).
- [[metis-llm-investigation-desktop-2026-05-19]] — Metis's situation is the *opposite* of IANUS — Metis has a unified compiler+memory stack (good) but a closed precompile path (bad); IANUS shows what a *researchable* unified stack looks like.
- [[cnn-dnn-edge-memory-wall-metis-embedded]] — appendix LLM cross-reference.
- [[metis-cxl-cim-memory-system]] — CXL fabric is a natural extension of IANUS's unified memory idea to a fourth tier.
- [[hpim-arxiv2025]] — benchmarks against IANUS, claims 1.50× speedup through intra-token pipelining.
- [[papi-asplos2025]] — contemporary dynamic-scheduling counterpart addressing IANUS's static assignment limitation.
- [[duplex-moe-pim-isca2024]] — Duplex applies IANUS's unified NPU+PIM memory reasoning to MoE workloads, using Op/B-based hot/cold expert partitioning across xPU+Logic-PIM; ISCA 2024.
