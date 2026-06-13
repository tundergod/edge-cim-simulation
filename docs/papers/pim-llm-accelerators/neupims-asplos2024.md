---
type: source
title: "NeuPIMs: NPU-PIM Heterogeneous Acceleration for Batched LLM Inferencing"
created: 2026-05-22
updated: 2026-05-22
raw_path: raw/papers/neupims-asplos2024.pdf
source_kind: paper
ingest_level: full
authors: [Guseul Heo, Sangyeop Lee, Jaehong Cho, Hyunmin Choi, Sanghyeon Lee, Hyungkyu Ham, Gwangsun Kim, Divya Mahajan, Jongse Park]
venue: ASPLOS
year: 2024
tags: [llm-inference, llm-pim, hbm-pim, npu-pim-heterogeneous, batched-inference, sub-batch-interleaving, dual-row-buffer]
---

# NeuPIMs: NPU-PIM Heterogeneous Acceleration for Batched LLM Inferencing

## TL;DR

NeuPIMs is a heterogeneous NPU + HBM-PIM accelerator for **batched** Transformer LLM inference. The NPU handles compute-bound GEMM (QKV generation, FFN) while a PIM-augmented HBM handles bandwidth-bound GEMV (multi-head attention). Two contributions enable concurrent NPU+PIM operation: (1) microarchitectural **dual row buffers** in each DRAM bank let normal memory traffic coexist with PIM ops; (2) algorithmic **sub-batch interleaving** pipelines two independent sub-batches so one runs GEMM on the NPU while the other runs GEMV on PIM. Result: 3×, 2.4×, 1.6× throughput over NPU-only, PIM-only, and naïve NPU+PIM-integrated baselines.

## Key claims

- **Decoder block = three operators with different AIs** (§1, Fig. 1a–4): QKV gen (GEMM), MHA (GEMV, AI ≈ 0.25–8 FLOP/B in generation phase), FFN (GEMM). Roofline analysis shows generation MHA is severely memory-bound.
- **Current PIMs operate in "blocked" mode** (§1): cannot do PIM ops and normal memory accesses simultaneously → forces NPU↔PIM serialization → inherent under-utilization.
- **Dual row buffers** (§1.2) permit concurrent regular-memory reads/writes and PIM GEMV in the same bank — relaxes the blocked-mode constraint without violating DRAM timing.
- **Sub-batch interleaving** (§1.2): partitions a batch into two sub-batches whose sequence-length sums are balanced; one sub-batch's GEMM (NPU) runs while the other's GEMV (PIM) runs.
- **Throughput gains** (§1, abstract): 3× over NPU-only, 2.4× over PIM-only, 1.6× over naïve NPU+PIM; NPU utilization 28%→65%, PIM utilization 17%→26%.

## Motivation

GPUs and TPUs are GEMM-efficient but underutilized on memory-bound GEMV (decoder MHA). PIM is GEMV-efficient but cannot do GEMM. Naïve integration serializes the two engines. The contribution is to make them genuinely concurrent — both at the DRAM-bank level (dual row buffer) and at the workload-scheduling level (sub-batch interleaving).

## Method

- Built on GPT-3 13B / 175B variants in simulation; integrates **ONNXim** NPU simulator with an in-house PIM simulator on **DRAMsim3**.
- Workload: ShareGPT and Alpaca traces.
- Compares against (a) NPU-only with non-PIM memory, (b) NPU+PIM integrated baseline, (c) naïve NPU+PIM, (d) NeuPIMs (full).

## Results

- 3× over NPU-only, 2.4× over PIM-only, 1.6× over naïve NPU+PIM (abstract).
- NPU utilization 28%→65%; PIM utilization 17%→26%.
- Simulator open-sourced (cited as github.com/casys-kaist/NeuPIMs).

## Contributions

1. Identifies blocked-mode PIM and GEMM-GEMV serialization as the two co-design bottlenecks for NPU+PIM LLM inference.
2. Dual-row-buffer PIM bank microarchitecture allowing concurrent normal and PIM ops.
3. Sub-batch interleaving runtime scheduling for NPU↔PIM pipelining.

## Limitations / open questions

- Simulator-only (ONNXim + DRAMsim3); HBM-PIM dual-row-buffer modification not silicon-validated.
- Server-class HBM target; not edge-applicable to LPDDR4 platforms like Metis.
- Batched serving — does not address single-batch edge use cases (where AttAcc, Cambricon-LLM, CENT target).
- Sub-batch interleaving requires sequence-length similarity between sub-batches; tail variance behavior not quantified.

## D1–D9 review lens

| # | Dimension | Reading |
|---|---|---|
| D1 | Baselines | Strong — naïve NPU+PIM, NPU-only, PIM-only all reported. |
| D2 | Novelty | Two clear deltas: dual row buffer (microarch) + sub-batch interleaving (algo). |
| D3 | Evaluation | Two GPT-3 sizes, two traces; broader MoE / long-context studies absent. |
| D4 | Platform | Simulator-only — HBM-PIM bank modification not silicon. D4 risk moderate. |
| D5 | Motivation | Strong — batched serving is the dominant data-center LLM workload. |
| D6 | Mechanism cost | DRAM-timing-compliant row-buffer modification — quantified at simulator level. |
| D7 | Venue | ASPLOS-natural; HW/SW co-design audience. |
| D8 | Consistency | Coherent. |
| D9 | Significance | High for batched serving; reusable PIM-LLM serving template. |

## Connections

**Phase 2 observations**

- The **dual row buffer** modification is a small but consequential change that unblocks NPU+PIM concurrency. NeuPIMs and IANUS attack the same "blocked-mode" PIM barrier from different angles: NeuPIMs at the bank microarchitecture, IANUS at the runtime scheduler with shared memory. The two papers landed at the same venue (ASPLOS'24) from KAIST + Georgia Tech and SNU + SK hynix + KAIST respectively — a parallel-discovery snapshot.
- **Sub-batch interleaving** is a scheduling lever orthogonal to vLLM's PagedAttention and Orca's iteration-level scheduling. A NeuPIMs-style serving stack would compose them rather than replace.

**Concepts / entities / projects / ideas**

- [[processing-in-memory-llm]] · [[memory-centric-computing]] · [[llm-serving]] · [[in-memory-computing]]
- [[ianus-asplos2024]] — sibling 2024 NPU+HBM-PIM design; unified memory vs dual-row-buffer is the contrast.
- [[cent-asplos2025]] — successor that abandons GPU entirely and uses CXL fabric + GDDR-PIM.
- [[cambricon-llm-micro2024]] — alternative substrate (NAND flash) for the same memory-wall problem.
- [[metis-aipu-nn-v2-2026-05-21]] — appendix M entry #34 (bold, "core comparable").
- [[metis-llm-investigation-desktop-2026-05-19]] — Metis's decode wall is the same GEMV-MHA pattern NeuPIMs addresses on HBM; Metis cannot do dual-row-buffer because its memory is LPDDR4 not HBM-PIM, and even on HBM the modification is silicon-level not OS-level.
- [[cnn-dnn-edge-memory-wall-metis-embedded]] — appendix LLM cross-reference.
- [[long-context-llm-cxl-optimization]] — server-tier serving context.
- [[orca-osdi2022]] · [[vllm-pagedattention-sosp2023]] — serving-system baselines on which NeuPIMs's scheduling sits.
- [[hpim-arxiv2025]] — successor work that replaces the NPU role with SRAM-PIM to unlock intra-token parallelism; claims 1.50× over IANUS, doesn't compare to NeuPIMs directly.
- [[papi-asplos2025]] — ASPLOS'25 follow-on that replaces static kernel mapping with dynamic GPU+PIM scheduling.
- [[lp-spec-arxiv2025]] — mobile-side sibling: speculative-decode on LPDDR-PIM vs NeuPIMs's batched-inference HBM-PIM angle.
- [[cxl-pnm-lpddr-hpca2024]] — alternative 2024 substrate: LPDDR5X CXL-PNM appliance.
- [[l3-dimm-pim-longcontext-arxiv2025]] — extends the GPU+PIM sub-batch interleaving idea to DDR4 DIMM-PIM with adaptive chunk-partitioned scheduling targeting long-context; cites NeuPIMs as HBM-PIM baseline (6.1× over); simulation, arXiv 2025.
- [[mi-llm-multiplier-free-pim-tc2026]] — real-hardware UPMEM LLM inference baseline (multiplier-free LUT); contrasts with NeuPIMs's simulated HBM-PIM; IEEE TC 2026.
- [[sieve-moe-pim-arxiv2026]] — Sieve extends the GPU+PIM partition idea to dynamic MoE expert assignment; NeuPIMs's sub-batch interleaving is an architectural ancestor of Sieve's phase-alternating dispatch.
- [[duplex-moe-pim-isca2024]] — Duplex applies the hot/cold operational partitioning principle to MoE experts on xPU+Logic-PIM; ISCA 2024 sibling applying NeuPIMs-style reasoning to MoE workloads.
- [[context-aware-moe-cxl-ndp-arxiv2025]] — Context-aware MoE on CXL-NDP; uses prefill-phase routing to inform expert placement; extends the NPU+PIM co-design thread to MoE on CXL substrates.
- [[pimphony-lolpim-longcontext-hpca2026]] — PIMphony/LoL-PIM extends NeuPIMs's DRAM-PIM decode offload idea to long-context with dynamic KV-cache management; HPCA 2026.
- [[starc-sparse-attention-pim-arxiv2025]] — STARC applies sparsity-aware KV remapping at PIM row granularity; builds on the decode-offload-to-PIM pattern established by NeuPIMs.
- [[repa-kvcache-pim-asplos2026]] — sibling ASPLOS paper: reconfigurable PIM jointly accelerating KV-cache offloading + attention compute; head-partitioned locality + sub-batch GPU/PIM pipeline; ASPLOS 2026.
- [[op-b-aware-scheduling]] — NeuPIMs' GEMM→NPU / GEMV→PIM split is an early instance of arithmetic-intensity-based Op/B device selection.
