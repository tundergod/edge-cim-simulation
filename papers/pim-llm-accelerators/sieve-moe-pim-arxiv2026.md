---
type: source
title: "Sieve: Dynamic Expert-Aware PIM Acceleration for Evolving Mixture-of-Experts Models"
created: 2026-05-30
updated: 2026-05-30
tags: [mixture-of-experts, processing-in-memory, llm-serving, hbm-pim, dynamic-scheduling, arithmetic-intensity, gpu-pim-coexecution, moe-acceleration]
raw_path: raw/papers/sieve-moe-pim-arxiv2026.pdf
source_kind: paper
ingest_level: weak
authors: [Jungwoo Kim, Rubens Lacouture, Genghan Zhang, Gina Sohn, Qizheng Zhang, Swapnil Gandhi, Christos Kozyrakis, Kunle Olukotun]
venue: "arXiv preprint"
year: 2026
---

## TL;DR

Sieve is a runtime scheduler and execution framework that partitions MoE expert computation dynamically between GPUs and their attached HBM-PIM stacks, using the per-batch token-to-expert distribution as the primary signal. It identifies a fundamental and worsening bimodal distribution in modern sparse MoE models — a small set of popular experts receives many tokens (compute-bound) while most experts receive very few (memory-bound GEMV) — and shows that all prior static PIM offloading strategies fail as this distribution evolves. Evaluated via cycle-accurate simulation (Ramulator 2.0 + HBM3E), Sieve achieves 1.3×–1.6× throughput and interactivity improvement over PIMoE (SOTA MoE-PIM) on Qwen3.5, GPT-OSS, and Qwen3 models.

---

## Motivation (expanded)

- **The bimodal expert distribution is a growing structural trend, not an edge case.** Modern MoE models (Qwen3, GPT-OSS, Qwen3-Next) activate fewer experts per token over time (lower "act-ratio"), creating a long tail where 44.2% of experts in Qwen3-Next at B=64 receive only a single token — pure GEMV — while a small minority processes large batches (GEMM). This trend is empirically quantified via Artificial Analysis Intelligence Index vs. activated parameter ratio across Mixtral→Qwen3→GPT-OSS→Qwen3-Next (§3.1, Fig. 3).

- **The disparity in arithmetic intensity is extreme and batch-persistent.** Popular experts (high N tokens) are compute-bound and efficient on GPU; unpopular experts (N=1, GEMV) are memory-bound and suited for PIM. At B=64, 47.6%/89.3%/65.9% of expert computations in Qwen3/Qwen3-Next/GPT-OSS are memory-bound; this persists even at B=256 (50.1%/56.6%), meaning purely GPU-side approaches hit a hard ceiling (§3.2, §3.3, Figs. 4–5).

- **Static PIM offloading rules are categorically broken for evolving MoE.** Prior work uses fixed thresholds (e.g., N ≥ threshold → PIM) or offloads only attention to PIM, ignoring expert heterogeneity. These rules were tuned for earlier, denser MoE models (Mixtral) and fail for newer, sparser ones because the arithmetic-intensity distribution has shifted (§3.4).

- **All-to-PIM and all-to-GPU strategies both saturate.** AllExp (all experts to PIM) keeps GPU idle during expert computation and cannot scale throughput with batch size; NoExp (all experts to GPU) saturates early because memory-bound experts stay memory-bound. Neither tracks the inverted-L Pareto frontier of throughput vs. interactivity (§3.4, Fig. 6b–d).

- **Inter-GPU communication overhead is a hidden cost that prior schedulers ignore.** Sieve's multi-GPU model assumes each GPU owns its HBM-PIM stacks exclusively (HBM dies accessible only through the attached GPU, interconnected via NVLink). This makes expert-parallelism inter-GPU communication a real bottleneck for token dispatch and aggregation that naive global-interconnect assumptions obscure (§4, §5.1).

- **PIM channel utilization collapses under expert parallelism.** Prior work (PIMoE) uses expert parallelism (EP) to assign different experts to different PIM banks/channels, but with a highly imbalanced expert distribution, many channels sit idle while a few are overloaded — wasting hardware and limiting effective bandwidth (§6.2, Fig. 10).

---

## Method / Idea (expanded)

- **Scheduling signal: runtime token-to-expert distribution.** On each GPU, after the gating network produces the routing map, Sieve runs a lightweight greedy scheduler (≈20 µs overhead on B200) that reads the token count N for each active expert. This is the primary and sufficient signal for partition decisions (§5.1).

- **Objective function: minimize max(T_Comm, T_GPU(G), T_PIM(S)).** The scheduler finds partition S* (experts to PIM) and G=E−S (experts to GPU) that minimizes the bottleneck across three terms: inter-GPU communication time T_Comm, GPU execution time T_GPU(G), and PIM execution time T_PIM(S). This jointly captures bandwidth, compute, and interconnect — unlike prior work which optimizes only one dimension (§5.1, Eq. 1).

- **Greedy descent algorithm with learned PIM cost table.** Experts are sorted descending by token count. Starting from S=E (all on PIM), the scheduler iteratively moves the highest-token expert to GPU if doing so reduces T_total. Stopping when moving the next expert would increase T_total. A runtime cost table maps observed token counts to actual PIM GEMV execution times (learned via exponential moving average); roofline estimates serve as cold-start fallback. Converges within a few iterations (§5.2).

- **Expert layout: tensor parallelism across PIM channels, not expert parallelism.** Each expert's weight matrix is sharded evenly across all PIM channels so every GEMV can exploit full channel-level parallelism regardless of which expert is running. This decouples channel utilization from the expert distribution — every channel stays uniformly loaded for memory-bound experts, eliminating the hot/cold channel imbalance of EP-based designs (§6.2).

- **Skinny GEMM → serialized GEMV decomposition on PIM.** Commercial PIM is optimized for dot-product (GEMV), not GEMM. For experts receiving N=2–4 tokens (skinny GEMM), Sieve decomposes the operation into N sequential GEMV commands. Sub-steps (i) distribute the token tensor to all PIM channel row buffers, (ii) issue the GEMV PIM command for FFN computation, (iii) read results back to GPU on-chip memory. This keeps PIM commands DRAM-interface compatible (NeuPIMs/Duplex-style PIM_GWRITE + PIM_GEMV interface) (§6.2).

- **Dependency graph for overlapping GPU+PIM execution.** Sieve models the MoE layer as a DAG of independent operations: attention (PIM), token dispatch (inter-GPU), Sieve scheduling, PIM expert FFN, GPU expert FFN, and final aggregation. The DAG exposes parallelism between PIM compute, GPU compute, and inter-GPU communication, overlapping all three. Popular (shared) experts get weight prefetch from HBM-PIM→GPU triggered early to maximize overlap with attention (§6.1, Fig. 8).

- **GPU kernel issues PIM commands dynamically.** A custom CUDA kernel reads the Sieve scheduler output at runtime, computes tensor sizes and DRAM addresses per-expert-per-token, and issues PIM_GWRITE + PIM_GEMV commands dynamically. No offline compilation or hardware modification required — purely a software-side scheduling layer on top of existing PIM command interfaces (§6.2).

- **Aggregation and synchronization.** Because unpopular expert results are computed on PIM and popular expert results on GPU, Sieve maintains per-token dedicated on-chip memory addresses for intermediate values. After PIM computation, results are DMA'd back to the originating GPU and combined with GPU-side expert outputs before the final weighted-sum aggregation (§6.1).

---

## Key claims

- **1.6× throughput + interactivity over PIMoE** on Qwen3.5-397B-A17B at B=256; 1.3× on GPT-OSS-120B and Qwen3-30B-A3B (§7.2, Fig. 9).
- Sieve is the only approach that smoothly scales peak throughput with batch size while maintaining interactivity SLAs; both AllExp and PIMoE saturate (§7.2).
- PIMoE shows severely unbalanced PIM channel utilization under bimodal expert distributions; Sieve's tensor-parallel layout achieves uniform channel utilization (§7.2, Fig. 10).
- Under colocated prefill-decode, Sieve achieves 2.4× (B=16) and 2.3× (B=32) throughput over NoExp — better than both AllExp and PIMoE which degrade due to increased prefill compute-bound expert ratio on PIM (§7.3, Fig. 11).
- Sieve scheduler overhead is ≈20 µs on a B200 GPU, negligible relative to inference latency (§5.2).
- Evaluated on NVIDIA B200 GPU + Samsung HBM3E PIM (96 GB/GPU, 8.0 TB/s HBM-PIM bandwidth, 32 pseudo-channels/stack) via Ramulator 2.0 cycle-accurate simulation (§7.1, Table 1).

---

## Why it might matter

Sieve directly addresses the same core problem as [[moe-upmem-inference]]: how to partition MoE expert execution between a near-memory compute substrate and a host processor when arithmetic intensity is highly heterogeneous. Its key transferable insights — using per-batch token-count as the scheduling signal, decomposing the scheduling problem into a min-bottleneck objective over three resource types, sharding each expert across all PIM channels (tensor parallelism rather than expert parallelism), and decomposing skinny GEMM into serialized GEMV — are all applicable to a real UPMEM design where the PIM substrate has even lower compute throughput and the token-distribution signal is equally available. The bimodal distribution analysis also gives us empirical grounding to justify why a static offloading rule for UPMEM MoE would be insufficient.

Note: Sieve is a **simulation-only** result (HBM-PIM / Ramulator 2.0); it is not a head-to-head baseline against real UPMEM hardware. Treat it as a method and design-inspiration source, not a deployment competitor.

`relevance: high`

---

## Connections

- [[mixture-of-experts]] — primary topic; Sieve's scheduler is designed around the bimodal token-to-expert distribution of modern sparse MoE models.
- [[processing-in-memory-llm]] — primary topic; Sieve is a PIM acceleration framework for LLM MoE inference.
- [[llm-serving]] — target workload; throughput/interactivity Pareto curve is the evaluation metric.
- [[memory-centric-computing]] — architectural framing; HBM-PIM as near-memory compute to exploit internal bandwidth.
- [[compute-in-memory]] — related paradigm; PIM processing units near DRAM banks.
- [[moe-upmem-inference]] — our driving idea; Sieve's dynamic scheduler design and tensor-parallel expert layout are directly transferable to UPMEM MoE.
- [[cent-asplos2025]] — prior PIM-for-LLM work; Sieve cites and contrasts against attention-offload-to-PIM strategies.
- [[neupims-asplos2024]] — hardware interface baseline; Sieve reuses NeuPIMs PIM_GWRITE/PIM_GEMV command interface and dual row buffer design.
- [[cambricon-llm-micro2024]] — related MoE-PIM prior work in the comparison family.
- [[edgemoe-2023]] — prior MoE serving work; cited as part of MoE serving landscape.
- [[specpim-asplos2024]] — prior PIM-for-LLM work in related work section.
- **duplex-moe-pim-isca2024** (batch sibling — page not yet created): closest method sibling; Sieve explicitly contrasts against Duplex's random/static expert-to-PIM assignment (Fig. 6d) and its global-interconnect assumption. Duplex's execution model is used as part of the simulation infrastructure.
- **context-aware-moe-cxl-ndp-arxiv2025** (batch sibling — page not yet created): related MoE-NDP co-execution work.
- **mi-llm-multiplier-free-pim-tc2026** (batch sibling — page not yet created): related PIM-for-LLM acceleration.
- **pim-llm-pgemmlib-cgo2025** (batch sibling — page not yet created): related PIM GEMM library for LLM.
- [[bimodal-expert-distribution]] — Sieve is the vault's primary source for empirical evidence of the bimodal token-to-expert skew in modern MoE models; root cause of arithmetic-intensity heterogeneity driving PIM expert scheduling. (page now exists)
- [[op-b-aware-scheduling]] — Sieve's greedy partition algorithm selects experts for PIM vs GPU based on per-expert arithmetic intensity (Op/B), instantiating Op/B-aware scheduling.
- Introduces **arithmetic intensity disparity scheduling** (using per-expert arithmetic intensity as the partition signal) — no concept page yet.
- Introduces **tensor parallelism across PIM channels** (vs. expert parallelism) as a PIM utilization strategy — no concept page yet.
