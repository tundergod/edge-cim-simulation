---
type: source
title: "Sparse Attention Remapping with Clustering for Efficient LLM Decoding on PIM"
created: 2026-05-30
updated: 2026-05-30
tags: [processing-in-memory, sparse-attention, kv-cache, llm-inference, long-context, data-mapping, clustering, hbm-pim, attacc]
raw_path: raw/papers/starc-sparse-attention-pim-arxiv2025.pdf
source_kind: paper
ingest_level: weak
authors: [Zehao Fan, Garrett Gagnon, Zhenyu Liu, Liu Liu]
venue: "arXiv preprint"
year: 2025
aliases: ["STARC"]
---

## TL;DR

STARC is a sparsity-aware data mapping scheme for PIM-based LLM decoding that resolves the fundamental mismatch between fine-grained token-wise KV sparsity and the coarse row-granularity of PIM memory arrays. By clustering semantically similar KV pairs via online K-means and physically co-locating each cluster in contiguous memory rows, STARC allows row-level computation to skip entire irrelevant clusters rather than fetching wasted rows. On the HBM-PIM (AttAcc) simulator, STARC reduces attention-layer latency by 19–31% and energy by 19–27% versus token-wise sparsity baselines, closing to 54–74% latency reduction and 45–67% energy savings relative to full KV retrieval.

---

## Motivation (expanded)

- **Dense-attention PIM designs are bandwidth-hungry at long context.** Existing PIM accelerators for MHA (e.g., AttAcc, NeuPIMs) offload the memory-bound attention layer to PIM while keeping QKV projection and FFN on GPU/NPU. As context lengths grow to tens of thousands of tokens, every decoding step must stream the entire KV cache through PIM, fully consuming internal HBM bandwidth and leaving PIM compute underutilized (§I, §II-A).

- **KV sparsity looks like an obvious fix, but its granularity is wrong for PIM.** State-of-the-art dynamic (token-wise) sparsity methods such as InfiniGen and SparQ achieve >90% sparsity with minimal accuracy loss by selecting only the top-B most relevant tokens per query. However, selected tokens are scattered irregularly across many DRAM rows. PIM arrays operate at **row granularity**: even one needed token forces activation of the entire row, fetching all 8 stored KV vectors per row in AttAcc's HBM3 config (§III-A, §IV). Token-wise sparsity therefore causes systematic over-fetching and wasted bitline toggling — the row-activation penalty is paid regardless of how many tokens in that row are actually relevant.

- **Page-wise sparsity (e.g., Quest) fits PIM layout but sacrifices retrieval quality.** Grouping tokens into fixed-size pages aligns with row granularity, eliminating partial-row waste. But page boundaries are defined by token position, not semantic relevance: pages typically contain only one or two important tokens surrounded by irrelevant ones (shown empirically on LLaMA3.1-8B with 4K context, Fig. 4). The hardware efficiency gain comes entirely at the expense of attention quality — a bad tradeoff for long-context tasks (§III-B).

- **The root cause is a layout–relevance mismatch.** Neither existing approach achieves both properties simultaneously: hardware-friendly access patterns AND retrieval of semantically important tokens. The mismatch is structural — conventional KV storage lays tokens out in sequential (positional) order, which is orthogonal to query-time relevance (§III-C).

- **Workload imbalance compounds the problem.** Because sparsity patterns change dynamically per query and per decoding step, static data placement strategies and fixed scheduling cannot pre-exploit sparsity. PIM units end up unevenly loaded: some banks compute on rows dense with important tokens, others activate rows that contribute nothing, leading to throughput loss and energy waste across parallel memory banks (§III-A, §IV).

- **The key insight: align memory layout with semantic proximity, not token position.** If KV pairs that tend to be selected together are co-located in the same DRAM rows, row-level access granularity becomes an asset rather than a liability — activating one row automatically fetches a high-relevance cluster, enabling coarse-grained execution skipping. This is the design rationale for STARC (§III-C, Fig. 1).

---

## Method / Idea (expanded)

- **Semantic clustering of KV pairs into hardware-aligned clusters.** At the end of the prefill phase, STARC applies K-means (cosine similarity metric, K-means++ init, 15 iterations) to all key vectors produced during prefill, grouping semantically similar keys — and their associated value vectors — into C = seq_len / 32 clusters. Each cluster is then written to physically **contiguous memory locations** aligned with PIM bank row boundaries (§V, Algorithm 1, Steps ①–③). The number of clusters is set to seq_len / 32 as a hardware-algorithm tradeoff: each cluster targets ~32 tokens, matching AttAcc's 8-vector burst granularity across 4 banks, improving row utilization and load balance.

- **Cluster centroids as the selection key.** During decoding, STARC does not scan individual KV pairs. Instead, the query vector is dot-producted against each **cluster centroid** (precomputed and stored separately) to rank clusters by relevance. Top-ranked clusters are selected until the KV cache budget B is reached; the final selected cluster may be truncated to stay within budget (Algorithm 1, lines 19–24). This enables coarse-grained, parallel retrieval without per-token scoring overhead at inference time.

- **Incremental re-clustering every 128 decoding steps.** Newly generated tokens accumulate in a temporary buffer during decoding — they are not assigned to existing clusters immediately, since their semantic neighborhood is not yet known and they have outsized influence on recent attention distributions. Every 128 steps, the buffer of I = 128 new KV pairs is clustered separately and merged into the global cluster set C ← C ∪ C_new (Algorithm 1, lines 7–18). This avoids expensive full re-clustering of the entire context while keeping clusters semantically coherent as the sequence grows. Prefill and decoding tokens are clustered separately because their key vector distributions diverge significantly over time (Fig. 7).

- **Prefill and decoding cluster separation.** A key empirical observation (Fig. 7) is that decoding-phase key vectors drift from the prefill distribution as generation progresses. STARC maintains separate cluster sets for prefill-derived and decoding-derived KV pairs. Both sets are searched during retrieval, but clustering them separately prevents the decoding distribution from corrupting prefill cluster structure, maintaining retrieval recall for long sequences.

- **Row-skipping via clusters enables effective PIM execution.** In AttAcc's memory organization, each memory access retrieves 8 KV vectors (one full row across 4 banks × 2 reads). With STARC's layout, these 8 vectors all belong to the same semantic cluster. When that cluster ranks below the budget threshold, the entire row is skipped — no row activation, no bitline toggling, no wasted compute. This converts irregular fine-grained sparsity masks into **structured, hardware-aligned skip operations** that PIM can exploit without complex controller logic (§IV, Fig. 5d).

- **Non-clustered (recent) tokens always included.** The buffer of newly generated tokens (not yet assigned to a cluster) is always included in the attention computation regardless of budget, because recent tokens disproportionately dominate the current attention distribution and evicting them causes accuracy collapse (Algorithm 1, line 24). This design choice keeps STARC practically accurate without requiring the budget to reserve space for them explicitly.

- **Transferability to UPMEM-style DRAM-PIM.** The core insight — *cluster KV pairs by semantic similarity; store each cluster in contiguous, bank-aligned rows; use centroid comparison for coarse-grained retrieval* — is independent of the specific PIM architecture. UPMEM DPUs also operate at row granularity (one MRAM row per transfer), and access-pattern regularity is equally important for UPMEM's in-order DPU processors. The STARC approach provides a direct blueprint for mapping long-context KV caches onto UPMEM: cluster assignments during prefill, contiguous per-cluster layout in MRAM, centroid-based retrieval to generate DPU task lists, with each DPU processing one or a few clusters per decoding step.

---

## Key claims

- STARC reduces attention-layer latency by **19–31%** and energy by **19–27%** vs. token-wise sparsity methods (InfiniGen, SparQ), at the same KV budget (§VI-C, Fig. 11–12).
- At KV budget 1024, STARC achieves **54–74% latency reduction** and **45–67% energy reduction** vs. full KV cache retrieval (§Abstract, §VIII).
- STARC total per-token energy reduction vs. full KV: **8–15%** (attention dominates only as decoding length grows) (§VI-C).
- Accuracy on LongBench (8 tasks, LongChat-7B-v1.5-32K): STARC matches or exceeds token-wise methods (InfiniGen, SparQ) and consistently outperforms page-wise Quest (§VI-B, Fig. 8).
- Recall rate of important tokens: STARC outperforms Quest and InfiniGen; slightly below SparQ at small budgets (§VI-B, Fig. 10).
- Evaluated on AttAcc simulator (HBM-PIM), DGX+AttAcc platform, LLaMA-7B target model, FP16 (§VI-A).
- **SIMULATION ONLY** — no real UPMEM or physical PIM hardware results.

---

## Why it might matter

STARC's sparsity-aware clustering remapping directly addresses the same row-granularity constraint that UPMEM MRAM imposes, making it the closest existing blueprint for a long-context KV-cache layout strategy on real UPMEM hardware; see [[moe-upmem-inference]] for the active UPMEM inference direction and the long-context-on-UPMEM extension. The paper is a simulation study (AttAcc/HBM-PIM), so its numbers do not transfer directly, but the *data-layout-for-sparsity* idea — cluster by semantics, store contiguously, retrieve by centroid — is hardware-agnostic and transferable.

**relevance: high**

---

## Connections

- [[processing-in-memory-llm]] — primary concept; STARC is a sparsity-aware mapping scheme for PIM-based MHA acceleration
- [[kv-cache-management]] — primary concept; the scheme reorganizes KV cache layout to enable sparse selective retrieval
- [[llm-serving]] — STARC targets the autoregressive decoding bottleneck in production LLM serving
- [[memory-centric-computing]] — exploits near-memory compute with awareness of DRAM row-access granularity
- [[neupims-asplos2024]] — related PIM system for batched LLM inference; STARC targets single-request long-context decoding on the AttAcc variant
- [[infinigen-osdi2024]] — directly compared baseline (token-wise dynamic sparsity); STARC outperforms it on PIM hardware efficiency while matching accuracy
- [[cent-asplos2025]] — related PIM-aware attention scheduling work; STARC differs by co-designing *data layout* rather than compute scheduling
- [[kv-cache-management-survey-2025]] — provides taxonomy of KV sparsity/compression methods within which STARC occupies the "clustering + hardware-aware layout" cell
- [[moe-upmem-inference]] — active idea this source should feed: the STARC remapping approach is a direct method inspiration for long-context KV layout on UPMEM
- Batch siblings (will exist): [[pimphony-lolpim-longcontext-hpca2026]], [[l3-dimm-pim-longcontext-arxiv2025]], [[mi-llm-multiplier-free-pim-tc2026]], [[pim-llm-pgemmlib-cgo2025]]
- [[repa-kvcache-pim-asplos2026]] — sibling long-context PIM paper: reconfigurable PIM jointly accelerating KV-cache offloading + attention compute; ASPLOS 2026.
- [[sparsity-aware-kv-remapping]] — STARC is the vault's primary instance of KV-cache layout matched to PIM row granularity so sparse-attention access becomes row-aligned block access.
- Introduces **sparsity-aware KV remapping** — page now exists: [[sparsity-aware-kv-remapping]]
- Introduces **semantic clustering for PIM row alignment** — no page yet
- Introduces **online incremental KV clustering** (per-inference, every N steps) — no page yet
- AttAcc PIM architecture (ref [15], AttAcc ASPLOS 2024) — entity page does not exist
