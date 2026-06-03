---
type: source
title: "KVSwap: Efficient On-Device Long-Context LLM Inference via KV Cache Swapping"
created: 2026-05-18
updated: 2026-05-18
tags: [kv-cache-management, on-device-llm-inference, flash-storage, long-context-inference]
raw_path: raw/papers/kvswap-ondevice-2025.pdf
source_kind: paper
ingest_level: full
authors: [Zhang, Xia, Wang]
venue: arXiv
year: 2025
---

# KVSwap: Efficient On-Device Long-Context LLM Inference via KV Cache Swapping

## TL;DR

KVSwap enables long-context LLM inference on on-device hardware (smartphones, edge devices) by storing the full KV cache on disk (NVMe or eMMC) and using a compact in-memory K cache representation as a prefetch predictor. Group-wise prefetching batches disk reads to match storage I/O characteristics. Achieves 1.8× NVMe and 4.1× eMMC throughput over FlexGen, while using 11× less memory than industry-grade vLLM.

## Key Claims

- Long-context on-device inference is memory-constrained: device DRAM cannot hold the full KV cache for sequences of thousands of tokens; disk offloading is necessary.
- Storing the full KV cache on disk creates I/O bottlenecks; naive sequential access doesn't exploit disk characteristics.
- **Compact in-memory K cache (predictor)**: keep only the key (K) vectors in GPU/CPU memory in compressed form; use them to predict which value (V) vectors are needed next. This is possible because K vectors are relatively small and attention sparsity is predictable from K alone.
- **Group-wise prefetching**: group KV blocks by attention pattern similarity; prefetch entire groups to amortize per-I/O overhead across NVMe (4KB block) and eMMC (page) boundaries.
- 11× less memory than vLLM for equivalent context lengths.

## Method

**Full KV on disk**: V vectors (and optionally K vectors) written to NVMe or eMMC after the prefill phase.

**In-memory K predictor**: K vectors retained in compressed form in GPU/DRAM. During decoding, the predictor scores disk-resident V blocks by query-K similarity to determine prefetch order.

**Group-wise prefetching**: K-based attention prediction groups semantically similar positions; a background thread prefetches the corresponding V groups from disk before they are needed. Group size tuned to match NVMe/eMMC I/O unit.

## Results

- **1.8× NVMe throughput** and **4.1× eMMC throughput** over FlexGen.
- **11.0× less memory** than industry-grade vLLM for equivalent context.
- Targets standard on-device hardware (smartphones / embedded).

## Contributions

1. Full-KV disk offloading for on-device LLM with compact K-only in-memory predictor.
2. Group-wise prefetching tuned to NVMe/eMMC I/O characteristics.
3. Largest memory reduction among on-device KV offloading systems (11× vs vLLM).

## Limitations / Open Questions

- eMMC (common in smartphones) has lower bandwidth than NVMe; even 4.1× improvement leaves absolute throughput limited.
- K-only predictor assumes attention sparsity is query-K-computable; models with complex attention patterns (sliding window, cross-attention) may break this assumption.
- Preprint status; not peer-reviewed.
- Energy overhead of continuous disk I/O on battery-constrained devices not analyzed.

## D1–D9 Review Lens

| Dim | Assessment |
|-----|-----------|
| **D1** | Adequate — FlexGen is a relevant baseline for offloading; vLLM used for memory comparison. Missing: comparison to [[infinigen-osdi2024]] (CPU offloading with speculative prefetch) and [[llm-in-a-flash-apple-2023]] (flash-based inference). |
| **D2** | Clear — combining full-disk KV storage with K-only predictor + group-wise prefetch is a practical systems contribution for on-device; delta from FlexGen (naive offloading) is well-defined. |
| **D3** | Limited in available pages — NVMe and eMMC results given; accuracy preservation and long-context quality evaluation needed. |
| **D4** | Strong — real NVMe and eMMC devices on target hardware. |
| **D5** | Strong — on-device DRAM capacity vs LLM KV cache size gap is quantified with 11× memory gap over vLLM. |
| **D6** | Partial — K predictor storage cost not quantified; group-wise prefetch I/O amplification not analyzed. |
| **D7** | arXiv preprint; targets MLSys/MobiSys/ASPLOS (on-device systems). |
| **D8** | Consistent; storage architecture figures expected. |
| **D9** | Meaningful — enabling long-context LLM on smartphones democratizes applications requiring 10K+ token context on-device. |

## Connections

- [[kv-cache-management]] — disk-based KV offloading as a new tier.
- [[on-device-llm-inference]] — enables long-context inference on memory-constrained devices.
- [[solid-state-drives]] — NVMe/eMMC as KV cache storage tier.
- [[llm-in-a-flash-apple-2023]] — LLM in Flash: stores model weights in flash, not KV cache; complementary approach.
- [[infinigen-osdi2024]] — InfiniGen: CPU-tier KV offloading with speculative prefetch; KVSwap extends to disk tier.
- [[fast-ondevice-llm-npu-asplos2025]] — NPU-based on-device acceleration; KVSwap addresses the memory side of the same problem.
- [[long-context-llm-cxl-optimization]] — research idea on CXL for LLM KV; KVSwap's disk approach is the alternative on memory-less CXL-less devices.
- [[pimphony-lolpim-longcontext-hpca2026]] — PIMphony/LoL-PIM addresses the same long-context KV management problem via dynamic KV placement on DRAM-PIM (pseudo-MMU DPA + token-centric partitioning); the hardware-tier complement to KVSwap's software disk-offload approach; HPCA 2026.
