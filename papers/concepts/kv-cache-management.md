---
type: concept
title: "KV Cache Management"
created: 2026-05-18
updated: 2026-05-18
tags: [llm-serving, kv-cache, memory-management, inference-optimization]
parents: [llm-serving]
---

Key-Value (KV) cache stores intermediate attention computation results (K and V matrices for each token in the context window) to avoid recomputation during autoregressive decoding. As LLMs scale to longer contexts and higher throughput, KV cache memory management becomes the dominant system bottleneck.

## Three-tier taxonomy (from [[kv-cache-management-survey-2025]])

**Token-level**: which tokens' KV entries to retain.
- *Selection / eviction*: sparse attention (only attend to top-k tokens), retrieval-based (fetch by similarity), LRU/recency-based eviction.
- *Budget allocation*: fixed or dynamic budget per head / per layer.
- *Merging*: group similar tokens' KV representations to reduce count.
- *Quantization*: per-head/per-channel low-bit KV compression.
- *Low-rank*: approximate K/V with low-rank matrices.

**Model-level**: architectural changes that reduce KV size by design.
- Multi-Query Attention (MQA), Grouped-Query Attention (GQA): share K/V heads across Q heads.
- Sliding window attention: only attend to local context window.
- State Space Models (SSMs, Mamba): replace KV cache with compressed recurrent state.

**System-level**: where and how KV is stored and moved.
- Paged attention (vLLM): virtual memory for KV; non-contiguous allocation.
- Prefix sharing / radix caching: reuse shared prefix KV across requests.
- CPU offloading: tier KV to CPU DRAM when GPU memory is full.
- Disk offloading: tier to NVMe/eMMC for on-device inference.
- Disaggregated KV pool: KV persists across requests and nodes.
- Workflow-aware eviction: evict based on workflow structure, not recency.

## Vault papers

- **InfiniGen** [[infinigen-osdi2024]]: SVD-based attention prediction + speculative cross-layer KV prefetch from CPU; 3× throughput (OSDI 2024).
- **KVFlow** [[kvflow-multiagent-prefix-2025]]: Agent Step Graph + steps-to-execution eviction + async prefetch for multi-agent; 2.19× concurrent throughput (arXiv 2025).
- **KVSwap** [[kvswap-ondevice-2025]]: full KV on disk + K predictor + group-wise prefetch; 4.1× eMMC, 11× memory reduction (arXiv 2025).
- **Mooncake** [[mooncake-fast2025]]: disaggregated KVCache pool (GPU/CPU/SSD) + KVCache-centric Conductor scheduler; 75% more Kimi requests served (FAST 2025).
- **Survey** [[kv-cache-management-survey-2025]]: three-tier taxonomy covering text + multimodal (arXiv 2025).
- **Autellix** [[autellix-llm-agent-serving-2025]]: exploits KV-cache locality in load balancing — routes long program calls to the engine hosting their KV history; deeper eviction-policy integration is future work.
- **EdgeMoE** [[edgemoe-2023]]: expert buffer management (predict-then-preload + frequency-based eviction) is structurally parallel to KV cache eviction; demonstrates the pattern in an on-device MoE context (TMC 2025).
- **Titans** [[titans-google-2025]]: architectural alternative — neural long-term memory replaces unbounded KV cache with constant-size recurrent state, updated at test time via surprise-driven gradient descent (arXiv 2025).
- **Mem0** [[mem0-production-agents-2025]]: non-parametric agent-tier memory — LLM-distilled facts in a vector store replace raw KV/context across sessions; LOCOMO SOTA at 12× lower p95 latency (arXiv 2025).
- **MemGPT/Letta** [[letta-memgpt-2023]]: OS-paging analogy at the application layer — LLM uses function calls to page data between context window ("RAM") and external storage ("disk"); foundation of Letta framework (arXiv 2023).

## Research angle

The `long-context-llm-cxl-optimization` idea targets KV cache as the primary workload for CXL-memory expansion — see [[long-context-llm-cxl-optimization]].

## Connections

[[llm-serving]] · [[on-device-llm-inference]] · [[llm-agent-systems]] · [[solid-state-drives]] · [[infinigen-osdi2024]] · [[kvflow-multiagent-prefix-2025]] · [[kvswap-ondevice-2025]] · [[mooncake-fast2025]] · [[kv-cache-management-survey-2025]] · [[autellix-llm-agent-serving-2025]] · [[edgemoe-2023]] · [[titans-google-2025]] · [[mem0-production-agents-2025]] · [[letta-memgpt-2023]] · [[long-context-llm-cxl-optimization]] · [[llm-test-time-memory]] · [[hpim-arxiv2025]] (KV cache in SRAM-PIM; long-context capacity limits unanalyzed) · [[papi-asplos2025]] (KV growth drives attention memory-boundedness motivating Attn-PIM) · [[warp-fdp-emulator-fast2026]] (FAST'26: FDP-aware SSD-tier placement; CacheLib kvcache WAF characterization with multi-lifetime separation) · [[l3-dimm-pim-longcontext-arxiv2025]] (DIMM-PIM KV mapping: rank-striped K cache + burst-granularity V cache; resolves bit-level and element-level DRAM layout mismatches for PIM-side MHA; arXiv 2025) · [[mi-llm-multiplier-free-pim-tc2026]] (real UPMEM LLM inference; KV cache placement on near-bank PIM) · [[duplex-moe-pim-isca2024]] (hot/cold KV split across xPU + Logic-PIM for MoE decode; ISCA 2024) · [[pimphony-lolpim-longcontext-hpca2026]] (dynamic KV-cache mgmt via DPA pseudo-MMU + token-centric partitioning for long-context decode; HPCA 2026) · [[starc-sparse-attention-pim-arxiv2025]] (sparsity-aware KV clustering/remapping aligned to PIM row granularity) · [[repa-kvcache-pim-asplos2026]] (reconfigurable PIM jointly accelerating KV-cache offloading + attention compute; head-partitioned locality + sub-batch GPU/PIM pipeline; ASPLOS 2026) · [[dynamic-pim-memory-management]] (runtime VA→PA translation + lazy allocation on PIM to break the static-address limit for KV-cache) · [[sparsity-aware-kv-remapping]] (KV-cache layout matched to PIM row granularity so sparse-attention access becomes row-aligned block access)
