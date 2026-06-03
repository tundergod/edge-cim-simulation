---
type: concept
title: "LLM Serving Systems"
created: 2026-05-18
updated: 2026-05-18
tags: [llm-serving, inference-systems, throughput, latency, scheduling]
parents: [in-memory-computing]
---

LLM serving systems manage the full pipeline from incoming request to generated text, including batching, scheduling, KV cache management, GPU kernel execution, and result delivery. The core challenge: LLM inference is (1) compute-intensive in the *prefill* phase (processing the input prompt) and (2) memory-bandwidth–intensive in the *decoding* phase (generating one token at a time using cached KV).

## Key system designs

| System | Core idea | Venue |
|--------|-----------|-------|
| **Orca** | Iteration-level scheduling + selective batching | OSDI 2022 |
| **vLLM** | PagedAttention: virtual memory for KV | SOSP 2023 |
| SGLang | RadixAttention: hierarchical prefix sharing | MLSys 2024 |
| **Mooncake** | KVCache-centric disaggregated cluster | FAST 2025 |
| **Pie** | Programmable serving via WebAssembly inferlets | SOSP 2025 |
| **Autellix** | Program-level scheduling (PLAS/ATLAS) | arXiv 2025 |
| **AIOS** | OS-kernel abstraction for LLM agents | COLM 2025 |
| **MemGPT/Letta** | LLM-directed virtual context (OS-paging analogy) | arXiv 2023 |
| **Mem0** | LLM-distilled agent memory store; 12× lower p95 latency than full-context | arXiv 2025 |
| **Titans** | Neural long-term memory updated at test time; KV-cache alternative | arXiv 2025 |

## Key tensions

- **Prefill vs decode**: compute vs memory bandwidth — disaggregating prefill/decode clusters (Mooncake) removes interference.
- **Throughput vs latency**: batching increases throughput but increases per-request latency.
- **KV memory**: the bottleneck as context grows — triggers offloading, eviction, disaggregation.
- **Programmability vs performance**: monolithic engines are fast but not extensible (Pie's target problem).
- **Call-level vs program-level**: optimizing individual LLM calls misses program-level HoL blocking (Autellix).

## Vault papers

- **Orca** [[orca-osdi2022]]: foundational iteration-level scheduling + selective batching; 36.9× over FasterTransformer on GPT-3 175B. Basis of all later serving systems.
- **vLLM / PagedAttention** [[vllm-pagedattention-sosp2023]]: OS-paging-inspired KV-cache management; 96% utilization vs 20–38% Orca; 2–4× throughput; de-facto open-source stack.
- **Mooncake** [[mooncake-fast2025]]: production Kimi architecture; disaggregated KV + Conductor scheduler; 75% more real requests served.
- **Pie** [[pie-sosp2025]]: programmable serving via inferlets (WebAssembly); 1.3–3.4× on agentic workflows.
- **Autellix** [[autellix-llm-agent-serving-2025]]: PLAS/ATLAS program-level scheduling; 4–15× over vLLM.
- **AIOS** [[aios-llm-agent-os-2025]]: OS kernel for LLM agents; 2.1× faster multi-agent execution.
- **InfiniGen** [[infinigen-osdi2024]]: CPU offloading + speculative KV prefetch; 3× throughput.
- **KVFlow** [[kvflow-multiagent-prefix-2025]]: workflow-aware STE eviction + async prefetch; 2.19× concurrent throughput on SGLang.
- **KV Cache Survey** [[kv-cache-management-survey-2025]]: three-tier taxonomy (token/model/system) covering serving-layer KV management techniques.
- **PAPI** [[papi-asplos2025]]: heterogeneous GPU+PIM LLM decoding with online dynamic kernel scheduling; FC-PIM (HBM) + Attn-PIM (PCIe/CXL); 1.8× over A100+AttAcc, 11.1× over AttAcc (ASPLOS 2025).
- **HPIM** [[hpim-arxiv2025]]: single-batch decode-latency-focused heterogeneous PIM (SRAM-PIM + HBM-PIM) with intra-token pipelining; 22.8×/1.50×/5.76× over A100/IANUS/CXL-PNM (arXiv 2025, simulator-only).
- **CXL-PNM (LPDDR5X)** [[cxl-pnm-lpddr-hpca2024]]: TCO-efficient transformer LLM serving; 512 GB / 1.1 TB/s per appliance; beats 8-GPU on latency/throughput/energy/cost (HPCA 2024).
- **LP-Spec** [[lp-spec-arxiv2025]]: LPDDR-PIM mobile-side speculative LLM serving; NPU+PIM heterogeneous (arXiv 2025).

## Connections

[[kv-cache-management]] · [[llm-agent-systems]] · [[on-device-llm-inference]] · [[speculative-decoding]] · [[llm-weight-quantization]] · [[processing-in-memory-llm]] · [[orca-osdi2022]] · [[vllm-pagedattention-sosp2023]] · [[mooncake-fast2025]] · [[pie-sosp2025]] · [[autellix-llm-agent-serving-2025]] · [[aios-llm-agent-os-2025]] · [[infinigen-osdi2024]] · [[kvflow-multiagent-prefix-2025]] · [[kv-cache-management-survey-2025]] · [[letta-memgpt-2023]] · [[mem0-production-agents-2025]] · [[titans-google-2025]] · [[computational-zone-storage-llm]] · [[long-context-llm-cxl-optimization]] · [[characterizing-mobile-soc-heterogeneous-llm-inference-sosp2025]] · [[papi-asplos2025]] · [[hpim-arxiv2025]] · [[cxl-pnm-lpddr-hpca2024]] · [[lp-spec-arxiv2025]] · [[warp-fdp-emulator-fast2026]] (FAST'26: storage-tier WAF characterization for FDP SSDs — Noisy RUH and Save Sequential failure modes that prefix-cache placement targets) · [[l3-dimm-pim-longcontext-arxiv2025]] (long-context DIMM-PIM serving: adaptive sub-batch prefill/decode interleaving + GPU/PIM parallelism scheduler; 6.1× over HBM-PIM; arXiv 2025) · [[mi-llm-multiplier-free-pim-tc2026]] (multiplier-free LUT LLM inference on real UPMEM near-bank PIM; real-hardware UPMEM baseline; IEEE TC 2026) · [[sieve-moe-pim-arxiv2026]] (dynamic token-distribution-aware GPU/PIM expert partitioning for MoE serving) · [[duplex-moe-pim-isca2024]] (Op/B-based hot/cold expert split across xPU + Logic-PIM for MoE LLM serving; ISCA 2024) · [[context-aware-moe-cxl-ndp-arxiv2025]] (prefill-routing oracle for one-shot expert placement on CXL-NDP) · [[pimphony-lolpim-longcontext-hpca2026]] (dynamic KV-cache mgmt + token-centric partitioning for long-context decode on DRAM-PIM; HPCA 2026) · [[starc-sparse-attention-pim-arxiv2025]] (sparsity-aware KV clustering/remapping aligned to PIM row granularity for long-context serving) · [[repa-kvcache-pim-asplos2026]] (reconfigurable PIM jointly accelerating KV-cache offloading + attention compute; head-partitioned locality + sub-batch GPU/PIM pipeline; ASPLOS 2026) · [[op-b-aware-scheduling]] (arithmetic-intensity Op/B device selection between compute-rich processors and near-memory PIM)
