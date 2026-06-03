---
type: source
title: "LP-Spec: Leveraging LPDDR PIM for Efficient LLM Mobile Speculative Inference with Architecture-Dataflow Co-Optimization"
created: 2026-05-25
updated: 2026-05-25
tags: [processing-in-memory, lpddr, speculative-decoding, mobile-inference, llm-serving, on-device-llm, heterogeneous-architecture, npu, energy-efficiency]
source_kind: paper
ingest_level: weak
raw_path: https://arxiv.org/abs/2508.07227
authors: [unknown — not extracted at weak level]
venue: arXiv
year: 2025
---

## TL;DR

LP-Spec proposes an NPU-PIM heterogeneous architecture that combines LPDDR5 processing-in-memory with speculative decoding for mobile LLM inference. The key insight is that speculative decoding's GEMM-heavy verification phase is mismatched to GEMV-optimized PIM, so the paper introduces a near-data memory controller that interleaves PIM computation with DRAM access and a hardware-aware draft token pruner that reduces redundant speculation. Results claim 13.21× speedup and 99.87× EDP improvement over a mobile NPU baseline.

## Key claims

- LPDDR5-PIM alone is poorly suited for speculative inference because the verification phase (GEMM) conflicts with PIM's GEMV-optimized compute fabric; LP-Spec augments PIM with a GEMM-enhanced microarchitecture to address this (§IV).
- A near-data memory controller enables simultaneous PIM computation and DRAM access with runtime data reallocation, increasing utilization (§IV).
- A hardware-aware draft token pruner eliminates low-confidence draft tokens before verification, reducing wasted GEMM compute (§V).
- Dynamic workload scheduling dispatches GEMV-dominated decode to PIM and GEMM-dominated verification to NPU (§V).
- Achieves 13.21× latency speedup and 7.56× energy efficiency gain over mobile NPU baseline; 12.83× better EDP than AttAcc PIM baseline; 415.31× better EDP than RTX 3090 GPU (§VI).

## Why it might matter

This paper directly intersects the mobile SoC + LPDDR-PIM + LLM speculative decoding design space that is central to our planned CIM-on-mobile simulator and multi-tenant heterogeneous edge SoC work. It is the first (known) paper to jointly co-optimize speculative decoding dataflow with LPDDR-PIM microarchitecture, making it important prior art — any submission in this space must differentiate against LP-Spec's architecture-dataflow co-optimization claims. The draft token pruner angle is also relevant to our dynamic-length HDC and KV-cache work if pruning criteria can be generalized. The 415× EDP gap vs. GPU baseline quantifies the mobile efficiency opportunity precisely.

**relevance: high**

## Connections

- [[processing-in-memory-llm]] — core mechanism; LP-Spec extends LPDDR5-PIM with GEMM support
- [[on-device-llm-inference]] — target deployment context (mobile SoC)
- [[speculative-decoding]] — the inference algorithm LP-Spec co-optimizes with PIM
- [[llm-serving]] — throughput/latency framing; NPU-PIM heterogeneous scheduling
- [[neupims-asplos2024]] — sibling PIM-for-LLM work (batched inference on PIM); LP-Spec targets speculative decode where NeuPIMS targets batching
- [[characterizing-mobile-soc-heterogeneous-llm-inference-sosp2025]] — sibling work characterizing mobile SoC heterogeneity for LLM inference; LP-Spec proposes a specific PIM-augmented architecture for the same target platform
- specpim-asplos2024 — sibling speculative-decoding-on-PIM work; no page yet
- papi-asplos2025 — sibling PIM-based LLM inference work; no page yet
- introduces LPDDR5-PIM GEMM enhancement — no concept page yet
- introduces near-data memory controller for simultaneous PIM+DRAM — no concept page yet
- [[cim-centric-llm-mobile-soc]] — sibling mobile-SoC LLM-PIM direction; LP-Spec is LPDDR-PIM + speculative-decode, we are SRAM-CIM + general decode + mixed-precision; closest substrate-and-platform overlap competitor.
