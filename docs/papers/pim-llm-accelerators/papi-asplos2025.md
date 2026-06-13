---
type: source
title: "PAPI: Exploiting Dynamic Parallelism in Large Language Model Decoding with a Processing-In-Memory-Enabled Computing System"
created: 2026-05-25
updated: 2026-05-25
raw_path: https://arxiv.org/abs/2502.15470
source_kind: paper
ingest_level: weak
authors: [Yintao He, Haiyu Mao, Christina Giannoula, Mohammad Sadrosadati, Juan Gómez-Luna, Huawei Li, Xiaowei Li, Ying Wang, Onur Mutlu]
venue: ASPLOS
year: 2025
tags: [pim, llm-inference, dynamic-scheduling, heterogeneous, gpu-pim, decode, llm-serving, memory-bandwidth, speculative-decoding, batching, kv-cache]
---

## TL;DR

PAPI is a heterogeneous GPU+PIM system for LLM decoding that addresses the key limitation of prior static GPU–PIM mapping approaches: kernel arithmetic intensity changes dynamically at runtime as batch sizes, speculation lengths, and request counts shift. It deploys two specialized PIM units (FC-PIM for fully-connected layers, Attn-PIM for attention) alongside GPU tensor cores, governed by a lightweight online scheduler that continuously re-evaluates and re-maps kernels based on observed parallelism. Reported gains are 1.8× over A100+AttAcc and 11.1× over AttAcc-only baselines.

## Key claims

- LLM decoding kernels are not statically compute- or memory-bound; arithmetic intensity of FC kernels varies continuously with request-level parallelism (RLP) and token-level parallelism (TLP), meaning static GPU–PIM assignment leaves performance on the table (§3.2).
- PAPI introduces a two-phase dynamic scheduler: an *initial schedule* before execution (estimate RLP×TLP against threshold α) and a *runtime reschedule* after each decoding iteration (count completed requests, re-evaluate) (§5.2).
- Hardware comprises three units: GPU tensor cores + HBM-based FC-PIM (NVLink-connected, 4 FPUs/bank, 12 GB) + disaggregated Attn-PIM (PCIe/CXL-connected, 1 FPU/2 banks, 16 GB) (§6.1–6.3).
- End-to-end speedup: 1.8× over A100+AttAcc and 11.1× over AttAcc-only on LLaMA-65B / GPT-3 175B; energy efficiency 3.4× over A100+AttAcc on creative-writing, 3.1× on general-qa (§7.2).
- Roofline analysis confirms attention kernels are always memory-bound regardless of batch size, while FC kernels transition from memory-bound to compute-bound as batch size grows — motivating the asymmetric FC-PIM vs Attn-PIM design (Figure 2).
- DRAM access dominates energy (96.7% without data reuse, dropping to 33.1% at 64× reuse), justifying near-memory compute (Figure 7).

## Why it might matter

PAPI is direct prior art for our planned CIM-enabled mobile SoC LLM work: it establishes the principle that PIM unit design must be kernel-specific (FC vs. Attn) and that a dynamic scheduler is necessary when parallelism levels vary — both constraints apply equally to on-device inference where batch size and speculative-decoding depth fluctuate at runtime. The FC-PIM / Attn-PIM split and the arithmetic-intensity threshold heuristic are concrete design anchors to compare against or adopt in a mobile CIM simulator.

relevance: high

## Connections

- [[processing-in-memory-llm]] — PAPI is a primary instance of PIM-accelerated LLM decode with heterogeneous GPU+PIM architecture.
- [[llm-serving]] — dynamic batching and scheduling of LLM decode requests is core to PAPI's motivation.
- [[on-device-llm-inference]] — while PAPI targets server-class hardware, the dynamic parallelism insight transfers directly to mobile/edge CIM scenarios.
- [[memory-centric-computing]] — PAPI's near-memory compute philosophy and energy analysis (96.7% DRAM access energy) exemplify memory-centric design.
- [[kv-cache-management]] — KV cache growth drives the attention memory-boundedness that Attn-PIM addresses.
- [[in-memory-computing]] — FC-PIM and Attn-PIM are specialized in-memory compute units; design tradeoffs (FPUs-per-bank) are discussed in §6.1–6.2.
- [[speculative-decoding]] — TLP from speculative decoding is one of the two parallelism axes PAPI tracks for dynamic scheduling.
- [[neupims-asplos2024]] — prior work on PIM for LLM serving that PAPI extends with dynamic (vs. static) kernel mapping.
- [[ianus-asplos2024]] — another static GPU+PIM heterogeneous LLM inference system that PAPI's dynamic scheduler is designed to outperform.
- [[cent-asplos2025]] — contemporaneous ASPLOS 2025 work on LLM inference efficiency; relevant for positioning PAPI's contributions.
- [[characterizing-mobile-soc-heterogeneous-llm-inference-sosp2025]] — characterizes heterogeneous LLM inference on mobile SoCs; directly relevant for extending PAPI's dynamic scheduling insights to mobile CIM targets.
- [[cim-centric-llm-mobile-soc]] — sibling research direction: mobile-SoC + CIM-centric (not GPU-centric); we differentiate by being mobile (not server), CIM (not HBM-PIM), and characterization-driven (not pre-committed FC/Attn split).
- [[repa-kvcache-pim-asplos2026]] — sibling ASPLOS paper: reconfigurable PIM jointly accelerating KV-cache offloading + attention compute; head-partitioned locality + sub-batch GPU/PIM pipeline; ASPLOS 2026.
- [[op-b-aware-scheduling]] — PAPI's arithmetic-intensity threshold heuristic (FC-PIM vs Attn-PIM selection) is the canonical Op/B-aware device scheduling approach in the vault.
