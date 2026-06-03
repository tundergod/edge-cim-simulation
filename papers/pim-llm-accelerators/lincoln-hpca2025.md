---
type: source
title: "Lincoln: Real-Time 50~100B LLM Inference on Consumer Devices with LPDDR-Interfaced, Compute-Enabled Flash Memory"
created: 2026-05-25
updated: 2026-05-25
raw_path: https://ieeexplore.ieee.org/document/10946816/
source_kind: paper
ingest_level: weak
authors: [Weiyi Sun, Mingyu Gao, Zhaoshi Li, Aoyang Zhang, Iris Ying Chou, Jianfeng Zhu, Shaojun Wei, Leibo Liu]
venue: HPCA
year: 2025
tags: [llm-inference, consumer-device, lpddr, compute-enabled-flash, in-flash-compute, on-device-llm, mobile-soc, flash-bandwidth, sparsity, ffn-reuse, eager-prediction, hardware-software-codesign]
---

# Lincoln: Real-Time 50~100B LLM Inference on Consumer Devices with LPDDR-Interfaced, Compute-Enabled Flash Memory

## TL;DR

Lincoln is a device-architecture co-design targeting real-time inference of 50–100B-parameter LLMs on consumer devices (phones, laptops), where the bottleneck is weight loading from flash to the NPU every generation iteration. It attacks both the *internal* flash bandwidth (via array-shrinking improvements that lower read latency and increase parallel flash planes per die) and the *external* LPDDR transmission bandwidth (by moving compute into the flash die itself, reducing the volume of data that must cross the interface). On the software side, FFN-Reuse (inter-iteration sparsity: identify and skip redundant FFN computations across iterations) and a modified eager-prediction method (intra-iteration attention sparsity: predict attention scores to skip unnecessary computation within each iteration) further reduce both compute and data-movement demands. HPCA 2025; Tsinghua University + MetaX Technology.

## Key claims

- **Dual bandwidth bottleneck**: 50–100B LLM weight loading is dominated by both (a) low internal NAND flash bandwidth and (b) low LPDDR transmission bandwidth between flash and NPU — both must be addressed together (abstract, §I).
- **Device-level: array shrinking extended**: builds on existing array-shrinking techniques to lower flash read latency and pack more parallel flash planes within each flash die, boosting internal bandwidth (§III, device design).
- **Architecture-level: compute-enabled flash via LPDDR interface**: places compute logic inside the flash module accessible over the standard LPDDR interface, so intermediate or partial results can be produced in-situ, reducing data volume delivered to the NPU (§III, architecture).
- **FFN-Reuse (inter-iteration sparsity)**: identifies FFN layers whose outputs are sufficiently similar across consecutive decode iterations and skips redundant computation, exploiting the structured temporal redundancy of autoregressive generation (§IV).
- **Intra-iteration sparsity via eager prediction**: adapts the "eager prediction" method to accurately predict attention scores ahead of time, skipping the computation and data loading of low-scoring attention heads or queries within a single iteration (§IV).
- **Real-time target**: the combined system is claimed to reach real-time throughput (≥3–10 tok/s threshold) for 50–100B models on consumer devices (abstract); exact measured numbers require full paper access.

## Why it might matter

Lincoln sits at the intersection of two ideas that our vault tracks closely. First, it is the most direct hardware-architecture treatment of the same problem [[cambricon-llm-micro2024]] addresses (in-flash compute for 70B+ on-device), but Lincoln uses the standard **LPDDR interface** rather than a custom chiplet D2D link — a critical practical distinction since LPDDR is already present on every consumer SoC, making Lincoln's design point more manufacturable and pin-compatible with real consumer devices. Second, and most directly relevant: our planned CIM-on-mobile-SoC simulator work contemplates **SRAM-CIM** as the substrate (tight-loop, high-bandwidth-on-die) while Lincoln uses **NAND flash as compute substrate** (high capacity, far from CPU/NPU, lower bandwidth). Lincoln is therefore the closest published competitor-reference architecture — same problem, same consumer-device target, different memory substrate. A submission comparing or complementing SRAM-CIM with flash-PIM must engage Lincoln directly.

The software techniques (FFN-Reuse, eager prediction) are substrate-independent and could migrate to an SRAM-CIM design as algorithm-level optimizations.

relevance: high

## Connections

**Links to existing pages (bidirectional):**
- [[on-device-llm-inference]] — Lincoln is a new hardware architecture for Strategy 1 (flash weight loading) + new HW substrate; extends the Cambricon-LLM and LLM in a Flash design points
- [[processing-in-memory-llm]] — Lincoln fills a new row in the substrate taxonomy: LPDDR-interfaced flash-PIM on consumer SoC (distinct from chiplet D2D in Cambricon-LLM)
- [[in-storage-computing]] — compute-enabled flash die is an instance of in-storage computing; extends [[2026-dac-flashhd]] pattern to LLM workloads at much larger scale
- [[memory-centric-computing]] — weight-loading bandwidth wall as primary design driver
- [[compute-in-memory]] — in-flash GEMV reduction is a CIM primitive
- [[llm-serving]] — single-batch decode on consumer device; complementary to server-side disaggregation
- [[solid-state-drives]] — NAND flash internal bandwidth engineering (array shrinking, plane parallelism)
- [[cambricon-llm-micro2024]] — closest competitor: chiplet NPU + dedicated in-flash compute for 70B; Lincoln differs in using LPDDR interface (no custom D2D) and targeting 50–100B
- [[llm-in-a-flash-apple-2023]] — software-only predecessor: windowing + bundling for flash-resident weights; Lincoln adds on-die compute to eliminate the bandwidth ceiling that limits Apple's approach
- [[characterizing-mobile-soc-heterogeneous-llm-inference-sosp2025]] — HeteroInfer characterizes the same consumer SoC memory bandwidth problem (68 GB/s LPDDR peak, GPU+NPU concurrency); Lincoln attacks the bandwidth wall from the flash side rather than the NPU-scheduling side
- [[cent-asplos2025]] — CENT uses CXL+GDDR-PIM for server-scale LLM; Lincoln is the consumer/mobile analog (LPDDR-flash-PIM)

**No existing page yet:**
- Mingyu Gao (Tsinghua, IIIS) — PI; active in memory-centric AI systems — no person page yet
- Zhaoshi Li (MetaX Technology) — industry co-author — no entity page yet
- MetaX Technology — Chinese AI chip company (cf. GPU alternative ecosystem) — no entity page yet
- Array shrinking (NAND flash plane density scaling) — no concept page yet
- LPDDR-attached compute module — no concept page yet
- [[cim-centric-llm-mobile-soc]] — sibling research direction targeting consumer-device LLM; Lincoln uses flash-PIM for very large models (50–100B), we use SRAM-CIM for mid models (1–13B); both attack the same on-device LLM aspiration from different substrate choices.
