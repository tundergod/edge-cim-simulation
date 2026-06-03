---
type: source
title: "PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System"
created: 2026-05-30
updated: 2026-05-30
tags: [processing-in-memory, long-context-llm, kv-cache, dram-pim, memory-management, llm-inference, mlir, dynamic-scheduling, hpca2026]
raw_path: raw/papers/pimphony-lolpim-longcontext-hpca2026.pdf
source_kind: paper
ingest_level: weak
authors: [Hyucksung Kwon, Kyungmo Koo, Janghyeon Kim, Woongkyu Lee, Minjae Lee, Gyeonggeun Jung, Hyungdeok Lee, Yousub Jung, Jaehan Park, Yosub Song, Byeongsu Yang, Haerang Choi, Guhyun Kim, Jongsoon Won, Woojae Shin, Changhyun Kim, Gyeongcheol Shin, Yongkee Kwon, Ilkon Kim, Euicheol Lim, John Kim, Jungwook Choi]
venue: "HPCA 2026"
year: 2026
aliases: ["LoL-PIM", "PIMphony"]
---

## TL;DR

PIMphony (formerly LoL-PIM, arXiv:2412.20166) is a multi-node DRAM-PIM orchestration system targeting long-context LLM decode (up to 1M tokens, 72B params). It diagnoses three root-cause inefficiencies in conventional PIM at long context — channel underutilization, I/O bottleneck, static memory management — and addresses each with a co-designed trio: Token-Centric PIM Partitioning (TCP), Dynamic PIM Command Scheduling (DCS), and Dynamic PIM Access (DPA). Implemented via an MLIR-based compiler and evaluated on cycle-accurate DRAM simulators, it achieves up to 11.3× throughput gain over PIM-only baselines.

---

## Motivation

- **Attention becomes the bottleneck at scale.** As context length grows, compute intensity (FLOPs/Byte) of the Attention operation drops sharply (Fig. 2a) while the KV-cache memory footprint grows with both context length and batch size (Fig. 2b), causing GPUs (and naïve PIM systems) to run out of memory before the arithmetic becomes binding. At 128K+ tokens the Attention layer dominates end-to-end latency, yet MAC unit utilization under prior PIM systems drops to ~52% at 32K context (§II.D).

- **Prior PIM systems partition by head/batch, not by token.** CENT and NeuPIMs assign head-batch pairs to PIM channels (Head-First Partitioning, HFP). In long-context inference a single request can occupy an entire channel's memory; available head-batch tiles shrink, leaving most channels idle regardless of how many PIM modules are deployed. Channel utilization collapses under both tensor parallelism and pipeline parallelism variants (§IV.A–B; Fig. 6).

- **Static PIM command scheduling serializes I/O and compute.** The PIM instruction pipeline is a rigid `WR-INP → MAC → RD-OUT` sequence issued in fixed order. The PIM controller enforces worst-case timing gaps between commands even when no true data dependency exists, causing pipeline stalls. For small Attention head dimensions (d_h = 128) typical in long-context GQA models, I/O transfers dominate latency and cannot be hidden by static schedulers (§V; Fig. 8).

- **PIM instructions embed physical addresses at compile time.** This is the root architectural constraint that locks every subsequent inefficiency. Because PIM loop operands reference fixed DRAM row/column addresses, the compiler must pre-allocate KV-cache at compile time assuming the worst-case maximum context length T_max. Observed average capacity utilization under static management is only 36.2% (§II.D; Fig. 10a). Moreover, variable-length real-world requests (Table II: mean 16K–61K, max 119K–209K tokens across LongBench/LV-Eval) waste enormous reserved capacity, and memory cannot be reclaimed or redistributed at runtime.

- **Systemic implication for UPMEM-class real PIM.** UPMEM's DPU also uses a statically compiled instruction stream with fixed SRAM addresses. The same fundamental limitation applies: KV-cache cannot be dynamically paged across DPU local SRAM without a software or hardware indirection layer analogous to DPA. This is the key transferable insight.

---

## Method / Idea

- **Token-Centric PIM Partitioning (TCP) — reorient parallelism from head/batch to token axis.** TCP distributes the token dimension of a single attention head across all PIM channels within a module. In QK^T, each channel handles a distinct token segment; in SV, each channel reduces over its assigned tokens. This decouples channel utilization from batch size and head count, enabling full-channel activation whenever token length ≥ 256 (QK^T) or 32 (SV) for a 16-channel, 16-bank module. TCP results in an intra-module aggregation overhead of <0.2% latency (single inter-channel reduction via EPU) (§IV.C).

- **Dynamic PIM Command Scheduling (DCS) — dependency-aware out-of-order command issue.** DCS augments the PIM controller with two hardware structures: a Dependency Table (D-Table, recording the most recent command to access each GBuf/OBuf entry) and a Status Table (S-Table, tracking execution completion timestamps). When a new command arrives, the controller checks the D-Table to assign a dependency ID; it issues the command as soon as the S-Table confirms its predecessor has completed, bypassing any artificial timing gap. This enables MAC and WR-INP/RD-OUT to overlap within a single buffer pipeline, reducing the example GEMV from 34 cycles to 22 cycles (§V.C; Fig. 7d). DCS also enables GQA row-reuse mapping: since queries/scores sharing the same DRAM row are processed before an ACT/PRE cycle, I/O transfers can be overlapped with the additional WR-INP transfers that GQA incurs.

- **I/O-aware buffering to support DCS.** DCS repurposes the existing single-entry GBuf as a multi-entry dedicated input buffer and expands OutRegs into larger dual-port Output Buffers (OBuf). Dual-port allows concurrent read (drain completed results) and write (accumulate new partial sums), so the MAC pipeline stays full even while I/O transfers are in flight. Area overhead: 0.47% of MAC unit area per bank (§VIII.C).

- **Dynamic PIM Access (DPA) — pseudo-MMU for KV-cache.** DPA introduces two new instructions: `Dyn-Loop` (loop bound determined at runtime from actual token length T_cur, not T_max) and `Dyn-Modi` (modifies a target operand field — e.g., the MAC row/column address — by a specified stride, generating a virtual address on the fly). Virtual addresses are resolved to physical addresses via a per-request Virtual-to-Physical Address (VA2PA) table maintained in the on-module dispatcher. The dispatcher allocates KV-cache in 1MB chunks on demand (lazy allocation), updating the VA2PA table as T_cur grows each decode step; host-PIM communication occurs only when new capacity is needed, not every decoding step (§VI; Fig. 11). DPA eliminates the linear growth of instruction size with T_max (Fig. 10c) and raises average capacity utilization to 75.6% from 36.2%.

- **On-module dispatcher architecture.** Sits inside the PIM HUB (Fig. 5). Contains: instruction buffer (compact DPA-encoded sequences), configuration buffer (per-request metadata: request ID, T_cur), VA2PA table. A decode unit combines DPA instructions + per-request state + VA2PA to generate the final physical PIM instruction stream, staged into the Instruction Queue for the Instruction Sequencer. The dispatcher runs entirely on-module; no host involvement per decode step.

- **MLIR-based compiler and runtime.** PIMphony extends MLIR's dialects with PIM-specific lowering passes: pattern matching identifies transformer decoder subgraphs (QK^T, SV, FFN); codegen passes emit compact DPA-encoded instruction sequences with dynamic partitioning and memory-allocation metadata. Runtime extends the IREE stack with PIM SDK HAL calls. Compilation is offline; runtime overhead is negligible (§VII).

- **Multi-node deployment.** TCP+DCS+DPA compose cleanly with Tensor Parallelism (TP) and Pipeline Parallelism (PP) across modules. TCP is intra-module only (no inter-module synchronization); TP/PP govern inter-module scheduling. The compiler targets both NeuPIMs (xPU+PIM) and CENT (PIM-only) simulator backends.

---

## Key Claims

- Up to **11.3× throughput improvement** over CENT (PIM-only baseline, 32 modules / 512GB) at 128K context on LV-Eval (§VIII.B; Fig. 13b).
- Up to **8.4× throughput improvement** on NeuPIMs (xPU+PIM baseline, 16 modules / 512GB) (§VIII.B; Fig. 14b).
- Scales to **1M context length** and **72B parameter** models; CENT collapses to ~2% MAC utilization at 1M tokens while PIMphony achieves 46.6× speedup at 1M context (Fig. 17b).
- DPA raises KV-cache capacity utilization from **36.2% → 75.6%** (Fig. 19).
- Hardware overhead lightweight: DCS adds 0.5% area / 1.3% power to HUB control blocks; DPA dispatcher <200KB total buffer (§VIII.C).
- Evaluated on cycle-accurate Ramulator-based DRAM simulator with AiMX PIM specifications; simulation-only (no real silicon).

---

## Why It Might Matter

PIMphony is not a competitor to our UPMEM-based work — it targets a custom DRAM-PIM ASIC — but its three mechanisms directly address the same bottlenecks we would hit if we ported long-context LLM decode onto real UPMEM DPUs. Specifically: (1) TCP's token-axis partitioning maps naturally to distributing KV-cache slices across DPUs; (2) DCS's dependency-aware scheduling is an analogue of what a smarter UPMEM task dispatch could do to hide SRAM-DRAM transfer latency within a DPU bank; (3) DPA's lazy virtual-to-physical address table is almost exactly what would be needed to implement paged-attention-style dynamic KV-cache on UPMEM (where DPU programs also use static addresses by default). See [[moe-upmem-inference]] and the sibling long-context UPMEM direction — PIMphony is the clearest published proof-of-concept that these mechanisms are feasible and yield large gains at long context.

`relevance: high`

---

## Connections

- [[processing-in-memory-llm]] — primary concept; PIMphony is a multi-node DRAM-PIM system for LLM decode, directly within scope.
- [[kv-cache-management]] — DPA implements lazy, chunked, virtual-to-physical KV-cache management on-module; this is the paper's most novel memory-system contribution.
- [[llm-serving]] — long-context serving at 1M tokens / 72B params is the primary deployment target.
- [[memory-centric-computing]] — PIM as the memory-centric accelerator for bandwidth-bound Attention.
- [[on-device-llm-inference]] — multi-node PIM-only deployment is a form of dedicated inference hardware (not a GPU cluster).
- [[cent-asplos2025]] — primary PIM-only baseline; PIMphony is benchmarked against CENT and subsumes its architecture with TCP+DCS+DPA.
- [[neupims-asplos2024]] — primary xPU+PIM baseline; PIMphony applies its techniques to NeuPIMs' NPU+PIM system.
- [[infinigen-osdi2024]] — cited for GPU-side continuous batching and KV-cache management; motivates PIM alternative.
- [[kvswap-ondevice-2025]] — related on-device KV-cache management; PIMphony's DPA is the PIM-hardware analogue of software KV swapping.
- [[kv-cache-management-survey-2025]] — PIMphony's DPA and TCP extend the problem space covered by the survey into PIM hardware.
- [[cambricon-llm-micro2024]] — related PIM-based LLM accelerator; motivational context for PIM-only LLM inference.
- [[moe-upmem-inference]] — our driving UPMEM-PIM idea; PIMphony's TCP/DCS/DPA are the closest published mechanism analogs for a long-context-on-UPMEM extension of that project.
- [[l3-dimm-pim-longcontext-arxiv2025]] — sibling paper: DIMM-PIM approach to long-context; closest architectural sibling to PIMphony.
- [[starc-sparse-attention-pim-arxiv2025]] — sibling paper: sparse-attention PIM mapping; complementary sparsity angle to PIMphony's dense-attention optimization.
- [[mi-llm-multiplier-free-pim-tc2026]] — sibling paper: multiplier-free PIM for LLM; different hardware angle.
- [[pim-llm-pgemmlib-cgo2025]] — sibling paper: GEMM library for PIM-LLM; compiler/runtime-level complement.
- [[repa-kvcache-pim-asplos2026]] — sibling paper: reconfigurable PIM jointly accelerating KV-cache offloading + attention compute; head-partitioned locality + sub-batch GPU/PIM pipeline; ASPLOS 2026.
- [[dynamic-pim-memory-management]] — DPA is the vault's canonical instance of runtime VA→PA translation + lazy allocation on PIM for KV-cache management.

**Un-pageable new concepts introduced:**
- *Token-Centric PIM Partitioning (TCP)* — partitioning the token dimension (not head/batch) across PIM channels — no page yet.
- *Dynamic PIM Command Scheduling (DCS)* — dependency-table-driven out-of-order PIM command issue — no page yet.
- *Dynamic PIM Access (DPA)* — on-module virtual-to-physical address translation for runtime KV-cache allocation in PIM — no page yet.
- *PIM pseudo-MMU* — lightweight dispatcher acting as memory management unit inside the PIM HUB — no page yet.
