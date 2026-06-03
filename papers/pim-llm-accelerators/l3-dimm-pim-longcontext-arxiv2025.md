---
type: source
title: "L3: DIMM-PIM Integrated Architecture and Coordination for Scalable Long-Context LLM Inference"
created: 2026-05-30
updated: 2026-05-30
raw_path: raw/papers/l3-dimm-pim-longcontext-arxiv2025.pdf
source_kind: paper
ingest_level: weak
authors: [Qingyuan Liu, Liyan Chen, Yanning Yang, Haocheng Wang, Dong Du, Zhigang Mao, Naifeng Jing, Yubin Xia, Haibo Chen]
venue: arXiv:2504.17584
year: 2025
tags: [llm-inference, processing-in-memory, dimm-pim, long-context, kv-cache, mha-offload, heterogeneous-inference, gpu-pim-scheduling, upmem-adjacent, simulation]
---

# L3: DIMM-PIM Integrated Architecture and Coordination for Scalable Long-Context LLM Inference

## TL;DR

L3 is a hardware-software co-designed system from Shanghai Jiao Tong University that offloads the KV-cache storage and decoding multi-head attention (MHA) entirely to DIMM-PIM host memory, freeing GPU HBM for FC/projection operations and larger batch sizes. Three innovations — in-flight zero-latency data re-layout, dependency-aware communication-computation overlap, and an adaptive chunk-partitioned batch scheduler — together eliminate the DRAM layout mismatch, hide PCIe transfer latency, and balance GPU/PIM parallelism. Evaluation is **simulation-only** (DRAMSim3 + AttAcc simulator); L3 achieves up to 6.1× throughput over HBM-PIM baselines on real-world long-context traces (OpenR1, Dolphin, OpenThoughts, LongBench). The DIMM DDR4 form factor and plug-and-play modular interface place L3's target hardware structurally adjacent to UPMEM.

## Motivation

- **Linear KV-cache capacity pressure** (§2.3, §1): KV-cache size grows linearly with context length and number of requests. A 7B Llama model on an A100 (80 GB HBM) accommodates only ~62 requests at 2K average token length; at 16K tokens MHA latency accounts for >61.3% of iteration time. HBM capacity is structurally inelastic (TSV/interposer density limits add-a-stack solutions).
- **Decoding MHA is exclusively bandwidth-bound** (§1, §2.4): FC (QKV-gen, projection, FFN) is compute-bound and benefits from batching; decoding MHA is bandwidth-bound and *does not benefit from batching* — it reads the entire KV cache from memory once per token generated. These two workload classes have opposite scaling behavior, motivating clean decoupling rather than shared-resource optimization.
- **DRAM layout vs. PIM compute mismatch — two distinct mismatches** (§2.5, §4.1): (a) *Bit-level mismatch* — DIMM burst data is spread across ×8 DRAM chips (one FP16 element's 16 bits land on 2 separate chips), whereas PIM computation requires the full element in a single chip. (b) *Element-level mismatch* — co-processed K elements (same token's head dimension) are interleaved across chips by address, while co-processed V elements are non-contiguous because they are generated at different iterations. Existing approaches use CPU-assisted offline transposition (≥2× base transfer latency penalty) to fix this, which is unacceptable in long-context decode loops.
- **PCIe transfer latency breaks the critical path** (§2.5, §5.1): offloading Q/K/V from GPU to host DIMM memory and returning attention results introduces PCIe-bandwidth-limited round trips. In long-context scenarios (>16K tokens) this overhead cannot be hidden behind FC computation using simple prefetching — the decoding MHA latency outlasts the FC window on both sides.
- **GPU/PIM work imbalance causes idle bubbles on both devices** (§2.5, §5.2, Fig. 3b): within a batch, LLM inference has autoregressive data dependencies — only MHA (prefill and decode) can overlap between GPU and DIMM-PIM. Execution latencies on the two sides vary unpredictably with batch size, token length, and chunking, producing idle bubbles on whichever device finishes first. This is compounded by the need to batch prefilling requests together with decoding requests.
- **HBM-PIM approaches fail to provide both capacity and bandwidth scalability** (§1, §2.6, Table 1): HBM stacks are expensive, not plug-and-play, and fixed per-die capacity does not scale modularly. DIMM-PIM inherits standard DDR JEDEC modularity — capacity and bandwidth both scale by adding DIMMs — matching the dual dimensionality of the KV-cache's memory demands.

## Method / Idea

- **Architectural split — GPU for FC, DIMM-PIM for decoding MHA + KV storage** (§3, §4, Fig. 4): L3 stores all KV caches in DIMM-PIM host memory (2 TB total across 16 channels × 2 DIMMs); model weights and activations stay in GPU HBM. QKV generation (prefill and decode) runs on GPU; the computed K/V vectors are offloaded to DIMMs; decoding MHA executes on DIMM-PIM; only the attention result (vector-sized) returns to GPU. FC operations remain fully on GPU with the batch size maximized by the freed HBM.
- **Zero-latency in-flight re-layout at the rank PU** (§4.1, Fig. 5a–b): a re-layout unit is placed on the DIMM's buffer chip (rank PU). During the GPU-to-CPU data transfer (KV offload), the rank PU performs *fine-grained bit exchange* across the two consecutive burst data beats using double-buffering — this resolves the *bit-level mismatch* (FP16 elements split across ×8 chips) without any additional transfer time and without violating DDR burst protocol. The trick: the host memory controller's SPD timing parameters are intentionally spoofed (tWL reduced by 1 cycle, tWR increased) so the rank PU has exactly one DDR timing slot to perform the in-buffer exchange before the data is committed to DRAM.
- **Two distinct KV mapping methods to resolve element-level mismatch** (§4.2, Fig. 5d–e): (a) *Score/K cache* — each newly generated K vector is partitioned vertically across DRAM chips and mapped to the same logic bank horizontally, so co-processed K elements reside in the same physical bank, enabling all-bank-parallel GEMV with adder-tree mode bank PUs. (b) *Context/V cache* — V vectors are distributed across multiple banks (4 banks) using burst transfer granularity; the S (score) vector is broadcast to all chips; bank PUs act as accumulators in accumulator mode. These two mappings together enable MHA kernel fusion with bubble-free pipelined execution.
- **Pipelined softmax unit on buffer chip** (§4.3, Fig. 5c): a chunk softmax unit on the rank PU partitions the DIMM-PIM hierarchy into three non-interfering domains (host CPU monitoring/config, rank PU managing transfer + chunk softmax, bank PUs executing score/context GEMV). Chunk softmax pipelines score computation with context computation, directly broadcasting softmax outputs to bank PUs without intermediate DRAM storage — this is the mechanism that achieves bubble-free full MHA kernel fusion in PIM.
- **Dependency-aware communication-computation overlap** (§5.1, Fig. 6–7): three techniques together remove most data communication from the critical path. (1) *Concurrent communication and computation*: the insight is that KV offload does not require all ranks to communicate simultaneously — only one rankset per channel is active in communication mode while others continue computing, preserving ~75% of DIMM-PIM's computation power during transfer. (2) *Load balancing across ranksets*: KV cache for different layers of the same request is distributed across ranksets alternately (layer-granularity striping), ensuring even transfer latency. (3) *Move Q/K/V for decoding off the critical path*: only the vector-sized Q, K, V needed for the current decoding iteration are transferred synchronously; the bulk of prefilling KV can be transferred asynchronously during projection/FFN operations (typically <16% of FF latency).
- **Adaptive chunk-partitioned cross-device batch interleaving** (§5.2–5.3, Fig. 7): L3 scheduler splits requests into two sub-batches and overlaps the *prefilling* of sub-batch 1 with the *decoding MHA* of sub-batch 0 on DIMM-PIM. Key feature: *fine-grained tunability* — the scheduler dynamically chunks overlong prefilling requests so that GPU prefill latency aligns with DIMM-PIM decode MHA latency, minimizing idle bubbles on both sides. A runtime profiling model (linear for PIM latency, Random Forest Regression for GPU latency) enables *performance predictability*, allowing the scheduler to pre-compute sub-batch execution times and make proactive chunking decisions.
- **DIMM form factor and structural proximity to UPMEM** (§2.2, §6.1): DIMM-PIM uses standard DDR4 DRAM modules with PUs placed at rank and bank level — plug-and-play into any DDR4 slot. UPMEM is also a DDR4-compatible DIMM-form-factor PIM device with per-bank DPUs. The layout challenges, DDR protocol constraints, burst-transfer granularity, and the rank/bank PU hierarchy that L3 addresses are directly structurally analogous to what would need to be solved for a long-context MHA offload system built on UPMEM. L3's solutions to the bit-level and element-level mismatches are therefore a concrete method blueprint.

## Key claims

- Up to **6.1× throughput** over HBM-PIM baselines (NeuPIMs, AttAcc) across GPT-89B/175B and OpenR1/Dolphin/OpenThoughts traces (§6.2, Fig. 8).
- Up to **9.2×** over CPU-offloading baseline (NEO/FastDecode) (§6.2).
- **5.1× throughput** when simultaneously scaling both bandwidth and capacity 8× (vs. 1.6× bandwidth-only or 1.1× capacity-only), demonstrating that L3 uniquely exploits *dual dimensionality* (§6.2, Fig. 9).
- Time-between-tokens (TBT) for L3 with 16 ranksets is **29–53% of GPU-only baseline** TBT — host memory integration does not sacrifice latency while improving throughput (§6.3, Fig. 11).
- **Larger batch sizes** without increasing TBT: for GPT-175B, L3 achieves up to 14.3× larger batches than GPU-only on DGX-A100, removing the batch-size ceiling caused by HBM KV-cache competition (§1).
- Simulation basis: DRAMSim3 + AttAcc simulator; TSMC 28nm hardware synthesis for rank PU and bank PU area/power validation (Table 2, Table 5). **Not a silicon prototype.**

## Why it might matter

L3 is a method blueprint for long-context MHA-decode offload onto DDR DIMM-form-factor PIM — and UPMEM is exactly that form factor. The three mechanisms (in-flight re-layout to fix DRAM bit/element mismatches; rankset-level communication-computation overlap to hide PCIe latency; adaptive sub-batch interleaving to balance GPU/PIM parallelism) are all transferable design patterns for [[moe-upmem-inference]] extended to long-context scenarios or a dedicated long-context-on-UPMEM project. The simulation-only nature means results are aspirational, not measured, but the problem decomposition is rigorous and hardware-protocol-grounded.

**Note: simulation paper.** L3's DIMM-PIM is a proposed architecture evaluated in simulation (DRAMSim3), not a commercial chip. UPMEM is *real silicon* — the directional value of L3 is the method, not the benchmark numbers.

relevance: high

## Connections

- [[processing-in-memory-llm]] — primary concept; L3 occupies the DIMM-PIM substrate row (absent from the vault's substrate taxonomy table — no page yet for DIMM-PIM as a distinct substrate)
- [[kv-cache-management]] — KV-cache capacity/bandwidth scaling is the core problem; L3's mapping methods (rank-striped K cache, burst-granularity V cache) are novel KV placement strategies
- [[llm-serving]] — long-context serving throughput and TBT tradeoff; L3's sub-batch interleaving targets the same batching/latency objective as [[neupims-asplos2024]]
- [[memory-centric-computing]] — DIMM-PIM as a modular, plug-and-play memory-centric compute substrate
- [[moe-upmem-inference]] — UPMEM is structurally DIMM-PIM; L3's layout + scheduling mechanisms are directly transferable design inputs for extending the MoE-UPMEM project to long-context decode
- [[neupims-asplos2024]] — closest prior art (NPU+HBM-PIM sub-batch interleaving); L3 adopts the same sub-batch idea but on DIMM-PIM with adaptive chunking and targeting capacity scalability HBM cannot offer
- [[cent-asplos2025]] — CXL-PIM full-model-on-PIM; orthogonal substrate choice (CXL vs. DDR DIMM) and workload split (all ops vs. MHA-only)
- [[cxl-pnm-lpddr-hpca2024]] — LPDDR5X-PNM (near-memory processing on host memory); same motivation of moving computation to host memory, different interface (CXL/LPDDR vs. DDR4 DIMM)
- [[infinigen-osdi2024]] — introduces X — no page yet for DDR-memory-side inference coordination (software-level), complements L3's hardware co-design framing
- [[kv-cache-management-survey-2025]] — survey context for the KV-cache capacity/bandwidth bottleneck that L3 targets
- [[repa-kvcache-pim-asplos2026]] — sibling long-context PIM paper: reconfigurable PIM jointly accelerating KV-cache offloading + attention compute; head-partitioned locality + sub-batch GPU/PIM pipeline; ASPLOS 2026.
- [[dynamic-pim-memory-management]] — runtime VA→PA translation + lazy allocation on PIM; L3's adaptive chunk allocation addresses the same static-address constraint.
