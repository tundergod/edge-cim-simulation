---
type: source
title: "REPA: Reconfigurable PIM for the Joint Acceleration of KV Cache Offloading and Processing"
created: 2026-05-30
updated: 2026-05-30
tags: [processing-in-memory, kv-cache, llm-inference, reconfigurable-computing, reram, long-context, attention, gemv, gpu-pim-hybrid]
raw_path: raw/papers/repa-kvcache-pim-asplos2026.pdf
source_kind: paper
ingest_level: weak
authors: [Yang Hong, Junlong Yang, Bo Peng, Jianguo Yao]
venue: "ASPLOS 2026"
year: 2026
---

## TL;DR

REPA is a GPU–PIM hybrid system using reconfigurable ReRAM PIM (REPA-PIM) to jointly handle KV cache offloading (persistence/retrieval) and decoding-stage attention compute in a single device. The key insight is that both problems share the same data — the KV cache — so a PIM device that can both store and compute on it eliminates the redundant round-trips that plague separate offloading and stage-split approaches. Three co-designed optimizations (bulk-wise microarchitecture, locality-aware data mapping, sub-batch pipelining with transfer overlapping) are required to overcome reconfigurable PIM's lower per-operation speed through massive parallelism. This is a simulation/proposed architecture paper, not evaluated on production hardware.

---

## Motivation (expanded)

- **Problem 1 — offloading overhead**: KV cache is 30–80% of GPU memory in LLM inference (§1). Real traces (Azure23, Llama2-7B) show median KV cache of 670 MiB per request; GPU offloading to SSD causes 0.3–0.8× slowdown at median and 0.5–2.0× at P99, even with only 1–4 evictions (Fig. 3a). The bottleneck is the bandwidth and latency cost of evicting to and reloading from SSD-based offloading systems.

- **Problem 2 — low GPU utilization in decoding**: Scoring (q×K^T) and context (S×V) operations are memory-bound with very low arithmetic intensity; GPU utilization for these non-batchable operations is only ~46% even at batch size 16, vs. ~79% for projection (Fig. 3b, §2). Scaling batch size helps projection but barely moves scoring/context — these are fundamentally bandwidth-limited on GPU.

- **Why existing solutions leave a gap**: Offloading systems solve Problem 1 but the offloaded KV cache cannot be processed until rescheduled — adding re-transfer latency. Stage-split systems (prefill on high-end GPU, decode on "wimpy" GPU) address Problem 2's arithmetic intensity mismatch but leave the decoding performance pit fundamentally unsolved (§1).

- **The joint opportunity**: Both problems are KV-cache problems. A PIM device that *is* the KV cache storage can simultaneously serve as the compute substrate for scoring and context, eliminating all data movement between storage and compute for the non-batchable decoding operations. This motivates a GPU–PIM architecture where GPU handles batchable tasks (FFN, projection, qkv generation, full prefill) and PIM handles non-batchable ones (scoring, context) plus KV persistence (§4, Fig. 8a).

- **Why reconfigurable ReRAM specifically**: DRAM PIM has limited parallelism (compute logic shared across banks/channels); analog ReRAM PIM is fast but has 3.6–22× less capacity than reconfigurable ReRAM at the same area budget due to ADC overhead (Fig. 7), making it unsuitable for storing the memory-hungry KV cache. Reconfigurable DRAM PIM cannot exploit wordline parallelism and is volatile. Reconfigurable ReRAM hits the sweet spot: non-volatile (enabling KV persistence across sessions), high capacity (4F² density), and supports massive in-array parallelism without per-bank CMOS logic (§3, Table 1).

- **The reconfigurability tradeoff**: Reconfigurable PIM needs 4–32× more memory operations per cell than DRAM PIM for the same computation (Table 2), so raw per-operation speed is lower. The entire REPA design is structured to overcome this through parallelism — bulk-wise instructions, multi-level controllers, locality-aware placement.

---

## Method / Idea (expanded)

- **System partition (§4, Fig. 8a)**: REPA is a GPU–REPA-PIM hybrid connected by an interconnect. GPU performs the full prefill stage and all *batchable* decoding tasks (FFN, projection, qkv generation). REPA-PIM performs all *non-batchable* decoding tasks (scoring = q×K^T + softmax, context = S×V) plus KV cache persistence. This clean functional split maps precisely onto the arithmetic-intensity gap: batchable ops tolerate GPU; non-batchable + storage ops go to PIM.

- **REPA-PIM 3D-stacked architecture (§5.1, Fig. 8b)**: One buffer die + eight PIM dies connected via TSVs. Each PIM die has 16 *tiles*; tiles are organized into *tile groups* enabling full parallelization. Each tile has a tile controller (TC) and multiple processing units (PUs); each PU contains 128 1024×2560 cell arrays divided into 4 *array groups*. The two sub-arrays per array (1024×1024 PIM region + 1024×256 temp region) are chosen specifically to match the per-head KV matrix invariant: d_head = 128 → 2048 cells wide → fits exactly in 2× cascaded 1024-column arrays.

- **Bulk-wise memory setting (BLK_SET) instruction (§5.2)**: The core ISA extension. A single BLK_SET instruction specifies a 64-wordline block address (encoded in 24 bits: TG/tile/PU/AG/block hierarchy), plus two input bitlines (16 bits each). This allows NOR-based addition and multiplication to be parallelized across 64 wordlines simultaneously, eliminating per-wordline instruction overhead. Since NOR types within an addition/multiplication have fixed output offsets, REPA-PIM infers output bitlines automatically — no variable-length parameters needed.

- **Multi-level controllers (§5.2, Fig. 8c–d)**: Three controller tiers: tile group controller (TGC) dispatches to tile controllers (TC), TC parallelizes across its PUs by forwarding operations, PU controller (PUC) manages per-array-group computation. Four PUCs per PU chosen as cost-effective sweet spot: speedup scales with #controllers/PU up to 4 (3.91×), then plateaus/declines due to cross-array-group gather overhead and per-die area cost (Table 4). The 4-PUC configuration costs 5.76 mm² per die for 3.91× speedup.

- **Locality-aware data mapping (§6, Figs. 9–10)**: KV matrices are grouped by attention head (not interleaved across far-apart banks/channels as in many DRAM PIM works). Each per-head K or V matrix (at most 1 MiB = 4 cell arrays) is split into four slices placed on *adjacent* cell arrays within the same array group. Three placement rules: (1) each per-head matrix → 4 free cell arrays, same AG; (2) K and V slices from the same decoder block → same AG; (3) per-head KV slices from different decoders → sequential within the AG. This maximizes BLK_SET parallelism (intermediate partial results stay local to the AG) and eliminates long-range inter-group data transfers for intermediate results.

- **Scoring and context execution on PIM (§6.2, Fig. 10)**: For scoring (q×K^T): q vector is replicated and broadcast to all cell arrays holding K slices; element-wise multiplication and reduction happen in-situ per row, producing partial scores S_ij per array group; partial results are gathered and softmax applied. For context (S×V): score rows are replicated, dot-producted with V slices via the same in-array NOR-MAC pipeline. Both operations are fully in-memory, with the final gather being the only inter-group communication.

- **Sub-batch pipelining + transfer overlapping (§7, Fig. 11)**: A batch is split into two interleaved sub-batches running alternately on GPU and REPA-PIM to keep both devices busy (sub-batch pipelining). Additionally, KV matrix transfers between GPU and REPA-PIM are overlapped with computation: during prefill, K transfer is overlapped with GPU scoring, V transfer overlapped with context; during decoding, batched q/k transfers overlapped with v generation, v transfer overlapped with q×K^T on PIM. This hides the KV transfer cost that would otherwise dominate given KV cache sizes.

---

## Key claims

- 1.5–6.5× speedup and 8–10× energy efficiency improvement vs. NVIDIA A100 GPU-only (§abstract, §11).
- Outperforms state-of-the-art DRAM PIM systems by up to 1.4× for long-context inference (§abstract).
- When integrated into FlexGen (existing offloading system), achieves 1.4–2.0× offloading speed and 1.2–1.4× end-to-end speedup (§abstract, §11).
- GPU utilization of scoring/context in decoding is fundamentally below 50% even at batch 16; projection reaches 79% (Fig. 3b, §2).
- Reconfigurable ReRAM PIM has 3.6–22× more capacity than analog ReRAM PIM at the same 10–100 mm² area budget (Fig. 7, §3.2).
- 4-controller/PU configuration is cost-effective sweet spot: 3.91× speedup at 5.76 mm² per-die overhead vs. diminishing returns beyond (Table 4, §5.2).

---

## Why it might matter

REPA's core transferable idea — that the *memory device holding the KV cache* should simultaneously be the *compute substrate for KV attention* — directly informs the [[moe-upmem-inference]] and long-context-on-UPMEM direction. UPMEM DPUs are fixed-function (no reconfigurable NOR arrays), but the same architectural logic applies: place KV slices near DPU compute, partition by attention head for locality, pipeline DPU attention compute with GPU batchable work via sub-batch interleaving, and overlap KV transfers with compute. The BLK_SET bulk-parallelism concept translates to UPMEM's MRAM-access-width tuning and DPU WRAM tiling strategies. Critically, REPA demonstrates that scoring and context — the exact operations UPMEM would handle — are bandwidth-limited enough that even a slower-per-op PIM device wins by moving compute to the data. **Note: REPA is simulation-based proposed hardware; treat as design-space inspiration and methodology reference, not an empirical baseline.**

relevance: high

---

## Connections

- [[processing-in-memory-llm]] — primary concept; REPA is a GPU-PIM hybrid targeting the decoding attention bottleneck via reconfigurable ReRAM PIM
- [[kv-cache-management]] — primary concept; REPA jointly addresses KV offloading and KV attention compute as a unified PIM problem
- [[llm-serving]] — decoding-stage GPU underutilization for scoring/context motivates the split; REPA's sub-batch pipelining is a serving-level scheduling technique
- [[memory-centric-computing]] — REPA instantiates in-memory GEMV for attention via NOR-based reconfigurable compute inside ReRAM cell arrays
- [[moe-upmem-inference]] — the locality-aware head-partitioned KV mapping and sub-batch GPU/PIM pipelining are directly transferable design ideas for UPMEM-based long-context inference
- [[pimphony-lolpim-longcontext-hpca2026]] — sibling long-context PIM work; REPA focuses on reconfigurable ReRAM + offloading joint optimization vs. LoL-PIM's DRAM-PIM long-context angle
- [[l3-dimm-pim-longcontext-arxiv2025]] — sibling long-context PIM proposal; contrasts with REPA's non-volatile ReRAM approach and offloading integration
- [[starc-sparse-attention-pim-arxiv2025]] — sibling PIM-for-attention work; STARC targets sparse attention patterns vs. REPA's dense KV attention on reconfigurable ReRAM
- [[neupims-asplos2024]] — prior PIM-for-batched-LLM work (DRAM PIM); REPA explicitly positions against DRAM PIM systems and claims 1.4× advantage for long-context (§abstract)
- [[papi-asplos2025]] — related PIM-for-attention system (CXL-enabled GPU-free); REPA is GPU-centric with PIM as offload/compute accelerator vs. PAPI's GPU-free angle
- [[cent-asplos2025]] — related LLM serving system at ASPLOS 2025; complementary serving-layer context
- [[infinigen-osdi2024]] — related KV cache offloading system (SSD-based); REPA targets integration with FlexGen-style offloading and is orthogonal to eviction policy mechanisms
- [[kv-cache-management-survey-2025]] — survey context for KV offloading landscape discussed in §2 and §10
- [[mi-llm-multiplier-free-pim-tc2026]] — real-UPMEM PIM baseline for LLM; contrast: REPA is ReRAM reconfigurable (simulation), MI-LLM is real UPMEM DPU deployment
- [[pim-llm-pgemmlib-cgo2025]] — real-UPMEM GEMM library for LLM; contrasts with REPA's in-cell NOR-MAC approach for GEMV in attention
- introduces *reconfigurable-ReRAM-PIM* as a distinct PIM paradigm (vs. DRAM PIM and analog ReRAM PIM) — no page yet
- introduces *bulk-wise memory setting (BLK_SET) instruction* for wordline-parallel NOR-based compute — no page yet
- introduces *sub-batch GPU/PIM pipelining* as a scheduling technique for hybrid GPU-PIM serving — no page yet
- [[dynamic-pim-memory-management]] — REPA's locality-aware head-partitioned mapping and persistent non-volatile KV storage touch the same design space as runtime VA→PA management for PIM KV-cache; see also PIMphony DPA.
- [[sparsity-aware-kv-remapping]] — REPA's head-partitioned locality mapping is the dense-attention counterpart to sparse KV remapping; both align KV layout to PIM access granularity.
