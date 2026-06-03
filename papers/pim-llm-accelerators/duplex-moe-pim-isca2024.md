---
type: source
title: "Duplex: A Device for Large Language Models with Mixture of Experts, Grouped Query Attention, and Continuous Batching"
created: 2026-05-30
updated: 2026-05-30
tags: [processing-in-memory, mixture-of-experts, llm-serving, grouped-query-attention, continuous-batching, hbm, logic-pim, heterogeneous-compute, op-b-analysis]
raw_path: raw/papers/duplex-moe-pim-isca2024.pdf
source_kind: paper
ingest_level: weak
authors: [Sungmin Yun, Kwanhee Kyung, Juhwan Cho, Jaewan Choi, Jongmin Kim, Byeongho Kim, Sukhan Lee, Kyomin Sohn, Jung Ho Ahn]
venue: arXiv:2409.01141 (MICRO/ISCA-class, Seoul National University + Samsung Electronics)
year: 2024
---

## TL;DR

Duplex proposes a single HBM-based device that integrates a high-Op/B processor (xPU, GPU-equivalent) and a novel Logic-PIM unit, routing each LLM layer or expert to whichever processor matches its arithmetic intensity (Op/B) at that moment. The core insight is that under continuous batching, MoE expert layers fluctuate between low and high Op/B — no single processor type handles both regimes efficiently. By splitting "hot" experts (many tokens) to xPU and "cold" experts (few tokens) to Logic-PIM at runtime, Duplex eliminates the dead-compute problem that plagues both pure-GPU and prior PIM architectures. Evaluation via cycle-accurate simulation on Mixtral, GLaM, and Grok1 shows up to 2.67× throughput and 42% energy savings vs. an H100 GPU baseline.

---

## Motivation (expanded)

- **Op/B as the primary lens.** LLM inference is dominated by memory-bound operations; the key metric is operations-per-byte (Op/B). Fully-connected (FC) layers in the prefill stage have high Op/B (many tokens, compute-bound); the same layers in decoding-only stages drop to Op/B ~ 1–32 (memory-bound). GPU compute utilization falls below 11% for MoE layers and ~2% for attention in decoding-only stages (§III.A, Fig. 4b). Neither compute-heavy GPUs nor memory-optimized PIM devices alone cover this range. (§I, §III)
- **MoE amplifies the memory bandwidth problem.** A token passes through only top-k experts (k=2 for Mixtral/Grok1; k=2 out of 64 for GLaM), but the system must hold all N_ex expert FFN weights in HBM and fetch whichever experts are selected. With a batch of many requests, this causes massive, irregular DRAM access to load expert parameters; the number of distinct experts accessed across a batch scales with batch size × k. (§II.B, §III)
- **Continuous batching raises Op/B unpredictably.** Stage-level scheduling (continuous batching) mixes a prefill sequence for a new request with ongoing decode sequences. A newly arrived request injects L_in tokens into the MoE layer, transiently raising Op/B for whichever experts it selects. After the prefill completes, those same experts revert to low Op/B. This fluctuation is the critical failure mode for a static device assignment: a PIM-only system gets starved during mixed stages; a GPU wastes bandwidth capacity during decode-only stages. (§II.C, §III.A, Fig. 5a)
- **GQA raises attention Op/B — but not enough for GPU.** Grouped-query attention (GQA, used in Llama-2, Mixtral, Grok1) groups K and V heads, reducing their DRAM footprint. With deg_grp = 4–8, each attention operation becomes a narrow GEMM (Q matrix has deg_grp columns), raising Op/B vs. MHA. Still, Op/B stays below the regime where GPU compute utilization is meaningful for decoding sequences; GQA makes prior bank-die PIM worse because Op/B exceeds the sweet spot of ALUs embedded in DRAM dies. (§II.B, §III.A)
- **Prior heterogeneous approaches require weight duplication.** Simply pairing a PIM device with a GPU by replicating MoE weights across both devices wastes memory capacity proportional to the number of devices, which directly reduces the maximum KV cache size and thus the achievable batch size and throughput. (§III.B, Fig. 5c)
- **Bank-die PIM has an area efficiency ceiling for Op/B > 1.** Placing processing units inside DRAM banks already consumes 20–27% of the DRAM die area and cannot exploit the logic process. For Op/B in the range 1–32 (exactly the MoE/GQA operating range), Bank-PIM and BankGroup-PIM show worse energy-delay-area product (EDAP) than Logic-PIM because their compute density is limited by DRAM design rules. (§IV.B, §IV.E, Fig. 8)

---

## Method / Idea (expanded)

- **Duplex device topology.** One xPU (a GPU-equivalent, HBM3-attached, high Op/B) and one or more Logic-PIM stacks share the same HBM3 DRAM dies. xPU and Logic-PIM operate simultaneously through independent active paths on the same DRAM, allocated by bank bundle index. No weight duplication: all expert FFN weights live in one shared memory space, partitioned by bank bundle assignment. (§IV, Fig. 7a)
- **Logic-PIM microarchitecture.** Logic-PIM exploits the trend of decreasing TSV pitch (50 µm → 22 µm in recent HBM). It adds dedicated TSVs to the power TSV area (9% area overhead), quadrupling internal bandwidth beyond standard HBM3 without touching the DRAM die layout. Processing units (512 FP16 MACs per stack at 650 MHz, plus softmax and activation modules) sit on the logic die, accessing all banks simultaneously via bank bundle parallelism — 8 banks read in parallel, yielding 4× effective memory bandwidth vs. a single-bank read. This targets Op/B 1–32, the exact range bank-die PIM handles poorly. (§IV.B, §IV.C, Fig. 6, Fig. 7b–c)
- **Per-layer Op/B-based processor selection.** At each stage, the system determines the Op/B of each layer: MoE layers in decode-only stages → Logic-PIM; FC layers in prefill or mixed stages with high Op/B → xPU; attention in decoding sequences → Logic-PIM; attention in prefill sequences → xPU. The mapping is static per stage type, not dynamically recomputed per token. (§V.A)
- **Hot/cold expert split — the key co-processing mechanism.** Within a single MoE layer execution, different experts process different numbers of tokens (because the gate distributes tokens unevenly and expert skew is inherent). Duplex uses a lookup table pre-populated with estimated processing times for each expert on xPU vs. Logic-PIM as a function of token count. At runtime, after gating: (1) compute total time if all experts run on xPU; (2) progressively offload the experts with fewest tokens to Logic-PIM until the xPU and Logic-PIM finish simultaneously (minimizing wall-clock time). This greedy balancing is done with the lookup table, making decision overhead negligible vs. actual expert execution time. (§V.B, Fig. 10d)
- **Attention co-processing (prefill + decode simultaneously).** In mixed stages, attention for prefill sequences (high Op/B, compute-bound) runs on xPU while attention for decoding sequences (low Op/B, memory-bound) runs in parallel on Logic-PIM. This requires KV matrix memory allocation to be segregated: prefill KV stored in xPU-mapped bank bundles, decode KV in Logic-PIM-mapped bank bundles; migration of K/V occurs once per stage transition. (§V.A, §V.B, §IX.C)
- **Memory partitioning to avoid bank bundle conflicts.** The full HBM address space is divided into four sections by bank bundle index. Expert FFN weights are distributed one-by-one across these four spaces (enabling simultaneous Logic-PIM and xPU access without conflicts). KV cache is alternated among three of the four memory sections, with the fourth reserved for prefill Q/K/V matrices. (§V.C)
- **Expert tensor parallelism for multi-device scaling.** In multi-node systems, Duplex applies expert parallelism between nodes and tensor parallelism within nodes for MoE layers (same distribution as the GPU baseline). The hot/cold co-processing benefit diminishes in multi-device setups with few experts per device, since fewer experts per device limits the granularity of the split. (§V.B)

---

## Key claims

- Up to 2.67× higher throughput vs. H100 GPU baseline (Mixtral, Duplex+PE+ET configuration) — §VII.A, Fig. 11
- 42.03% less energy consumption for Mixtral; 33.28% for GLaM; 34.59% for Grok1 vs. GPU — §VII.D, Fig. 15
- 2.57× lower E2E latency and 58.3% reduction in median TBT latency for GLaM — §VII.B, Fig. 12
- Outperforms 2×GPU in most throughput configurations by exploiting higher memory bandwidth for low-Op/B operations — §VII.A
- Outperforms Bank-PIM by 2.05× throughput in equivalent configuration; Logic-PIM shows superior EDAP over Bank-PIM and BankGroup-PIM for Op/B ≥ 8 — §VII.C, Fig. 8, Fig. 14
- Logic-PIM area overhead: 17.80 mm² per stack (14.71% of a 121 mm² HBM3 logic die) — §VII.E
- All results from cycle-accurate simulation (Ramulator-based); no silicon prototype — §VI

---

## Why it might matter

The hot/cold expert split by Op/B is directly transferable to the [[moe-upmem-inference]] design problem: UPMEM DPUs are effectively a Logic-PIM analogue (memory-near compute, low Op/B sweet spot), and the same gate-induced token imbalance across experts means a greedy token-count-based assignment of cold experts to DPUs and hot experts to CPU/GPU could be implemented in software without new hardware. Duplex is a simulation-only architecture proposal — not a real-hardware competitor — so it functions as method inspiration for how to characterize and exploit expert skew in a heterogeneous inference system.

`relevance: high`

---

## Connections

### Primary concepts
- [[mixture-of-experts]] — central object of study; Op/B analysis of MoE layers under continuous batching is the paper's core motivation
- [[processing-in-memory-llm]] — proposes Logic-PIM as a new PIM microarchitecture; directly extends the design space of PIM-for-LLM
- [[llm-serving]] — continuous batching model analyzed in detail; TBT/T2FT/E2E latency framing used throughout
- [[kv-cache-management]] — attention co-processing requires KV migration between bank bundles across stage transitions; KV recomputation (PagedAttention) discussed as complementary (§IX.C)
- [[memory-centric-computing]] — Op/B-centric device selection is a memory-centric design philosophy
- [[compute-in-memory]] — Logic-PIM is a logic-die CIM variant; explicitly compared against Bank-PIM and BankGroup-PIM on EDAP

### Related source pages
- [[neupims-asplos2024]] — prior work co-processing PIM + NPU for attention; Duplex generalizes the co-processing idea to MoE and adds expert-level granularity
- [[cent-asplos2025]] — continuous batching + PIM design space; shares motivation re: KV cache and batching pressure on memory bandwidth
- [[specpim-asplos2024]] — speculative decoding on PIM; orthogonal acceleration strategy; same HBM-attached PIM hardware context
- [[ianus-asplos2024]] — heterogeneous NPU+PIM device; most directly related prior heterogeneous single-device design; Duplex cites [40] (likely Ianus) as the prior hetero system with MoE weight duplication problem
- [[edgemoe-2023]] — edge MoE inference; complementary scope (resource-constrained vs. datacenter), but shares the expert-routing-aware compute scheduling theme

### Batch sibling pages (will exist after ingest)
- [[sieve-moe-pim-arxiv2026]] — closest sibling: dynamic GPU/PIM expert partition driven by expert popularity/access pattern; same hot/cold split concept instantiated on real GPU+PIM hardware
- [[context-aware-moe-cxl-ndp-arxiv2025]] — CXL-attached NDP for MoE; different interconnect, overlapping expert offload motivation
- [[mi-llm-multiplier-free-pim-tc2026]] — multiplier-free PIM for LLM; different PIM compute style, complementary energy angle
- [[pim-llm-pgemmlib-cgo2025]] — software PIM GEMM library for LLM; software layer that Duplex's Logic-PIM would benefit from

### New concepts introduced (no page yet)
- **Logic-PIM** — logic-die PIM variant exploiting dense TSVs for internal HBM bandwidth amplification (4×) with processing units on the logic die rather than the DRAM die; fills the Op/B 1–32 gap between Bank-PIM (Op/B ~1) and xPU/GPU (Op/B >> 32)
- **Expert and attention co-processing** — simultaneous execution of hot experts on xPU and cold experts on Logic-PIM within the same MoE layer call, using a token-count lookup table for greedy assignment
- [[op-b-aware-scheduling]] — Duplex's hot/cold expert selection via the lookup table is a canonical instance of Op/B-aware device scheduling. (page now exists)
- [[bimodal-expert-distribution]] — Duplex's continuous-batching Op/B analysis empirically demonstrates the bimodal token-to-expert distribution that drives hot/cold split scheduling. (page now exists)
