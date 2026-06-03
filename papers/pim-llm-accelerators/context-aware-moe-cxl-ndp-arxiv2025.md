---
type: source
title: "Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems"
created: 2026-05-30
updated: 2026-05-30
tags: [mixture-of-experts, near-data-processing, cxl, llm-inference, quantization, expert-placement, gpu-ndp]
raw_path: raw/papers/context-aware-moe-cxl-ndp-arxiv2025.pdf
source_kind: paper
ingest_level: weak
authors: [Zehao Fan, Zhenyu Liu, Yunzhen Liu, Yayue Hou, Hadjer Benmeziane, Kaoutar El Maghraoui, Liu Liu]
venue: "arXiv preprint"
year: 2025
---

## TL;DR

MoE inference is memory-bound once expert weights exceed GPU HBM capacity, forcing expensive repeated parameter transfers over PCIe. This paper shows that expert activation patterns are highly context-dependent (varying across input sequences and decoding steps), making static or purely reactive expert placement suboptimal on GPU-NDP systems. The key insight is that prefill-stage routing statistics strongly predict decoding-stage expert activations, enabling a one-shot placement decision per sequence that pins hot experts on GPU HBM and executes cold experts in-place on CXL-attached NDP, converting parameter movement into cheaper activation movement. A companion context-aware mixed-precision quantization scheme allocates 1–4 bit widths to NDP-resident experts by importance, relieving NDP compute pressure without accuracy collapse.

## Motivation (Expanded)

- **MoE capacity vs. bandwidth wall.** Mixtral-8×22B requires ~280 GB in FP16, far exceeding a single GPU's HBM. All experts must remain accessible, so naive inference over PCIe forces repeated large parameter transfers that dominate latency (expert-migration cost can exceed 90% of Transformer block execution time, §1 / Abstract).
- **CXL-NDP as the offload tier.** CXL-attached NDP devices offer DDR-class capacity (512 GB in the evaluated config) and high internal bandwidth (512 GB/s) at lower cost than HBM expansion. Executing cold experts *in-place* on NDP converts bulky weight movement into small activation movement across the CXL link — the fundamental bandwidth trade-off (§2.2).
- **Context-agnostic placement is the unsolved problem.** Prior GPU-NDP MoE systems (MoNDE, PIIMoE) rely on static or reactive policies. Static mappings ignore per-sequence hot/cold skew; on-demand migration at every decoding step re-introduces CXL bandwidth pressure. Both fail to exploit MoE's inherent activation heterogeneity (§1, §2.2).
- **Expert activation is highly context-dependent.** Empirical analysis on Mixtral-8×7B (WikiText-2, C4) shows activation frequency is far from uniform across experts and varies significantly between input sequences and across decoding steps within a sequence (Fig. 2, Fig. 3, §3.1). A global frequency map therefore cannot capture per-sequence importance; static partitioning will misplace experts for many inputs.
- **Prefill routing predicts decoding routing.** The cosine similarity between prefill-stage and decoding-stage expert activation probability distributions is 0.89 on average across all layers for Mixtral-8×7B (Fig. 4, §3.2). This provides a reliable, low-overhead oracle: collect routing statistics once during prefill, then commit expert placement for the entire decoding phase with no further migration.
- **NDP compute is constrained.** NDP devices operate under tight power and area budgets (64×(4×4) systolic arrays at 1 GHz in the simulated platform, Table 2). Executing cold experts at full FP16 precision on NDP would bottleneck the NDP tier and erode the bandwidth advantage. Quantization is not merely optional compression — it is required to keep NDP computation within budget (§2.3).

## Method / Idea (Expanded)

- **Two-phase architecture: Expert Placement Module + Expert Bitwidth Selector.** Both modules consume the same prefill-stage routing statistics, run once per sequence before decoding begins, and produce a fixed GPU/NDP assignment and a fixed per-expert bitwidth map that hold for the entire decode phase (Fig. 1, §4).
- **Expert importance score (per layer, per expert).** For each expert *e* in layer *l*, two statistics are collected during prefill: activation frequency *P_{l,e}* (how often the expert is selected) and routing-score sum *W_{l,e}* (confidence of each selection). These are normalized and combined as *S_{l,e} = α P̃_{l,e} + (1−α) W̃_{l,e}* with mixing coefficient α. The top-K experts by *S_{l,e}* are pinned to GPU HBM in FP16; the remainder stay on NDP (Algorithm 1, lines 9–18).
- **One migration per sequence.** Expert placement is decided at the prefill/decode boundary and held fixed; each inference sequence undergoes at most one expert migration step. This amortizes migration overhead over the entire decode phase and eliminates the step-by-step transfer overhead of reactive systems (§4.1).
- **Computation follows data residency.** During decoding, when a token's routing decision selects an expert, that expert is executed on whichever device hosts it (GPU for hot experts, NDP for cold experts). This locality principle is the core of converting parameter movement to activation movement: only small intermediate activations cross the CXL link (Algorithm 1, lines 23–32).
- **Prefix-structured mixed-precision quantization on NDP.** For each NDP-resident expert, GPTQ-quantized replicas are pre-cached at 1, 2, 3, and 4 bits. Assignment proceeds as a layer-wise optimization: experts are ranked by importance (descending), and a prefix structure allocates higher bitwidths to more important experts and lower bitwidths to less important ones, subject to a per-layer average bitwidth budget *b̄*. The gain of upgrading expert *i* from 1-bit to *b*-bit is precomputed as *Δ_i(b) = L_i(1) − L_i(b)* (MSE loss reduction) using a calibration dataset; prefix sums *C_2, C_3, C_4* allow O(E²_NDP) optimal search per layer (§4.2).
- **Offline calibration + online routing.** The loss table *L_{l,e}(b)* is built once offline using a small calibration corpus (C4, 1024 samples). At inference time, only the prefill routing statistics and the precomputed importance ordering are needed to run both modules — no online re-calibration (Algorithm 1, lines 1–8 vs. 9–32).
- **Overlap of GPU and NDP execution.** The system pipeline overlaps GPU computation (hot experts in FP16) with NDP execution (cold experts in low precision), achieving latency hiding that reactive migration-heavy baselines cannot exploit (§5.2, Figs. 5–6).
- **Transferable placement signal for non-CXL substrates.** The prefill-routing-predicts-decoding insight is substrate-agnostic: any system with heterogeneous memory tiers (including UPMEM DRAM-PIM) can use prefill statistics as a placement oracle to partition experts across compute tiers without per-step migration.

## Key Claims

- Up to **8.7×–11.2×** decoding throughput improvement over MoNDE (state-of-the-art GPU-NDP baseline) on the same simulated GPU-NDP system; 3-bit variant achieves 6.6–8.3× end-to-end speedup, 2-bit achieves 7.9–10.6× (§5.2, Fig. 6).
- Up to **18×–19×** speedup over HOBBIT (GPU-only mixed-precision offloading baseline) on Mixtral-8×7B and 8×22B respectively (§5.2).
- NDP-side execution alone sees ~5× (3-bit) and ~8× (2-bit) latency reduction vs. MoNDE-NDP, primarily from reduced expert migration and lower-precision execution (§5.2).
- Accuracy cost: **0.13% average drop** for 3-bit variant, 3.4% for 2-bit variant, relative to full-precision MoNDE (Table 3, §5.3).
- Evaluated on Mixtral-8×7B and Mixtral-8×22B; benchmarks include MMLU, MathQA, HellaSwag, ARC-E/C, BoolQ, WinoGrande, PIQA (§5.1).
- **Simulation only** — system simulated on Ramulator with a single H100 GPU + 1×DDR-based NDP device connected via PCIe Gen4 ×16 (Table 2). No real hardware validation.

## Why It Might Matter

The prefill-guided one-shot placement mechanism is directly applicable to [[moe-upmem-inference]]: UPMEM banks are heterogeneous compute tiles with fixed locality, and the same prefill routing statistics could replace runtime profiling or static expert assignment when partitioning experts across DRAM-PIM DIMMs. The mixed-precision NDP quantization design (importance-ranked prefix-split bitwidth allocation) is also a concrete blueprint for handling UPMEM's low per-bank compute throughput. Note that this paper is **simulation-based** on a CXL-NDP platform with dedicated systolic-array compute units — it is method inspiration, not a real-hardware baseline; UPMEM's architecture differs significantly (in-DRAM SIMD, not systolic, no CXL fabric). On the substrate axis, compare with [[cent-asplos2025]] (GPU-free LLM serving on CXL-PIM, real hardware), which shares the CXL memory-fabric motivation but targets full model serving rather than expert-tier offloading.

relevance: high

## Connections

- [[mixture-of-experts]] — primary subject; expert placement, hot/cold partitioning, and routing-statistics-guided bitwidth allocation for MoE inference.
- [[processing-in-memory-llm]] — core technique; NDP executes cold experts in-place adjacent to their weight storage.
- [[memory-centric-computing]] — CXL-attached near-data processing as the capacity/bandwidth offload tier; parameter-movement-to-activation-movement conversion.
- [[llm-serving]] — decode-phase throughput is the primary metric; batched decoding throughput evaluated across multiple I/O lengths.
- [[on-device-llm-inference]] — Mixtral-8×22B serving on a single GPU-NDP node without multi-GPU scale-out.
- [[cent-asplos2025]] — sibling on the CXL-PIM substrate axis; GPU-free LLM serving on CXL-PIM vs. GPU+CXL-NDP hybrid here; compare expert-offloading strategies.
- [[cxl-pnm-lpddr-hpca2024]] — related CXL near-memory processing for LLM; this work extends the CXL-NDP concept to MoE routing-aware placement.
- [[neupims-asplos2024]] — NDP batched inference system; context-aware expert placement extends the per-request heterogeneity insight to NDP tier assignment.
- [[edgemoe-2023]] — MoE expert offloading on constrained devices; this work addresses the reactive/static placement limitation that EdgeMoE shares.
- [[moe-upmem-inference]] — the transferable core idea: prefill-stage routing statistics as a one-shot placement oracle for heterogeneous memory-tier expert partitioning is directly applicable to UPMEM-based expert sharding.
- [[sieve-moe-pim-arxiv2026]] — batch sibling; compare expert selection/placement strategies on PIM substrates.
- [[duplex-moe-pim-isca2024]] — batch sibling; PIM-side MoE expert execution with grouped query attention.
- [[mi-llm-multiplier-free-pim-tc2026]] — batch sibling; low-precision PIM computation for LLM inference.
- [[pim-llm-pgemmlib-cgo2025]] — batch sibling; PIM GEMM library for LLM; compare compute-unit utilization strategies.
- Introduces **context-aware expert placement** as a formal mechanism (prefill-routing-predicts-decoding oracle + one-shot per-sequence placement) — no dedicated concept page yet.
- Introduces **prefix-structured mixed-precision quantization for NDP experts** (importance-ranked prefix-split bitwidth allocation under a per-layer average bitwidth budget) — no dedicated concept page yet.
- [[bimodal-expert-distribution]] — expert activation heterogeneity (context-dependent hot/cold skew) is the motivation for context-aware one-shot placement; links to the bimodal skew concept. (page now exists)
