---
type: source
title: "PIM Is All You Need: A CXL-Enabled GPU-Free System for Large Language Model Inference (CENT)"
created: 2026-05-22
updated: 2026-05-22
raw_path: raw/papers/cent-asplos2025.pdf
source_kind: paper
ingest_level: full
authors: [Yufeng Gu, Alireza Khadem, Sumanth Umesh, Ning Liang, Xavier Servot, Onur Mutlu, Ravi Iyer, Reetuparna Das]
venue: ASPLOS
year: 2025
tags: [llm-inference, llm-pim, cxl, gpu-free, dram-nb-pim, pnm, llama-2, tco, mutlu]
---

# PIM Is All You Need: A CXL-Enabled GPU-Free System for Large Language Model Inference (CENT)

## TL;DR

CENT is a **GPU-free**, all-PIM LLM inference system. It builds a CXL network of CXL devices, each containing 16 memory chips with two GDDR6-PIM (PIM) channels and processing-near-memory (PNM) compute units. Hierarchical PIM+PNM supports the entire transformer block. Compared to A100 GPU baselines (at the largest supported batch size), CENT achieves **2.3× higher throughput, 2.9× less energy, and 5.2× more tokens-per-dollar TCO** on Llama-2 70B. The paper's argument: memory-bound LLM decode does not need expensive compute throughput; it needs cheap memory bandwidth + capacity, and CXL+PIM delivers both.

## Key claims

- **LLM decode is memory-bound + capacity-hungry** (§1, §2): GPU operational intensity for Llama-2-70B is ~21% utilization on 4 A100; KV cache scales with context (up to 1M-token windows), forcing capacity that pushes batch size down.
- **GDDR6-PIM (AiM-class) delivers ~16 TB/s aggregate internal BW** (§2) vs A100's 2 TB/s external HBM bandwidth — 8× the bandwidth at lower cost per GB.
- **CENT architecture** (§1): CXL switch connects multiple CXL devices; each device has 16 chips × 2 GDDR6-PIM channels + PNM (RISC-V + accelerators) for Softmax / sqrt / div / non-MAC ops.
- **End-to-end performance** (§1 abstract, §1.3): 2.3× throughput, 2.9× less energy, 5.2× tokens-per-dollar TCO vs A100 baselines on Llama-2 70B.
- **CXL primitives** (§1): inter-device send/receive/broadcast/multicast/gather via CXL transactions; intra-device shared-buffer.
- **Parallelism strategies** (§1.3): Pipeline Parallel (PP), Tensor Parallel (TP), and hybrid TP-PP — mapping transformer blocks across CXL devices.

## Motivation

State-of-the-art LLM serving uses multi-GPU systems (e.g., 4×A100, 80 GB each), in which GPUs are massively under-utilized because LLM decode is memory-bound. Memory-capacity scaling (long context windows, large KV caches) further reduces achievable batch size, exacerbating GPU under-utilization. The cost-benefit case for GPUs in this regime is poor — CENT removes the GPU entirely and builds an all-PIM CXL serving fabric.

## Method

- **CENT simulator** (open-source: github.com/Yufeng98/CENT) — cycle-accurate device + CXL fabric.
- Workloads: Llama-2 70B (and scaled to Grok 314B, Llama-3 405B, DeepSeek-V3 671B in projection).
- Context lengths up to 32K (extended discussion to 1M).
- Comparison baselines: 4×A100 80 GB at maximum supported batch (also discussed: lower batch as context grows).
- Energy and TCO accounted (PIM chip + CXL switch + system power).

## Results

- 2.3× throughput, 2.9× less energy, 5.2× tokens/dollar TCO on Llama-2 70B vs A100.
- Higher gains predicted at longer contexts and larger models (Grok 314B, Llama-3 405B, DeepSeek-V3 671B).
- Hybrid TP-PP balances latency and throughput; pure PP prioritizes throughput; pure TP prioritizes latency.

## Contributions

1. First GPU-free CXL-fabric LLM serving system using GDDR6-PIM (AiM-class) chips.
2. Hierarchical PIM (matrix-mul) + PNM (Softmax / sqrt / non-MAC) covering the whole transformer block.
3. CXL communication primitive set (send/receive/broadcast/multicast/gather) supporting TP/PP/hybrid parallelism over CXL.
4. Open-source cycle simulator.

## Limitations / open questions

- CXL devices with embedded GDDR6-PIM do not exist commercially — silicon credibility is simulator-only.
- 1M-context projections are extrapolations; not measured.
- The TCO argument depends on assumed GDDR6-PIM chip and CXL switch costs — sensitive to vendor pricing.
- Mixed-batch / dynamic-context serving (vLLM-style) interaction with hierarchical PIM scheduling is open.

## D1–D9 review lens

| # | Dimension | Reading |
|---|---|---|
| D1 | Baselines | A100 baselines plus extensive ablation on parallelism strategies. AiM/HBM-PIM cited but not direct end-to-end comparison. |
| D2 | Novelty | "All-PIM, no GPU" framing + CXL-fabric is the delta over AttAcc / NeuPIMs / Cambricon-LLM. |
| D3 | Evaluation | Llama-2 70B in depth; extrapolation to Grok/Llama-3/DeepSeek. |
| D4 | Platform | Simulator-only; chips and CXL switch do not exist. D4 is the dominant attack surface. |
| D5 | Motivation | Strong — LLM serving cost / GPU under-utilization is well-documented. |
| D6 | Mechanism cost | TCO model presented; CXL switch latency / area not silicon-validated. |
| D7 | Venue | ASPLOS-natural. |
| D8 | Consistency | Coherent. |
| D9 | Significance | Very high — credible "what if we remove GPU" thought experiment with strong numbers; sets the all-PIM CXL-LLM design point. |

## Connections

**Phase 2 observations**

- "PIM Is All You Need" is the most provocative LLM-PIM paper of 2025 — it argues that GPUs are *the wrong tool* for memory-bound LLM decode and demonstrates a credible all-PIM alternative. The TCO argument (5.2× tokens-per-dollar) is the strongest economic case for PIM-LLM in the vault.
- Mutlu's group + Reetuparna Das (UMich) collaboration; the paper deliberately reads as a manifesto for the post-GPU LLM-serving world. Compare with the IANUS / NeuPIMs / Cambricon-LLM cohort which all *augment* GPU/NPU rather than replace it.
- The **CXL + PIM combination** is exactly the substrate the vault's [[metis-cxl-cim-memory-system]] idea contemplates from the SRAM-CIM side — CENT is the GDDR-PIM-tier counterpart. The four-tier hierarchy framing (L1/L2 / on-device / host / CXL) becomes more compelling if CXL itself is *compute-capable*.
- Long-context (1M) extrapolations in CENT line up with the [[long-context-llm-cxl-optimization]] motivation; CENT's TP/PP mapping over CXL is what that idea's CXL fork would need to differentiate against.

**Concepts / entities / projects / ideas**

- [[processing-in-memory-llm]] · [[memory-centric-computing]] · [[computational-memory-hierarchy]] · [[llm-serving]] · [[in-memory-computing]]
- [[metis-aipu-nn-v2-2026-05-21]] — appendix M entry #62 (bold, "core comparable"); CENT is the closest published reference for "CIM + CXL hierarchy for LLM".
- [[metis-llm-investigation-desktop-2026-05-19]] — CENT's 16 TB/s GDDR6-PIM vs Metis's measured 24 GB/s LPDDR4 on-card wall: the bandwidth gap (~666×) is what makes "no GPU but lots of PIM" workable. Metis's substrate is structurally on the wrong side of this gap for LLM.
- [[metis-cxl-cim-memory-system]] — direct architectural inspiration for the four-tier hierarchy idea.
- [[long-context-llm-cxl-optimization]] — server-tier CXL competitor.
- [[cxl-pim-storage-vs-memory-upmem]] — [[YYYY-dac-cxl-pim]] benchmarking; CENT is the LLM-specific point in this PIM-vs-CXL-PIM design space.
- [[YYYY-dac-cxl-pim]] · [[pim-case-study-atc2021]] — UPMEM PIM-class baseline; CENT differs by using GDDR6-PIM and CXL fabric rather than DDR4-PIM with native DRAM access.
- [[neupims-asplos2024]] · [[ianus-asplos2024]] · [[cambricon-llm-micro2024]] — sibling LLM-PIM substrates.
- [[cnn-dnn-edge-memory-wall-metis-embedded]] — appendix LLM cross-reference.
- [[hpim-arxiv2025]] — related single-batch heterogeneous PIM (SRAM-PIM + HBM-PIM); HPIM claims 5.76× over CXL-PNM but conspicuously does not compare to CENT despite same niche.
- [[papi-asplos2025]] — co-venue ASPLOS 2025 work on heterogeneous PIM-based LLM inference; PAPI is GPU+PIM dynamic, CENT is all-PIM.
- [[cxl-pnm-lpddr-hpca2024]] — temporal predecessor (HPCA 2024) in the GPU-free CXL-PIM serving lineage; LPDDR5X-PNM vs CENT's GDDR-PIM substrate.
- [[lincoln-hpca2025]] — consumer/mobile analog: LPDDR-flash-PIM for 50–100B on-device vs CENT's CXL+GDDR-PIM for server-scale.
- [[l3-dimm-pim-longcontext-arxiv2025]] — GPU+DIMM-PIM heterogeneous contrast: L3 *keeps* GPU for FC and offloads only MHA-decode to DDR4 DIMM-PIM; compares to CENT as an all-PIM substrate-choice alternative (Table 1); arXiv 2025, simulation.
- [[mi-llm-multiplier-free-pim-tc2026]] — real-hardware UPMEM near-bank PIM LLM inference; complementary real-hardware data point to CENT's GDDR-PIM simulation; IEEE TC 2026.
- [[sieve-moe-pim-arxiv2026]] — Sieve takes CENT's GPU+PIM fabric and extends it to dynamic token-distribution-aware MoE expert partitioning; direct application of CENT's substrate to MoE workloads.
- [[duplex-moe-pim-isca2024]] — Duplex applies Op/B-based hot/cold expert partitioning to xPU+Logic-PIM; CENT's all-PIM argument motivates the logic-PIM side of this split; ISCA 2024.
- [[context-aware-moe-cxl-ndp-arxiv2025]] — Context-aware MoE on CXL-NDP; extends CENT's CXL-fabric premise to MoE with prefill-routing expert placement.
- [[pimphony-lolpim-longcontext-hpca2026]] — PIMphony/LoL-PIM addresses long-context KV management on DRAM-PIM substrates similar to CENT's GDDR-PIM; HPCA 2026 follow-on angle.
- [[starc-sparse-attention-pim-arxiv2025]] — STARC's sparsity-aware KV remapping is directly applicable to CENT's all-PIM attention execution; PIM row-granularity alignment is a substrate concern shared with GDDR-PIM.
