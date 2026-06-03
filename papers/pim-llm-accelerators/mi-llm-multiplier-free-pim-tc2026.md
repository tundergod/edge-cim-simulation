---
type: source
title: "MI-LLM: Multiplier-Free LLM Inference on Commodity Processing-in-Memory Hardware"
created: 2026-05-30
updated: 2026-05-30
tags: [processing-in-memory, llm-inference, quantization, lookup-table, upmem, near-bank-pim, multiplier-free, model-partitioning, llm-weight-quantization]
raw_path: raw/papers/mi-llm-multiplier-free-pim-tc2026.pdf
source_kind: paper
ingest_level: full
authors: [Puyun Hu, Minhui Xie, Linjiang Li, Kuiyaohui Zhang, Erge Xiang, Jing Wang, Size Zheng, Xiao Zhang, Yunpeng Chai]
venue: "IEEE Transactions on Computers"
year: 2026
---

# MI-LLM: Multiplier-Free LLM Inference on Commodity Processing-in-Memory Hardware

## TL;DR

MI-LLM is the first system to run full LLM inference entirely on commodity near-bank PIM (NBP) hardware (UPMEM), bypassing the hardware multiplier deficit by replacing all matrix multiplications with lookup-table (LUT) accesses. A learning-based LUT construction method preserves model accuracy; a locality-aware row-reordering scheme beats SRAM capacity constraints; and a distributed matrix-partitioning strategy cuts the 0.05 GB/s inter-PE communication bottleneck. On UPMEM, MI-LLM achieves an 8–9% throughput gain and 11–27% energy-efficiency gain over GPU (NVIDIA A6000), while incurring only 0.24 perplexity point degradation versus FP8 quantization.

---

## Motivation / Problem

GPUs waste ~99% of compute on LLM decode due to the memory bottleneck: a 38.7 TFLOPS GPU delivers only 321 GFLOPS peak on linear kernels. NBP hardware (UPMEM: 2560 PEs, 2 TB/s aggregate bandwidth) has the bandwidth but not the ALUs — its 12.5 GFLOPS peak is 3000× below GPU because 20nm process nodes leave no room for hardware multipliers. Three specific NBP obstacles block naive porting:

1. **Low compute capability** — no hardware multipliers; software FP simulation yields <1/8 bandwidth utilization.
2. **SRAM latency exposure** — NBP PEs cannot reorder instructions or schedule threads, so any DRAM access falls on the critical path; LUT sizes exceed 64 KB SRAM, causing constant DRAM spills.
3. **Throttled inter-PE bandwidth** — all PE↔PE transfers must route through the host CPU, capping cross-core bandwidth at 0.05 GB/s (versus 0.5 GB/s to DRAM), making naive tensor-parallel partitioning self-defeating.

---

## Method

### Core idea: replace multiplication with LUT lookup

An LUT is an $m \times n$ table of precomputed products. At runtime, each operand is quantized to an 8-bit index; the result is retrieved by a 2D table lookup — zero multiplications. MI-LLM uses $256 \times 256$ LUTs (8-bit × 8-bit) stored as 32-bit integers. The 8-bit choice is the sweet spot: 4-bit yields too few quantization levels; 16-bit blows past PE SRAM; 8-bit fits within PE SRAM and maps to hardware byte-addressing.

### Learning-based LUT construction (offline)

Naive uniform quantization into a fixed LUT degrades perplexity by ~0.5 points. MI-LLM treats quantization values as learnable parameters and optimizes them with gradient descent via the straight-through estimator (STE), minimizing reconstruction loss $\min_Q \|\text{quant}_Q(W)X - WX\|^2$ layer-by-layer. This yields non-uniform quantization values matched to each layer's weight distribution — capturing outlier-heavy LLM weight statistics better than simple FP8 or uniform INT8. Two further optimizations:

- **Fused unary functions**: since activation functions (SiLU, GeLU) follow immediately after linear ops, their computation is absorbed into the same LUT lookup (kernel fusion at LUT level), eliminating a second table traversal.
- **Softmax-specialized LUT**: the standard softmax requires exponentials over a wide dynamic range. MI-LLM uses $\exp(x) = 2^{x/\ln 2} = 2^{\lfloor x/\ln 2 \rfloor} \times 2^{\{x/\ln 2\}}$, precomputing only the fractional part in a compact LUT; the integer part is a free bit-shift on NBP.

### Locality-aware LUT lookup (online kernel)

Because LUT size (256×256 entries) exceeds PE SRAM (64 KB), naive random lookups hit DRAM every access. MI-LLM exploits that in a matrix-vector product the same LUT *row* is reused across all columns sharing the same input value:

- **Inter-row reordering**: cluster identical activation values together so each LUT row is loaded into SRAM once and reused for all columns whose input hash to it. Weight matrix rows are *virtually* reordered (metadata only, no physical data movement).
- **Intra-row reordering**: for sparse matrices in CSR format, sort the nonzero indices within each row so elements sharing the same product value are contiguous — they share one SRAM register load instead of re-fetching from DRAM.

Together these achieve 3.08× speedup over a naive LUT kernel on the dense path; sparse kernels reach 1.38× over base-sparse after intra-row reordering.

### Model partitioning to minimize inter-PE communication (offline)

A single UPMEM PE cannot hold even a 7B-parameter model. Naive tensor parallelism forces each PE result to transit through the CPU, saturating the 0.05 GB/s link. MI-LLM partitions by exploiting structural independence:

- **MHA**: each attention head is assigned to one PE; heads are fully independent, so the host only aggregates after all heads finish (two host interactions total per MHA layer).
- **MLP (SwiGLU)**: $W_\text{gateproj}$ and $W_\text{uproj}$ are split column-wise (each PE handles a sub-vector of the intermediate dimension); $W_\text{downproj}$ is split row-wise. Each PE computes a sub-vector result independently; the host performs the final addition. Host involvement is limited to two round-trips per layer.

This partitioning reduces inter-PE communication to a minimum while keeping each PE's LUT and quantized data local (maximizing SRAM reuse).

---

## Results

**Platform**: UPMEM PIM-DIMM ×20 (2560 PEs, 64 MB DRAM + 64 KB SRAM per PE); baseline GPU: NVIDIA A6000 (48 GB GDDR6X). Models: Llama2-7b, Llama2-7b-chat, Llama2-13b-hf, Qwen3-32b.

**Accuracy (perplexity, WikiText2)**:

| Model | FP16 | FP8 | MI-LLM |
|---|---|---|---|
| Llama2-7b | 5.69 | 6.19 | 5.81 |
| Llama2-7b-chat | 7.36 | 6.36 | 5.81 |

MI-LLM's perplexity degradation versus unquantized FP16 is 30% of that caused by FP8 quantization. Decoder-level loss quantiles: 90th-pct loss 0.16 (vs FP8: 1.30) on WikiText2 — substantially tighter.

**Throughput (tokens/sec, end-to-end, Fig. 9a)**:

- MI-LLM vs CPU (llama.cpp): 2.12 vs 3.89 tok/s (Llama2-7b) — 2.88× CPU speedup on inference; falls to ~¼ of A6000 GPU throughput in absolute token rate due to process-node gap (DDR4 PIM vs GDDR6X).
- Qwen3-32b could not run on A6000 (memory overflow); MI-LLM ran it at 0.86 tok/s.
- Normalized metric (ALU ticks per output token, Fig. 9b): MI-LLM requires **80% fewer ALU operation ticks per token** than the GPU — the clearest proof of multiplier-free hardware efficiency.

**Kernel-level FLOPS (Fig. 10a)**: MI-LLM achieves 381 GFLOPS on UPMEM (92.9% of theoretical maximum), versus PyGim FP32 at 34 GFLOPS (8.1% of theoretical) and PyGim INT32 at 109 GFLOPS. Versus GPU: 1.09× speedup in kernel throughput.

**Energy efficiency (Fig. 11d)**: 1.11× improvement over GPU (NVIDIA A6000); 2.48× over PyGim on PIM. Power: 12.8 W/DIMM at 400 MHz.

**Kernel speedup breakdown (Fig. 10b)**: base FP32→INT32 replacement: 8.78×; +LUT: 2.19×; +SRAM blocking: 3.08×; final MI-LLM: ~1.09× over GPU kernel throughput.

---

## Contributions

1. First complete LLM inference system running entirely on commodity NBP hardware without CPU involvement in computation.
2. Learning-based LUT construction with STE that achieves non-uniform quantization adapted to each layer's weight distribution — 30% of FP8's perplexity cost.
3. Locality-aware LUT lookup via inter-row and intra-row reordering for both dense and sparse linear kernels; exploits SRAM hierarchy to eliminate DRAM spills during lookup.
4. Distributed model partitioning scheme exploiting MHA head-independence and MLP column/row splits to reduce inter-PE communication to two host round-trips per layer.
5. Empirical characterization of NBP bottlenecks for LLM inference (compute gap, DRAM latency exposure, inter-PE bandwidth) — useful as a platform analysis companion to [[pim-case-study-atc2021]].

---

## Limitations

- **Raw throughput deficit**: absolute token/sec is ~¼ of A6000 GPU; the process-technology gap (DDR4-based NBP vs GDDR6X) limits real-world deployment.
- **Data transfer dominates runtime**: data transfer consumes ~1/3 of total execution time; UPMEM SDK cannot interleave requests across ranks, leaving available transfer bandwidth underutilized.
- **8-bit LUT precision ceiling**: 8-bit indices give 256 quantization levels per operand; finer granularity (e.g., per-channel) is structurally barred by SRAM capacity, capping accuracy recovery.
- **No prefill optimization**: the paper focuses on decode (memory-bound) and does not address prefill throughput, which would require batch matmul and is even more compute-limited on NBP.
- **No MoE or sparse-expert support**: the partitioning scheme assumes dense transformer layers; conditional expert routing (MoE) is not evaluated and would require additional host-side routing logic.
- **Residual/normalization on host**: layer norm and residual additions are still performed on host CPU; this may become a bottleneck at higher batch sizes or longer contexts.
- **Single-model scope**: only Llama2 and Qwen3 families tested; models with different architectural variants (e.g., GQA at large scale, sliding-window attention) are uncharted.

---

## D1–D9 Review Lens

| Dim | Assessment |
|---|---|
| D1 SOTA baseline fairness | Compares against FP8-quantized GPU (strong quantization baseline), PyGim INT32 (best prior NBP kernel), PIM-DL (best prior LUT-on-NBP system), and CPU (llama.cpp); the GPU comparison uses A6000, not A100/H100, which weakens the throughput claim but is consistent with the lab's hardware. |
| D2 Novelty boundary | LUT-for-multiplication is not new (LUT-NN, PIM-DL); the novel contributions are: learning-based quantization value optimization for LUTs, combined inter/intra-row reordering for NBP SRAM-awareness, and the complete end-to-end deployment without CPU arithmetic involvement. Boundary is clearly articulated. |
| D3 Evaluation completeness | Covers accuracy (perplexity, loss quantiles), kernel FLOPS, end-to-end throughput, energy, and per-technique ablation; hardware overhead analysis included. Missing: prefill benchmarks, batch size > 1, attention KV-cache behavior at context length > 1. |
| D4 Verification platform credibility | **Strength**: runs on real UPMEM PIM-DIMM hardware (20 DIMMs, 2560 PEs) — not a simulator. All throughput and energy figures are measured, not modeled. Peak performance is cross-validated against theoretical upper bound (92.9% utilization). |
| D5 Motivation soundness | Memory-wall and multiplier-deficit framing is accurate and well-quantified (3000× FLOPS gap, 0.05 GB/s inter-PE bandwidth). The three NBP challenges each map cleanly to a proposed mechanism. |
| D6 Mechanism cost quantification | LUT construction is offline (one-time cost, no runtime overhead stated). Vector reordering overhead is 0.1% of execution time. Data transfer is ~1/3 of runtime — identified as bottleneck but not fully resolved. Area/power cost of LUT tables vs multiplier hardware not compared. |
| D7 Venue-scope fit | IEEE Transactions on Computers — appropriate; TC regularly publishes PIM/near-memory system papers. The paper has sufficient depth and system breadth for a journal submission. |
| D8 Writing/figure/number self-consistency | Abstract claims 9% throughput and 11% energy improvement over GPU; body and figures confirm these (Fig. 9, Fig. 11). The ALU-ticks metric (Fig. 9b) is clearly defined and well-motivated. Minor inconsistency: abstract says "27% energy improvement" while Fig. 11d / body text says "1.11×" (11%) vs GPU and "2.48×" vs PyGim — the 27% likely refers to the PyGim-normalized or a different configuration; should be disambiguated. |
| D9 Significance/transferable impact | Establishes that commodity NBP hardware can run full LLM inference with reasonable accuracy; the LUT+learning-quantization approach is transferable to any multiplier-free or multiply-scarce hardware substrate. The 80% ALU-tick reduction is a strong transferable metric. The model-partitioning analysis is directly reusable for MoE-on-UPMEM work. |

---

## Connections

### Vault pages — direct links

- [[processing-in-memory-llm]] — situates MI-LLM within the PIM-for-LLM landscape; extends the near-bank PIM cluster.
- [[compute-in-memory]] — LUT-based linear kernels are a form of near-memory computation; the multiplier-free philosophy parallels CIM analog computation.
- [[in-memory-computing]] — NBP is the near-bank variant of in-memory computing; MI-LLM is the first full LLM system in this family.
- [[memory-centric-computing]] — data-centric framing: computation moved to where data lives, eliminating memory wall.
- [[llm-weight-quantization]] — the learning-based LUT construction is a novel non-uniform post-training quantization method; directly extends this concept's space.
- [[on-device-llm-inference]] — UPMEM NBP is a memory-class device; running Qwen3-32b on 20 DIMMs that overflow A6000 is a concrete example of capacity-driven on-device inference.
- [[llm-serving]] — end-to-end decode throughput characterization contributes to LLM serving infrastructure knowledge.
- [[kv-cache-management]] — paper does not address KV cache; but the SRAM capacity constraint on NBP makes this an open gap directly relevant to this concept.
- [[mixture-of-experts]] — MoE support is explicitly absent (a stated limitation); this gap is the motivation for [[moe-upmem-inference]].
- [[pim-case-study-atc2021]] — UPMEM structural limits study; MI-LLM's bottleneck analysis (inter-PE bandwidth, SRAM hierarchy, compute gap) directly confirms and extends those findings on a newer model/workload.
- [[cent-asplos2025]] — CXL-PIM GPU-free LLM serving; alternative near-memory substrate; MI-LLM is a direct comparison point for commodity DDR4 NBP vs CXL-PIM architectures.
- [[cambricon-llm-micro2024]] — flash-PIM LLM; another non-GPU LLM substrate; MI-LLM is the memory-class near-bank PIM counterpart.
- [[neupims-asplos2024]] — NPU+PIM batched LLM inference; different substrate (HBM-PIM with NPU) and batch-focused; MI-LLM is the decode-focused, commodity DRAM-PIM counterpart.
- [[ianus-asplos2024]] — PIM-based attention; MI-LLM addresses the MLP linear path more completely and provides a commodity-hardware deployment complement.
- [[specpim-asplos2024]] — speculative decoding on PIM; complementary acceleration technique that could stack with MI-LLM's LUT kernels.
- [[cxl-pnm-lpddr-hpca2024]] — CXL/LPDDR near-memory compute; adjacent substrate comparison point.
- [[hpim-arxiv2025]] — another PIM-LLM preprint; related work in the same near-bank PIM space.
- [[lincoln-hpca2025]] — PIM-LLM system; related substrate.
- [[papi-asplos2025]] — PIM accelerator; related near-memory inference work.
- [[lp-spec-arxiv2025]] — low-precision speculative decoding; complementary quantization + decoding-acceleration angle.
- [[moe-upmem-inference]] — **primary motivation link**: MI-LLM provides the baseline LUT kernel design, partitioning framework, and perplexity/throughput reference numbers for extending UPMEM deployment to MoE workloads. The model-partitioning analysis (MHA head independence, MLP column/row split) is directly reusable for MoE expert partitioning. MI-LLM's stated gap (no MoE support) is the exact entry point for this idea.
- [[llm-test-time-memory]] — adjacent PIM-LLM idea; MI-LLM's LUT infrastructure could be a compute substrate for test-time memory writes on UPMEM.
- [[pim-llm-pgemmlib-cgo2025]] — closest real-PIM sibling (UPMEM tiled-GEMM LLM library); MI-LLM's LUT-based approach is the accuracy-preserving alternative to GEMM-on-NBP.

### Direct predecessor — vault page now exists

- [[pim-dl-asplos2024]] — **methodological predecessor**: established LUT-NN on commodity DRAM-PIM for DNN inference (BERT/ViT); MI-LLM inherits the LUT-on-UPMEM approach and extends it to full LLM inference (Llama2, Qwen3) with learning-based LUT construction and locality-aware kernel design. MI-LLM cites PIM-DL as [9] (Li et al., ASPLOS 2024).
- [[repa-kvcache-pim-asplos2026]] — sibling ASPLOS paper: reconfigurable PIM jointly accelerating KV-cache offloading + attention compute; head-partitioned locality + sub-batch GPU/PIM pipeline; ASPLOS 2026.

### Cited prior work — no vault page yet
- cites LUT-NN [10] (Tang et al., MobiCom 2022 — LUT with centroid learning for neural inference) — no page yet; the paper MI-LLM most directly extends for LUT construction.
- cites LUT-GEMM [11] (Park et al., arXiv 2022 — quantized matrix multiplication via LUT for large-scale generative models, FP8 weights) — no page yet; key prior LUT-GEMM baseline.
- cites SqueezeLLM [25] (Kim et al., ICML 2023 — dense-and-sparse non-uniform quantization) — no page yet; informs MI-LLM's non-uniform quantization initialization strategy.
- cites PyGim [32] (Gómez-Luna et al., arXiv 2023 — GNN on NBP, characterized bottlenecks) — no page yet; the direct PIM ML baseline MI-LLM benchmarks against.
- cites TransPimLib [33] (Item et al., arXiv 2023 — transcendental functions on NBP) — no page yet; complementary NBP math library.
- cites pLUTo [35] (Ferreira et al., MICRO 2022 — massively parallel LUT computation in DRAM) — no page yet; in-DRAM LUT design that inspired the NBP LUT approach.
- cites NeuPIMs [41] (He et al., ASPLOS 2024) — vault page [[neupims-asplos2024]] exists; confirms the reference.
