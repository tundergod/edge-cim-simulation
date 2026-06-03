---
type: source
title: "HPIM: Heterogeneous Processing-In-Memory-based Accelerator for Large Language Models Inference"
created: 2026-05-25
updated: 2026-05-25
raw_path: https://arxiv.org/abs/2509.12993
source_kind: paper
ingest_level: full
authors: [Cenlin Duan, Jianlei Yang, Rubing Yang, Yikun Wang, Yiou Wang, Lingkun Long, Yingjie Qi, Xiaolin He, Ao Zhou, Xueyan Wang, Weisheng Zhao]
venue: arXiv
year: 2025
tags: [pim, sram-pim, hbm-pim, llm-inference, heterogeneous-pim, simulator, memory-centric-computing, single-batch, intra-token-parallelism, hardware-software-codesign]
---

# HPIM: Heterogeneous Processing-In-Memory-based Accelerator for Large Language Models Inference

## TL;DR

HPIM proposes the first memory-centric heterogeneous PIM architecture pairing SRAM-PIM (latency-critical attention) with HBM-PIM (weight-intensive GEMV/FFN) for single-batch LLM inference. A four-stage hardware-aware compiler partitions workloads and orchestrates a tightly-coupled pipeline that exploits fine-grained intra-token parallelism to break the serial dependency in autoregressive decoding. The system is evaluated via a cycle-accurate simulator (DRAMsim3-extended for HBM-PIM + custom Verilog HDL for SRAM-PIM) across the OPT-350M to OPT-30B family at FP16. Peak speedup is 22.8× over A100 GPU, 1.50× over IANUS, and up to 5.76× over CXL-PNM in tokens/second.

## Key claims

- **Heterogeneity is necessary**: a single PIM substrate cannot simultaneously satisfy the diverse bandwidth, capacity, and latency requirements of LLM inference; SRAM-PIM's ultra-low latency is optimal for Q×K^T / S×V, while HBM-PIM's high capacity and bandwidth are optimal for QKV/FFN weight-intensive GEMVs (§I, §III).
- **Intra-token parallelism**: the key scheduling insight — overlapping K-vector generation in HBM-PIM with transpose-to-K^T and Q×K^T in SRAM-PIM eliminates idle cycles that prior monolithic PIM designs leave on the critical path (§IV-C, Fig. 10b).
- **22.8× peak speedup over A100**, average 6.2× across OPT-350M–30B; 3.1×/2.4×/2.6× latency reduction vs A100 for OPT-6.7B/13B/30B at (256-input, 768-output) (Fig. 11).
- **1.50× speedup over IANUS** at (256-input, 512-output); advantage grows with sequence length as decoding phase dominates (Fig. 12).
- **5.76× higher throughput over CXL-PNM**: HPIM exploits HBM3 internal bandwidth; CXL-PNM is bottlenecked by external CXL interconnect (Fig. 12).
- **Layer-wise decoding breakdown** (OPT-13B, 1K output): QKV generation 3.74× speedup (4538→121 ms relative to A100 reference), projection 4.64×, FFN 2.99×, attention compute 3.81× — all effectively masked by deep pipelining (Fig. 13, §V-C).
- **Decoding is 73.8% of OPT-13B execution time** even at modest input/output ratios, validating the single-batch target (§II-A).

## Motivation

LLM deployment at single-batch (edge/latency-sensitive) inference suffers three compounding problems: (1) enormous weight footprints that saturate external memory bandwidth during autoregressive decoding GEMV; (2) diverse operational characteristics — GEMM for prefill, GEMV for decode, nonlinear ops (softmax, GELU, LayerNorm) all requiring different compute and memory properties; (3) serial token-level dependencies that prevent inter-batch parallelism and make naive time-multiplexed PIM designs wasteful.

Prior heterogeneous NPU+PIM designs (NeuPIMs, AttAcc, IANUS) target high-throughput batched inference and cannot exploit intra-token parallelism. Single-substrate near-memory approaches (CXL-PNM, TransPIM) are limited either by interconnect latency or memory capacity. The authors identify that "a single memory technology cannot simultaneously satisfy the diverse requirements" — motivating a clean split: SRAM-PIM for latency-critical attention computation, HBM-PIM for weight-heavy linear projection and FFN.

## Method

### Architecture overview

HPIM is a co-designed hardware–software stack with three hardware components and a four-stage compiler.

**SRAM-PIM subsystem**: 32 cores, each with:
- Tensor Compute Unit (TCU) — 64×64 systolic array for prefill GEMM
- Vector Compute Unit (VCU) — element-wise ops (softmax exponential, GELU, square root)
- Scalar Compute Unit (SCU) — scalar arithmetic and control
- PIM Unit (16 macro groups × 8 FP16 multipliers per macro) — in-SRAM attention compute
- 384 KB activation memory; dedicated transpose and transfer units for inter-module data movement
- High-throughput Network-on-Chip linking all 32 cores; unified 32-bit ISA with 3-stage fetch–decode–execute pipeline

**HBM-PIM subsystem**: 4 HBM3 modules (8-die × 24 Gb/die via TSVs); each die has 2 channels × 2 pseudo-channels × 8 bank groups × 4 banks. Lightweight MAC units (1 PU/bank, 16 FP16 multipliers/bank) embedded at bank level — minimal DRAM integration complexity. Two modes: standard read/write and compute mode (simultaneous weight/activation streaming + partial-sum buffering).

**Centralized controller / hardware instruction scheduler**: orchestrates cross-subsystem synchronization and task sequencing.

### Compiler (four stages)

1. **Operator analysis and annotation** — tags each graph node as GEMV, GEMM, or nonlinear.
2. **Stage-specific mapping** — prefill: all GEMM dispatched to TCU in SRAM-PIM; decode: workload partitioned between SRAM-PIM and HBM-PIM for concurrent execution.
3. **Hybrid tiling** — head-wise parallelism (HP) for multi-head attention assignment; tensor-wise parallelism (TP) for spatial matrix partitioning across channels/banks.
4. **Code generation** — produces per-subsystem instruction streams with synchronization and prefetching directives.

### Scheduling strategy

**HBM-PIM mapping (decode)**: Q/K/V weight matrices partitioned head-wise (each head → dedicated HBM channel), then column-wise tiling + channel-wise interleaving for TP. Other FC layers use row-wise slicing across channels. Row-wise mapping aligns with native HBM access granularity.

**SRAM-PIM mapping (decode)**: when heads = cores (e.g., 32-head model on 32 cores), each head maps to one core for localized attention. When heads exceed cores, computation spans execution phases. When cores exceed heads, intra-head TP distributes across cores with All-Gather for softmax normalization recovery.

### Pipeline (decode phase — the critical innovation)

The decoding pipeline overlaps two independent execution paths:
- **HBM-PIM path**: K vector generation → V vector generation → projection/FFN weight-intensive GEMVs
- **SRAM-PIM path**: transpose K → K^T → Q×K^T → softmax (VCU, distributed max/sum via All-Gather) → S×V

Key overlap: K generation in HBM-PIM is concurrent with transpose-to-K^T in SRAM-PIM; Q×K^T computation overlaps with V generation. This is the primary source of the latency advantage over IANUS, which serializes these operations.

## Simulator methodology

This is the critical section for our work — HPIM is simulator-only with no silicon validation.

**SRAM-PIM simulator**: custom Verilog HDL-based RTL simulator. Models each core's TCU, VCU, SCU, PIM Unit, and the NoC. Cycle-accurate. No mention of calibration against real SRAM-CIM silicon or published validation against a hardware prototype. No error-bound or sensitivity analysis reported.

**HBM-PIM simulator**: extended DRAMsim3 with:
- Bank-level compute units (MAC timing and energy)
- PIM-aware scheduling (differentiating compute mode vs. read/write mode)
- HBM3 timing and power parameters (presumably from HBM3 JEDEC specifications or vendor data)
No mention of calibration against real HBM-PIM silicon (e.g., SK Hynix AiM, Samsung HBM-PIM). Validation is internal consistency only — no comparison to measured hardware throughput or energy.

**What is modeled**: memory timing, bank-level compute latency, inter-subsystem data transfer, pipeline synchronization.

**What is abstracted / not reported**: area breakdown, thermal modeling, chip-to-chip interconnect latency details, PVT variation, actual fabrication cost or process node, real nonlinear op latency vs. assumption.

**Configuration (Table IV)**:
- SRAM-PIM: 32 cores, TCU (64×64 PEs, 1 MAC/PE), VCU, PIM Unit (16 MGs, 8 FP16 multipliers/macro), 384 KB activation memory
- HBM-PIM: 4 modules, HBM3 (8 die, 24 Gb/die), 1 PU per bank with 16 FP16 multipliers

**Summary of D4 posture**: pure simulation, no hardware prototype, no calibration against real silicon, no error bars. This is the most significant vulnerability for reviewers and the clearest gap for our work.

## Results

| Comparison | Metric | Value |
|---|---|---|
| vs. A100 GPU | Average speedup | 6.2× |
| vs. A100 GPU | Peak speedup | 22.8× |
| vs. A100 (OPT-6.7B, 256 in / 768 out) | Latency reduction | 3.1× |
| vs. A100 (OPT-13B, 256 in / 768 out) | Latency reduction | 2.4× |
| vs. A100 (OPT-30B, 256 in / 768 out) | Latency reduction | 2.6× |
| vs. IANUS (256 in / 512 out) | Speedup | 1.50× |
| vs. CXL-PNM | Throughput (tokens/s) | up to 5.76× |

Layer-wise OPT-13B decoding (1K output tokens, vs. A100 absolute timing):
- QKV generation: 4538 ms → 121 ms (3.74×)
- Attention projection: 1832 ms → 395 ms (4.64×)
- FFN layers: 7902 ms → 2646 ms (2.99×)
- Attention compute: 4862 ms → 1285 ms (3.81×)

Models evaluated: OPT-350M, 1.3B, 2.7B, 6.7B, 13B, 30B under FP16, batch size = 1.

## Contributions

1. First memory-centric heterogeneous PIM architecture integrating SRAM-PIM and HBM-PIM in a unified LLM inference accelerator.
2. Hardware-aware compiler with hybrid tiling (HP + TP) for adaptive workload partitioning across heterogeneous substrates.
3. Tightly-coupled decoding pipeline exploiting intra-token parallelism to overlap SRAM-PIM attention with HBM-PIM weight-load, breaking the autoregressive serial bottleneck.
4. Cycle-accurate evaluation framework extending DRAMsim3 + Verilog HDL simulation across OPT model family.

## Limitations / open questions

- **No silicon validation (D4)**: entirely simulation-based; neither SRAM-PIM nor HBM-PIM simulator calibrated against measured hardware. Results could diverge significantly from real deployment.
- **FP16 only**: no sub-byte (INT8, INT4, INT3) evaluation. Modern edge and mobile LLM inference has largely moved to 4-bit or mixed precision; the FP16 assumption may overstate weight footprints and understate achievable throughput on quantized models.
- **Single-batch only**: the prefill phase is dispatched entirely to SRAM-PIM TCU — batched inference (which relaxes the memory-bound constraint) is not addressed and would require a different partitioning strategy.
- **No energy / area analysis**: the paper reports performance speedups but provides no energy breakdown, area overhead of PIM integration, or power efficiency comparison. D6 is completely unaddressed.
- **OPT family only**: OPT is a 2022-vintage model family. Evaluation on LLaMA-3, Mistral, Qwen, Phi, or Gemma — models with GQA, RoPE, SwiGLU — would stress-test the scheduler's generalizability claims.
- **No attention to KV cache growth**: as sequence length grows, KV cache can approach or exceed SRAM-PIM storage capacity. The paper does not analyze SRAM-PIM capacity limits under long-context workloads.
- **HBM3 is assumed, not commercially available for edge/mobile**: HBM3 is an expensive, high-power, large-form-factor memory technology; the system configuration is closer to a server-class accelerator than an edge deployment, despite single-batch being framed as edge/latency-sensitive.
- **Baselines limited to A100, IANUS, CXL-PNM**: NeuPIMs, CENT, PAPI, SpecPIM, Cambricon-LLM are mentioned in related work but not quantitatively compared.

## D1–D9 review lens

| Dim | Assessment | Score |
|---|---|---|
| **D1 Baselines** | Compares vs. A100, IANUS, CXL-PNM — but NeuPIMs, CENT (CXL+PIM all-PIM), PAPI, Cambricon-LLM are named in related work yet absent from evaluation. The strongest same-regime competitor (CENT, arXiv 2024) is not compared. Missing quantitative comparison to several stated contemporaries is a clear D1 weakness. | Weak |
| **D2 Novelty** | Heterogeneous SRAM+HBM PIM is a clear and well-motivated delta from prior monolithic or NPU-centric PIM work. The intra-token pipeline overlap is the concrete novel mechanism. The delta from IANUS (which also does NPU+PIM partitioning but without the SRAM-PIM attention substrate) is stated. Well-bounded. | Strong |
| **D3 Evaluation** | Only OPT family at FP16, batch=1. No ablation on the impact of the pipeline overlap vs. simpler scheduling. No sensitivity to sequence length extremes, model architecture variants (GQA/SwiGLU), or quantization. No multi-batch results. Narrow workload coverage for the breadth of the claim. | Weak |
| **D4 Platform** | Pure simulation. DRAMsim3-extended + Verilog HDL, no calibration to real HBM-PIM silicon (SK Hynix AiM, Samsung HBM-PIM), no calibration to any SRAM-CIM prototype, no error bounds. This is the paper's deepest vulnerability at architecture venues. | Very weak |
| **D5 Motivation** | The motivation is well-constructed: roofline analysis, decoding-phase dominance (73.8%), and the diversity of LLM operation requirements are all quantified. The argument for why heterogeneity is necessary — not just helpful — is crisp. | Strong |
| **D6 Mechanism cost** | No area overhead, no energy breakdown, no power analysis. For a paper targeting "low-latency single-batch" deployment the complete absence of energy/area numbers is a major gap. A reviewer would flag this immediately. | Very weak |
| **D7 Venue** | arXiv preprint — no venue gate to evaluate. As a submission target, this fits DAC / MICRO / HPCA / ISCA scope; the D1 and D4 weaknesses would likely be the primary rejection risks. | N/A |
| **D8 Consistency** | Paper appears internally consistent; results tables and figures align with abstract claims. The 22.8× peak is appropriately contextualized as best-case (short input / long output ratio). No apparent contradictions identified. | Strong |
| **D9 Significance** | The heterogeneous PIM framing is genuinely broader than single-substrate prior work and the intra-token pipeline mechanism is transferable. However, the FP16-only, OPT-only, simulation-only scope limits the breadth of the impact claim. If validated on real hardware, significance would be high. | Mid |

## Why it matters to us

HPIM is the closest published competitor to our planned CIM-on-PCIe + MMIO unified memory + cross-validated Metis silicon direction. It establishes (a) that the community has arrived at SRAM-PIM + HBM-PIM heterogeneity as the right architectural answer for single-batch LLM inference, and (b) that the current state of the art remains entirely simulation-only with no silicon calibration — exactly the gap our Metis-cross-validated simulator fills. Our differentiators are not marginal engineering details; they are structural: we have real silicon (Axelera Metis AIPU as a validated digital SRAM-CIM reference point), a PCIe-native integration path, and MMIO-unified memory rather than a custom point-to-point architecture. HPIM's D4 weakness is our D4 strength. Critically, HPIM's FP16-only, OPT-only, energy-blind evaluation leaves quantization-aware inference, energy efficiency, and modern model architectures (GQA, SwiGLU) as open ground. **Relevance: high.**

## Differentiators we can claim vs HPIM

- **Real silicon cross-validation**: our simulator is calibrated against measured Axelera Metis AIPU behavior on real LLM decode workloads ([[metis-llm-investigation-desktop-2026-05-19]]), giving us D4 credibility HPIM entirely lacks. This is the single largest structural advantage.
- **PCIe-native integration path**: HPIM assumes a custom monolithic SoC with tightly-coupled SRAM-PIM and HBM-PIM. Our CIM-on-PCIe design targets an incremental deployment path compatible with existing server and mobile SoC infrastructure — a more realistic deployment scenario for near-term adoption.
- **MMIO unified memory**: HPIM maintains a partitioned memory view with explicit inter-subsystem data movement. Our MMIO-unified approach eliminates the mapping overhead and enables a unified programmer model across the CIM and external memory hierarchy.
- **Quantization-aware evaluation**: HPIM evaluates FP16 only. We can characterize INT8 / INT4 / mixed-precision decode performance, which is more representative of production edge deployment and changes the relative advantage of bandwidth-limited vs. compute-limited substrates.
- **Modern model coverage**: OPT is a 2022 family. Characterizing LLaMA-3, Mistral, Qwen with GQA and SwiGLU tests whether the head-wise partitioning strategy generalizes (GQA reduces head count; this could invalidate HPIM's 1-head-per-core mapping at 32 cores for models with 8 GQA heads).
- **Energy and area analysis**: HPIM provides no energy breakdown. Our work can provide energy-per-token measurements from real power instrumentation on the Metis card, filling the D6 gap HPIM ignores entirely.
- **Mobile SoC context**: HPIM uses HBM3 (expensive, high-power, data-center-class). Our platform targets mobile SoC form factor with appropriate memory technology constraints, making the edge/on-device claim credible where HPIM's is aspirational.
- **Heterogeneous interface characterization**: we can measure and model the MMIO latency between CIM and DRAM tiers, which HPIM abstracts away in its centralized controller model.

## Connections

- [[processing-in-memory-llm]] — HPIM is a primary exemplar of the SRAM-PIM + HBM-PIM heterogeneous substrate design pattern; adds intra-token pipeline overlap as the key new mechanism
- [[memory-centric-computing]] — HPIM self-describes as "memory-centric"; directly instantiates the design principle
- [[on-device-llm-inference]] — single-batch target aligns with edge/on-device deployment, though HBM3 choice is server-class
- [[llm-serving]] — autoregressive decode latency is the primary metric; relevant to serving-system framing
- [[in-memory-computing]] — SRAM-PIM in-array compute is the SRAM-CIM instance of this concept
- [[compute-in-memory]] — the SRAM-PIM subsystem's macro-group compute is CIM in the strict sense
- [[sram-imc]] — HPIM's SRAM-PIM with 16-macro-group per core, 8 FP16 multipliers/macro is a digital SRAM-IMC instance
- [[kv-cache-management]] — KV cache resides in SRAM-PIM; capacity limits under long context are an unaddressed open question
- [[llm-weight-quantization]] — FP16-only evaluation is a gap; sub-byte quantization impact on the heterogeneous partition is unexplored
- [[neupims-asplos2024]] — prior NPU+HBM-PIM design; HPIM replaces NPU with SRAM-PIM for attention and adds intra-token parallelism
- [[ianus-asplos2024]] — closest prior work and primary performance baseline (1.50× HPIM advantage); IANUS uses unified NPU+PIM memory but serializes attention and weight-load
- [[cent-asplos2025]] — CXL+GDDR-PIM all-PIM design; framed as related work but conspicuously absent from quantitative comparison; occupies the same single-batch niche
- [[cambricon-llm-micro2024]] — NPU+flash-PIM design using die-stacking; mentioned in related work context as a different heterogeneous approach
- [[metis-llm-investigation-desktop-2026-05-19]] — our Metis silicon investigation; provides the real-hardware D4 anchor that HPIM lacks
- [[llm-test-time-memory]] — HPIM's SRAM-PIM substrate is a plausible host for test-time parametric writes; the intra-token pipeline model is directly applicable to test-time-write scheduling on heterogeneous memory
- [[llm-in-a-flash-apple-2023]] — flash-offload baseline for single-batch inference; positioned as the prior bandwidth-bottleneck framing that PIM-based approaches aim to supersede
- **[[cim-centric-llm-mobile-soc]]** — **closest competitor / direct prior art**; HPIM is the primary differentiation target for the new idea page. HPIM's D4 gap (simulator-only, no silicon calibration, FP16-only, no energy/area) is exactly what real-Metis cross-validation + INT8/INT4 + mixed-precision are designed to fill.
