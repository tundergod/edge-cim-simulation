---
type: entity
title: Axelera Metis Card (CIM card + RTX 3090 host)
entity_kind: system
created: 2026-05-19
updated: 2026-05-19
verified: 2026-05-19
sdk_version: voyager-1.6
tags: [edge-ai, ai-accelerator, axelera, compute-in-memory, llm-inference, benchmarking-platform, memory-wall]
---

# Axelera Metis Card (CIM card + RTX 3090 host)

The **"Axelera CIM card"** resource in [[HOME]]: a production-class Axelera Metis quad-core AIPU PCIe card with on-card LPDDR (large enough to hold multi-GB LLM weights), hosted on `wei-tmp-ubuntu` (Ubuntu 24.04.4, kernel 6.8.0-117) alongside an RTX 3090 as the GPU reference. Distinct hardware from the [[system-aetina-rkc-a02|Aetina RKC-A02 platform]], which carries the **Alpha** M.2 module with no real on-card DRAM.

This page records the **LLM-inference bottleneck investigation** (2026-05-18→19, Voyager SDK `release/v1.6`, `axelera-llm==1.6.0`). Headline: Metis batch-1 LLM **decode is a hard on-card-DDR memory wall**, and the LLM path is **precompiled-only / closed-toolchain**, so the card is poorly suited for LLM *architecture* research. See the cross-device synthesis in [[metis-aipu-llm-architecture-research-fit]].

> **Source provenance**: ingested from the experiment folder `xxx2: 3090+aipu/` (raw notes deleted after ingest, per owner request). Reproducible scripts/JSON lived in `voyager-sdk/llm-investigation/`; the durable findings are captured here. Vendor/SDK facts cross-checked against the [Voyager SDK](https://github.com/axelera-ai-hub/voyager-sdk) and [axelera.ai Metis spec](https://axelera.ai/ai-accelerators/aipu/metis).

---

## Hardware / architecture (Metis AIPU)

| Field | Value |
|-------|-------|
| AIPU | **Quad-core** AIPU, proprietary RISC-V + **Digital In-Memory Computing (D-IMC)**; INT8 weights/activations, INT32 accumulate; ~**214 TOPS INT8** total (53.5 TOPS/core); 15 TOPS/W (vendor) |
| Control core | Application-class **RISC-V** running a **closed RTOS** (boot / peripherals / orchestration). **Not user-modifiable** — SDK only ships a flash tool for stock firmware |
| On-chip memory | per-core 1 MiB compute memory + 4 MiB L1; shared 32 MiB L2; **>52 MiB on-chip total** |
| On-card memory | **Board LPDDR4x** — holds full LLM weights (1.2 GB @1B → 7 GB @8B), unlike the Alpha M.2 |
| Clock | 800 MHz (measured) |
| Host link | PCIe **Rev1** test board, **no power telemetry** (all power figures are nameplate-TDP estimates) |
| Host | `wei-tmp-ubuntu`, Ubuntu 24.04.4, kernel 6.8.0-117; RTX 3090 (fp16, HF/CUDA) as GPU reference |

**Host↔device split (measured):** tokenizer / embedding lookup (`embeddings.npz`, vocab 128256 × dim 2048) / sampling / detokenize run on **host**; the `prefill` and `gen` Megakernels run on **device**; PCIe carries embedding vectors ↔ logits. This explains the ~19% host/IO share in E1.

---

## The investigation — why Metis is "slow" on LLMs

Primary model: Llama-3.2-1B-Instruct, precompiled `llama-3-2-1b-1024-4core-static` (4-core, static graph, `max_seq=1024`, weights ≈1.18 GiB INT8). Method-B (assumption-free total-cycle differencing) for decode marginal cost.

### Headline findings

1. **Batch-1 decode is a pure on-card LPDDR weight-streaming memory wall.** Decode time ∝ model weight bytes across 1B→8B, **linear fit r²=0.997**, constant effective DDR bandwidth **≈24.2 GB/s** (CV 4.6% over a ~6× size range). Against the 214 TOPS datasheet peak, decode uses **~0.02%** of compute (≈4600× off peak).
2. **Prefill and decode sit on opposite roofline regions.** Prefill (static padded L=1024) is compute-region (~8.1 TFLOP/s effective, 176× decode); decode is pinned to the memory roof. The true roofline knee (sweep batch, AI≈2·B) is **unmeasurable** — the precompiled artifact is hard batch=1.
3. **Prefill device time is flat vs real prompt length** (static padded graph): llama-1024 slope≈0, R²=0.03; phi3-2048 independent re-check over a 16× span still flat, R²=0.004. Across buckets, prefill ∝ bucket length.
4. **More cores do not fix decode.** 4-core/1-core speedup *decreases* with model size: 1B **1.31×** → 3B **1.20×** → 8B **1.12×** (compute-bound would be a constant 4×). Bigger model → more bandwidth-bound. No 2-core artifact exists (`network.py` asserts `batch==cores`; ships 1c & 4c only).
5. **Metis LLM support is minimal and not Axelera's focus (vendor-confirmed).** Forum (community.axelera.ai, staff "Spanner"): LLM is *experimental*, *must be precompiled*, no public LLM compiler, *"LLMs aren't the primary focus for Axelera."*
6. **Cannot self-compile an LLM.** Public `axcompile` is an ONNX→INT8 **vision/CNN** compiler (31 ops; no LayerNorm/RMSNorm, RoPE, generic Attention, Gather/Embedding, Gelu, dynamic shapes). The prefill+gen dual-kernel autoregressive artifact is produced **only by Axelera's internal toolchain**.

### Key measurements

| Experiment | Result |
|---|---|
| **E0** 3-way throughput (Llama-3.2-1B) | GPU RTX 3090 (fp16) **187 tok/s** · AIPU **15.0 tok/s** · CPU **0.67 tok/s** → GPU ≈12.5× AIPU ≈279× CPU |
| **E1** device vs host/IO (893-tok run) | Metis on-die **80.7%** (53.7 ms/tok) / host+IO 19.3% → decode is device-bound |
| **E2** device-internal IO vs compute | decode **98.5–99.6% on-card DDR weight-stream I/O, 0.4–1.5% MAC** → MAC array ~99% idle |
| **E-B** model-size sweep (decode, 4c) | 1B 53.5 / 2B 85.5 / 3B 122.5 / 8B 310.8 ms·tok⁻¹; r²=0.997; **eff DDR 24.23 GB/s**; device% 81→93 with size |
| **phi3-mini bucket sweep** (512/1024/2048, weights identical) | decode ≈constant across buckets (memory-bound, seq-length independent); eff BW ~20 GB/s (architecture-dependent constant, *not* exactly Llama's 24) |
| **Prefill amortization** | llama-1B/1024 asymptote 14.98 tok/s, 95%-breakeven @64 tok; llama-8B 2.92 @64; phi3-mini/2048 4.69 @128 → bigger bucket/model = lower asymptote, breakeven shifts right |
| **>1024 failure mode** | encoder does not truncate; `stream_response` loop `range(max_tokens − prompt_len)` = empty → 0 tokens, no exception (silent no-op; SDK lacks input validation, **not** our bug) |

### Limits / honest caveats

- Power = nameplate-TDP estimate only (PCIe Rev1 board, no telemetry; AIPU power unmeasurable). Real observation: AIPU keeps host CPU at ~1.6% vs ~100% for the CPU path.
- GPU size-ladder uses **ungated** Qwen2.5 + Phi-3-mini (Llama/Velvet are gated, no HF_TOKEN) → scaling-trend reference, not a same-weights head-to-head.
- B1 int4 projection (decode **~2.04×**, ~31 tok/s; int3 ~2.7×) is **linear extrapolation**, not measured (no int4 artifact in zoo).
- B2 speculative decoding **infeasible** on this SDK: no draft/assisted API; single-token gen kernel forced to shape (1,1).
- batch=1 / no-2-core / RTOS-closed / 24 GB/s mechanism details have **no official forum statement** — verified by own SDK/artifact inspection (forum silent, nothing contradicts).

---

## Strategic read

The card runs LLMs but the *interesting* knobs for architecture research are all behind a closed wall: the memory wall can only be broken by **batching / int4 / shorter-seq buckets / higher-BW Metis M.2 Max** — and **every one of those is vendor-only** because LLM artifacts cannot be self-compiled. What *is* doable today without the vendor: prefill-amortization discipline (one prefill, KV-cache reuse, ≥64-tok outputs) and turning padding into value (stuff few-shot/RAG context up to ~896 tok at zero latency penalty, since prefill is flat). For *architecture* contributions (custom allocator, weight scheduling, multi-model L2 swap, batching), the closed toolchain is the blocker — see [[metis-aipu-llm-architecture-research-fit]].

---

## Connections

- [[system-aetina-rkc-a02]] — the *other* Metis device (Alpha M.2, no on-card DDR); sister investigation (LLM cannot even run there)
- [[metis-aipu-llm-architecture-research-fit]] — unified synthesis: is Metis AIPU viable for LLM architecture research?
- [[compute-in-memory]] · [[in-memory-computing]] · [[sram-imc]] — Metis is digital in-memory compute (D-IMC)
- [[on-device-llm-inference]] — batch-1 edge LLM memory wall, measured on real silicon
- [[memory-centric-computing]] — decode is data-movement-bound, MAC ~99% idle
- [[metis-aipu-full-stack-memory-management]] — idea directly impacted: the full-stack levers are closed
- [[cim-weight-changing-large-model]] — Metis weight-streaming *is* the weight-change-overhead problem, measured (≈24 GB/s wall)
- [[one-token-llm-multi-agent-iot]] — prefill-amortization / PrefillOnly economics quantified here
- [[microscaling-data-format-llm]] — int4 projection (~2.04× decode) motivates low-bit weight formats, but artifact is vendor-only
- [[multi-layer-computing-analysis-dnn]] — Metis as a measured IMC-layer data point in the multi-layer stack
- **[[cim-centric-llm-mobile-soc]]** — **only real CIM-running-LLM anchor** for the new High idea's simulator (Phase 2 L4 cross-validation). Vendor-precompiled FP16 Llama-3 family runs here; on-card 16 GB DRAM ≠ simulator's host-MMIO topology, so this is a bridging anchor not a direct platform match.
