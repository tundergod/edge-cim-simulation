---
type: source
title: "Metis LLM Investigation on Desktop (Production Card + RTX 3090)"
created: 2026-05-22
updated: 2026-05-22
raw_path: raw/notes/metis-llm-investigation-desktop-2026-05-19.html
source_kind: note
ingest_level: full
authors: [Wen-Sheng Lim]
year: 2026
tags: [metis, llm-inference, memory-wall, sram-cim, decode, prefill, voyager-sdk]
---

# Metis LLM Investigation on Desktop (Production Card + RTX 3090)

## TL;DR

End-to-end LLM characterization (2026-05-19) on a production-class Metis card hosted in an RTX 3090 box (Voyager SDK v1.6, axelera-llm 1.6.0). Headline: Llama-3.2-1B runs at 15 tok/s on Metis vs 187 tok/s on the RTX 3090 host (12.5× slower). Decode is **purely memory-bound** at an effective **24.23 GB/s** on-card LPDDR4 bandwidth (decode time ∝ weight bytes, r²=0.997, CV 4.6%); MAC ~99% idle per decode step. Prefill is compute-bound (8.12 TFLOP/s lower bound, AI≈2005 FLOP/B), flat vs prompt length inside a static padded bucket. Every mitigation (int4, batching, shorter-seq buckets, M.2 Max) is **vendor-only** — no public LLM self-compile path.

## Key claims

- **Headline gap** (§A): Llama-3.2-1B at 15 tok/s on Metis vs 187 tok/s on RTX 3090 host (12.5× slower).
- **Decode is memory-bound** (§B, Figure B2, Table B1): decode-step time scales linearly with model weight bytes, r²=0.997; **effective DDR bandwidth = 24.23 GB/s, CV 4.6%**.
- **MAC ~99% idle per decode step** (§B.3): compute is not the bottleneck; the AIPU is starved.
- **4-core decode scaling = 1.32× of ideal** (§B.4): shared on-card LPDDR4 is the contention point.
- **Prefill is compute-bound** (§C, Table C1): 8.12 TFLOP/s lower bound at AI ≈ 2005 FLOP/B; flat vs prompt length within a static padded bucket (R²=0.03 for the slope).
- **Short prompts pay full prefill cost** (§C.2): the static padded bucket means a 64-token prompt costs the same prefill device-time as an 896-token prompt — turn padding into value.
- **Vendor-only mitigations** (§D): int4 weights predict 2× decode speedup; batching predicts proportional throughput gain; Metis M.2 Max ~2× BW. Each requires a new precompiled artifact.
- **Owner-side discipline (vendor-free)** (§E): prefill amortization (one prefill, KV reuse, ≥64-token outputs); context-stuffing up to bucket cap at zero added latency.
- **Self-compile path blocked** (§F): `axcompile` is ONNX→INT8 for vision/CNN only; no LayerNorm/RMSNorm, RoPE, generic Attention, Gather/Embedding, dynamic shapes.

## Motivation

The team needed a sharp, quantified answer to "is Metis a viable LLM-architecture research platform?" Earlier qualitative impressions ("LLM is slow on Metis") were insufficient for a strategic decision. This investigation isolated decode vs prefill, characterized the bandwidth ceiling, and tested every plausible vendor-free mitigation.

## Method

- **Workload**: Llama-3.2-1B INT8 (vendor-provided precompiled artifact via axelera-llm 1.6.0).
- **Sweep**: model size (filtered Llama-3.2 variants), prompt length (32 → 1024 tokens within and across static bucket boundaries), output length (1 → 256 tokens), core count (1, 2, 4), batch (1).
- **Telemetry**: per-step latency from axelera-llm; MAC utilization from Voyager profiler; effective on-card DDR BW computed as (weight bytes touched / decode-step time).
- **Statistics**: linear regression of decode time vs weight bytes (r², CV); orthogonal AI vs prompt-length slope test (R² for slope).
- **Reference**: same Llama-3.2-1B run on the host RTX 3090 via vLLM for a sanity gap.

## Results

- 24.23 GB/s is a *hard* memory wall on the production-class card: independent of model variant, prompt length, batch=1 output count.
- Prefill flatness inside the bucket is the one piece of latency the owner can *use*: stuff context up to the bucket cap free of charge.
- The KV-cache reuse + prefill-amortization economics make Metis tolerable for **PrefillOnly / one-token agent** patterns where decode is short or absent — but not for chat workloads where decode dominates.
- Every architectural lever a paper would touch (custom allocator, weight-placement scheduling, multi-model L2 swap, batching, low-bit datapaths) is sealed behind the vendor precompile.

## Contributions / What's reusable

- The 24.23 GB/s figure is now the canonical measured reference for Metis production-class decode bandwidth — feeds [[cim-weight-changing-large-model]] and any future cross-platform CIM memory-wall study.
- The prefill-flatness finding is a system-level optimization the owner can apply without vendor involvement — recorded as "owner-side discipline" for future LLM-on-Metis vision-pipeline auxiliary use.
- The measurement harness (sweep + regression + AI calculation) is reusable for any production-class CIM card and is the basis for the CNN/DNN desktop measurement plan ([[cnn-dnn-memory-wall-metis-desktop]]).

## Limitations / open questions

- Single model (Llama-3.2-1B) measured in depth; broader workload sweep (Phi, Qwen, Gemma) would strengthen the generality claim but does not change the wall.
- Whether the wall is identical on Metis M.2 Max (higher LPDDR4 BW) is unverified.
- Whether `axcompile` for LLM operators is on Axelera's public roadmap is not confirmed.
- The 187 tok/s RTX 3090 reference is a single configuration; not exhaustive but adequate for the 12.5× gap headline.

## Connections

- [[sram-imc]]
- [[compute-in-memory]]
- [[memory-centric-computing]]
- [[system-axelera-metis-card]] — entity page for the desktop card
- [[system-aetina-rkc-a02]] — sister embedded platform
- [[metis-aipu-nn-v2-2026-05-21]] — v2 direction report that demotes this data to appendix-only
- [[metis-exp-board-rkc-a02-2026-05-18]] — sister embedded-board audit
- [[metis-aipu-llm-architecture-research-fit]] — synthesis thread (LLM closure)
- [[cim-weight-changing-large-model]] — directly measured here
- [[microscaling-data-format-llm]] — int4 prediction (~2.04× decode) motivates low-bit formats
- [[one-token-llm-multi-agent-iot]] — prefill-amortization quantified
- [[cnn-dnn-memory-wall-metis-desktop]] — alt-path successor that reuses this measurement harness
- [[hpim-arxiv2025]] — simulator-only competitor (no silicon calibration); the D4 gap that our Metis silicon measurements (this report + Step-1) fill
- **[[cim-centric-llm-mobile-soc]]** — **direct downstream consumer**: this Metis Card LLM characterization (Llama-3.2-1B @ 15 tok/s on Metis card, decode 24.23 GB/s memory wall) is the **L4 end-to-end LLM cross-validation anchor** for the new High idea's simulator. Cross-platform companion: Metis Card has on-card 16 GB DRAM (vs simulator's host-MMIO topology) — bridging assumption noted in idea page Method section.
