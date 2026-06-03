---
type: concept
title: "LLM Weight Quantization (Sub-Byte)"
created: 2026-05-22
updated: 2026-05-22
tags: [llm-quantization, post-training-quantization, int4, int3, sub-byte, weight-only, memory-wall, on-device-llm]
parents: [on-device-llm-inference]
aliases: [llm-ptq, weight-only-quantization, sub-byte-llm-quantization]
---

# LLM Weight Quantization (Sub-Byte)

Reducing LLM weight precision below INT8 (typically to INT4, INT3, or extreme INT2/ternary) is the **primary lever** for relieving the LLM memory wall. The wall is bandwidth-bound at batch=1 decode (AI ≈ 2 ops/byte for INT8 — see [[metis-llm-investigation-desktop-2026-05-19]] §B): each bit shaved off the weight directly multiplies effective decode throughput.

## Design space

| Axis | Options |
|------|---------|
| **What is quantized** | Weight only (W4A16, GPTQ, AWQ) · Weight + activation (W8A8, SmoothQuant, LLM.int8()) |
| **Training requirement** | Quantization-aware training (QAT) · Post-training quantization (PTQ — used at LLM scale) |
| **Calibration data** | Hessian-based (GPTQ) · Activation-magnitude scaling (AWQ) · Outlier-aware (SmoothQuant) |
| **Salience** | All weights equal (RTN) · Hessian-weighted (GPTQ) · Activation-aware (AWQ) |
| **Output format** | Pure low-bit (GPTQ, AWQ) · Mixed-precision (LLM.int8()) |

## Key papers in vault

| Paper | Bits | Method | Speedup vs FP16 | Notes |
|-------|------|--------|------------------|-------|
| [[gptq-frantar-2023]] | W3/W4 | OBQ + Cholesky + lazy update | ~3.25× A100, ~4.5× A6000 | First PTQ to scale to 175B in 4 GPU hours |
| [[awq-lin-2024]] | W4 | Activation-aware per-channel scaling | 3.2–3.3× across desktop/mobile | MLSys'24 best paper; first 70B on Jetson Orin |

## Why memory wall × quantization is the dominant lever

For batch=1 decode:
- INT8 (Llama-3.2-1B on Metis card): **15 tok/s** at 24.23 GB/s wall ([[metis-llm-investigation-desktop-2026-05-19]] §B).
- INT4 predicted: **~30 tok/s** (vendor-only mitigation noted at §D).
- INT3 (GPTQ extreme): theoretical ~40 tok/s.

The vendor-closed precompile path on Metis ([[metis-llm-investigation-desktop-2026-05-19]] §F) blocks this lever — there is no `axcompile` flag to emit an INT4 LLM artifact, even though the academic methods (GPTQ, AWQ) are open-source and battle-tested.

## Composability

| With | Effect |
|------|--------|
| [[speculative-decoding]] | Stacks: AWQ+EAGLE composable (EAGLE §1). |
| [[kv-cache-management]] | Orthogonal: KV-cache compression handled separately from weight quantization. |
| [[on-device-llm-inference]] | AWQ+TinyChat is the de-facto on-device 4-bit recipe; PowerInfer-2, llama.cpp, MLC-LLM all consume W4 weight files. |
| [[compute-in-memory]] · [[memory-centric-computing]] | Sub-byte datapaths *increase* effective in-array compute density — but only if the PIM/CIM substrate supports them (most academic prototypes assume INT8 or INT16). |

## Connections

[[on-device-llm-inference]] · [[memory-centric-computing]] · [[gptq-frantar-2023]] · [[awq-lin-2024]] · [[microscaling-data-format-llm]] · [[metis-llm-investigation-desktop-2026-05-19]] · [[speculative-decoding]] · [[cambricon-llm-micro2024]] · [[cnn-dnn-edge-memory-wall-metis-embedded]] · [[characterizing-mobile-soc-heterogeneous-llm-inference-sosp2025]] · [[hpim-arxiv2025]] (FP16-only — sub-byte impact on heterogeneous partition unexplored, design gap) · [[mi-llm-multiplier-free-pim-tc2026]] (replaces multipliers with LUT-based operations on UPMEM near-bank PIM — a quantization-adjacent weight-representation technique; IEEE TC 2026) · [[pim-dl-asplos2024]] (LUT-NN on real commodity DRAM-PIM via eLUT-NN + Auto-Tuner; DNN-scoped weight-quantization-via-LUT method for DRAM-PIM; ASPLOS 2024)
