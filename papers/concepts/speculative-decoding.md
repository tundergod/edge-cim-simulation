---
type: concept
title: "Speculative Decoding"
created: 2026-05-22
updated: 2026-05-22
tags: [speculative-decoding, llm-inference, memory-wall, draft-model, tree-attention, distribution-preserving]
parents: [llm-serving]
aliases: [speculative-sampling, spec-decode]
---

# Speculative Decoding

Speculative decoding accelerates LLM autoregressive decode by **breaking the one-token-per-forward-pass barrier**. The core idea: a cheap draft mechanism proposes K candidate tokens; the expensive target model verifies them all in *one* forward pass via attention masking; accepted tokens advance, rejected ones reset. The mechanism is distribution-preserving when verification uses rejection sampling (Leviathan'22).

The lever is increasing arithmetic intensity. Plain decode at batch=1 forces a full HBM-to-cache weight load per generated token (AI ≈ 2 op/byte for INT8 — measured directly on the Axelera Metis card at 24 GB/s in [[metis-llm-investigation-desktop-2026-05-19]]). Verifying K candidates per forward pass effectively raises decode AI by K×.

## Design space

| Axis | Options |
|------|---------|
| **Draft source** | Separate small model (Leviathan'22, Chen'23) · Multi-heads on base (Medusa) · Single extra transformer layer (EAGLE) · N-gram + Jacobi iteration (Lookahead) |
| **Draft granularity** | Token sequence (classic) · Token tree (Medusa, EAGLE) |
| **Draft conditioning** | Token-level (most) · Feature/hidden-state level (EAGLE) |
| **Verification** | Rejection sampling (strict; distribution-preserving) · Typical acceptance (temperature-thresholded; faster, slight drift) |
| **Distribution preservation** | Strict (vanilla spec-decode, EAGLE) · Greedy-only (Medusa-1, Lookahead) · Approximate (Medusa-2, typical acceptance) |

## Key papers in vault

| Paper | Method | Speedup | Distribution |
|-------|--------|---------|--------------|
| [[medusa-cai-2024]] | Multi-head + tree attention; no draft model | 2.2–2.8× | Greedy-strict / non-greedy approximate |
| [[eagle-li-2024]] | Feature-level draft layer + shifted-token conditioning | 2.7–3.5× | Strict (greedy + non-greedy) |
| [[specpim-asplos2024]] | DLM+TLM on PIM hardware via architecture-dataflow DSE | 1.52×/2.02× vs GPU/PIM | — (hardware-level) |
| [[lp-spec-arxiv2025]] | LPDDR5-PIM + GEMM augmentation + draft-token pruner (mobile) | 13.21× vs mobile NPU, 12.83× EDP vs AttAcc-PIM | — (hardware-level) |
| [[papi-asplos2025]] | Speculative-decode TLP tracked by online scheduler on GPU+PIM | 1.8× over A100+AttAcc | — (hardware-level) |

## Practical implications

- **Composable with quantization** (AWQ + EAGLE, GPTQ + EAGLE): stack the levers.
- **Composable with KV-cache management** (vLLM PagedAttention + spec-decode): orthogonal.
- **Implementation dependency**: spec-decode needs custom verification kernels (tree-attention or rejection-sampling kernels). This is *blocked by closed-source vendor compile paths* — directly relevant to [[metis-llm-investigation-desktop-2026-05-19]] §F (axcompile cannot self-compile generic Attention) and §D where spec-decode is listed as a "vendor-only mitigation."
- **Batch=1 dominance**: the regime where classic batching gives no benefit; spec-decode is the only software-level lever.

## Connections

[[llm-serving]] · [[on-device-llm-inference]] · [[medusa-cai-2024]] · [[eagle-li-2024]] · [[specpim-asplos2024]] · [[lp-spec-arxiv2025]] · [[papi-asplos2025]] · [[llm-weight-quantization]] · [[processing-in-memory-llm]] · [[metis-llm-investigation-desktop-2026-05-19]] · [[one-token-llm-multi-agent-iot]] · [[cnn-dnn-edge-memory-wall-metis-embedded]]
