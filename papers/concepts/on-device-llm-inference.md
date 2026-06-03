---
type: concept
title: "On-Device LLM Inference"
created: 2026-05-18
updated: 2026-05-18
tags: [on-device-llm-inference, mobile-inference, edge-ai, model-compression, npu]
parents: [edge-ai]
---

Running large language models (billions of parameters) on consumer devices (smartphones, edge boxes) without cloud servers. The core constraint is device memory: consumer devices have 4–16 GB DRAM, while frontier LLMs require tens to hundreds of GB. Three complementary strategies address this:

## Strategy 0: Sub-byte weight quantization

The most universally applicable lever: shrink weights from FP16/INT8 to INT4/INT3, directly reducing the bandwidth-bound decode time proportionally. See [[llm-weight-quantization]].

- **GPTQ** [[gptq-frantar-2023]]: 3–4-bit one-shot PTQ via OBQ; 175B in ~4 GPU hours.
- **AWQ** [[awq-lin-2024]]: activation-aware 4-bit weight-only quantization; 3–4× speedup; first 70B on Jetson Orin 64 GB.

## Strategy 1: Load from flash/storage on demand

Store model weights in flash (NVMe/UFS) and load only active parameters. Exploits activation sparsity (~90–97% in ReLU models) or expert sparsity (MoE) to reduce per-token I/O.

- **LLM in a Flash** [[llm-in-a-flash-apple-2023]]: windowing + bundling; 4–25× over naive. Apple Silicon. Up to 2× DRAM model capacity.
- **EdgeMoE** [[edgemoe-2023]]: expert-centric memory; non-experts in DRAM, experts on disk; predict-then-preload; 1.19–2.77×.
- **PowerInfer-2** [[powerinfer2-smartphone-2024]]: neuron clusters (dense→NPU, sparse→CPU); I/O-Aware Orchestration; 27.8× over llama.cpp; first 47B on smartphone.
- **Lincoln** [[lincoln-hpca2025]]: LPDDR-interfaced compute-enabled flash + array-shrinking; FFN-Reuse + eager-prediction sparsity; **real-time 50–100B on consumer device** (HPCA 2025).

## Strategy 2: NPU/hardware acceleration

Utilize the mobile NPU (Neural Processing Unit) for dense/regular compute, offloading sparse computation to CPU/GPU.

- **fast-ondevice-llm-npu** [[fast-ondevice-llm-npu-asplos2025]]: chunk-sharing graphs + shadow outlier execution + OoO scheduling; 22.4× prefill speedup; >1000 tok/s prefill on COTS NPU.
- **Cambricon-LLM** [[cambricon-llm-micro2024]]: chiplet NPU + dedicated in-flash-compute NAND for on-device 70B LLM at 3.44 tok/s; 22–45× over UFS-offloading.
- **HeteroInfer** [[characterizing-mobile-soc-heterogeneous-llm-inference-sosp2025]]: GPU+NPU *joint* heterogeneous execution on mobile SoC — layer- and tensor-level operator partitioning + microsecond inter-processor sync; 1.34–6.02× over GPU-only / NPU-only engines (SOSP 2025).
- **HPIM** [[hpim-arxiv2025]]: Heterogeneous SRAM-PIM (attention) + HBM-PIM (FFN/QKV) with intra-token pipelining; targets single-batch latency-sensitive deployment (arXiv 2025).
- **PAPI** [[papi-asplos2025]]: GPU + heterogeneous PIM (FC-PIM HBM + Attn-PIM) with online dynamic kernel scheduling; 1.8× over A100+AttAcc (ASPLOS 2025).
- **LP-Spec** [[lp-spec-arxiv2025]]: LPDDR5-PIM + GEMM augmentation + hardware-aware draft-token pruner; mobile-SoC speculative decode; 13.21× over mobile NPU baseline (arXiv 2025).

## Strategy 4: Speculative decoding for batch=1

The single-user / agentic regime where batching gives no benefit. See [[speculative-decoding]].

- **Medusa** [[medusa-cai-2024]]: multi-head + tree-attention; 2.2–2.8× at batch=1; no draft model needed.
- **EAGLE** [[eagle-li-2024]]: feature-level draft layer; 2.7–3.5×; distribution-preserving.

## Strategy 3: KV cache offloading for long context

Store the KV cache (not model weights) on disk for long-context inference.

- **KVSwap** [[kvswap-ondevice-2025]]: full KV on disk + compact K predictor + group-wise prefetch; 4.1× eMMC, 11× memory reduction vs vLLM.

## Vault research angles

- [[llm-image-generation-mobile]]: edge LLM inference including image-gen — open idea.
- [[one-token-llm-multi-agent-iot]]: ultra-efficient multi-agent IoT LLM inference.

## Connections

[[edge-ai]] · [[llm-serving]] · [[kv-cache-management]] · [[solid-state-drives]] · [[mixture-of-experts]] · [[llm-weight-quantization]] · [[speculative-decoding]] · [[processing-in-memory-llm]] · [[llm-in-a-flash-apple-2023]] · [[edgemoe-2023]] · [[powerinfer2-smartphone-2024]] · [[fast-ondevice-llm-npu-asplos2025]] · [[kvswap-ondevice-2025]] · [[gptq-frantar-2023]] · [[awq-lin-2024]] · [[medusa-cai-2024]] · [[eagle-li-2024]] · [[cambricon-llm-micro2024]] · [[titans-google-2025]] · [[llm-image-generation-mobile]] · [[one-token-llm-multi-agent-iot]] · [[cnn-dnn-edge-memory-wall-metis-embedded]] · [[characterizing-mobile-soc-heterogeneous-llm-inference-sosp2025]] · [[hpim-arxiv2025]] · [[papi-asplos2025]] · [[lp-spec-arxiv2025]] · [[lincoln-hpca2025]] · [[mi-llm-multiplier-free-pim-tc2026]] (multiplier-free LUT LLM inference on real UPMEM near-bank PIM; real-hardware on-device inference baseline; IEEE TC 2026) · [[context-aware-moe-cxl-ndp-arxiv2025]] (prefill-routing oracle for one-shot expert placement on CXL-NDP; resource-constrained device context) · [[pimphony-lolpim-longcontext-hpca2026]] (dynamic KV-cache mgmt for long-context decode on DRAM-PIM; HPCA 2026)
