---
type: source
title: "Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference"
created: 2026-05-22
updated: 2026-05-22
raw_path: raw/papers/characterizing-mobile-soc-heterogeneous-llm-inference-sosp2025.pdf
source_kind: paper
ingest_level: weak
authors: [Le Chen, Dahu Feng, Erhu Feng, Yingrui Wang, Rong Zhao, Yubin Xia, Pinjie Xu, Haibo Chen]
venue: SOSP
year: 2025
tags: [mobile-soc, heterogeneous-computing, llm-inference, on-device-ai, npu, gpu-npu-parallelism, memory-bandwidth, tensor-partitioning]
---

# Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference

## TL;DR

This paper conducts a systematic characterization of GPU, NPU, and memory subsystem performance on modern mobile SoCs (primarily Snapdragon 8 Gen 3), exposing three NPU idiosyncrasies — stage-sensitive, order-sensitive, and shape-sensitive performance — and one shared memory bandwidth opportunity: no single processor saturates the SoC's 68 GB/s peak, but GPU+NPU together reach ~60 GB/s. Drawing on these observations, the authors build HeteroInfer, an LLM inference engine that schedules operators across GPU and NPU at both layer and tensor granularity with a fast microsecond-level synchronization mechanism. HeteroInfer achieves 1.34×–6.02× end-to-end speedup over GPU-only and NPU-only baselines on billion-parameter LLMs while maintaining negligible interference with co-running GPU applications (§1, §7).

## Key claims

- **NPU stage performance**: due to the fixed systolic-array size (32×32 on Snapdragon 8 Gen 3), tensors not aligned to 32 incur padding overhead; misaligned shapes can reduce NPU utilization dramatically (§3.2, Fig. 3).
- **NPU order-sensitive performance**: reversing tensor dimension order in Matmul can cause up to 6× performance difference on the same NPU because it changes how weights fit the weight-stall paradigm (§3.2, Fig. 4).
- **NPU shape-sensitive performance**: when the input tensor's column size is large relative to the weight tensor, NPU falls to GPU-comparable or worse performance (§3.2, Fig. 4).
- **Memory bandwidth underutilization**: a single processor (CPU, GPU, or NPU) achieves only 40–45 GB/s during decoding workloads; concurrent GPU+NPU raises aggregate bandwidth to ~60 GB/s (out of 61.9 GB/s practical max), enabling memory-bound speedup (§3.3, Fig. 5).
- **GPU synchronization cost**: explicit GPU sync (clFinish) imposes ~400 µs fixed latency per command on mobile platforms, comparable to individual kernel run times — naive GPU+NPU parallelism can therefore hurt (§3.1).
- **HeteroInfer layer-level execution**: Matmul/FFN-down → NPU; RMSNorm/SwiGLU → GPU; order transposition applied to exploit NPU order-sensitive characteristic (§4.1).
- **HeteroInfer tensor-level execution**: weight-centric partition splits weight matrix along row dimension across GPU and NPU subgraphs; static partition ratio determined offline by a profiler-solver (§4.2, Fig. 7).
- **Fast synchronization**: HeteroInfer replaces clFinish with predictable-kernel-waiting-time synchronization, achieving µs-level GPU–NPU sync and 4.01× decoding speedup on Llama-8B from synchronization alone (§4.3, Fig. 17).
- **End-to-end**: first LLM engine to exceed 1000 tokens/s prefill and 50 tokens/s decoding on mobile with high-precision (FLOAT) computation; 1.34×–6.02× over SOTA GPU-only and NPU-only engines (§1, §5).
- **Interference**: running concurrently with League of Legends (GPU-intensive game), HeteroInfer causes only 0.5%–2.2% FPS drop; GPU-only baseline drops to 0 FPS during prefill (§5.7, Fig. 19).

## Why it might matter

This paper is a near-exact methodological parallel to the Phase-0 characterization planned for [[cnn-dnn-edge-memory-wall-metis-embedded]]: both profile the GPU, NPU, and memory subsystem of a commercial mobile SoC to attribute inference bottlenecks, then exploit heterogeneous execution to close them. The workload here is LLM (transformer) rather than CNN/DNN, and the target SoC is Qualcomm Snapdragon rather than Rockchip RK3588, but the core characterization methodology — per-operator profiling, bandwidth saturation analysis, tensor-shape sensitivity study — and the conclusions (no single unit saturates bandwidth; NPU performance highly shape-dependent) are directly transferable. The detailed NPU characterization vocabulary (stage/order/shape sensitivity) and the GPU sync cost measurement protocol offer ready-made analytical primitives for the HPCA'27 work. The [[system-aetina-rkc-a02]] platform's RKNPU2 is a systolic-array NPU like Hexagon; the shape-sensitivity findings likely generalize.

**relevance: high**

## Connections

- [[cnn-dnn-edge-memory-wall-metis-embedded]] — methodological template: same mobile SoC characterization approach (GPU/NPU/memory profiling), same bottleneck attribution goal; LLM workload vs. CNN/DNN but analytical framework is directly applicable to Phase-0.
- [[on-device-llm-inference]] — HeteroInfer is a state-of-the-art on-device LLM inference engine for mobile platforms.
- [[edge-ai]] — targets mobile edge deployment with privacy and latency goals.
- [[llm-serving]] — addresses prefill/decode phase optimization for LLM serving on resource-constrained devices.
- [[memory-centric-computing]] — core insight is that memory bandwidth is the binding constraint in decoding; heterogeneous execution is the vehicle to saturate it.
- [[llm-weight-quantization]] — paper uses W4A16 quantization and explicitly contrasts with INT-only approaches that degrade accuracy (§2.3, §6); accuracy preservation is a design constraint.
- [[system-aetina-rkc-a02]] — the RKC-A02 hosts an RK3588 with RKNPU2 (systolic-array NPU) + Mali-G610; HeteroInfer's NPU characterization findings (stage/order/shape sensitivity of systolic arrays) apply directly to this platform's RKNPU2.
- [[powerinfer2-smartphone-2024]] — cited as a baseline; uses sparse NPU offloading at the cost of accuracy; HeteroInfer claims to outperform it while preserving FLOAT accuracy.
- [[fast-ondevice-llm-npu-asplos2025]] — related prior work on fast on-device NPU-based LLM inference; part of the competitive landscape HeteroInfer addresses.
- [[multi-tenant-heterogeneous-edge-soc-contention]] — §5.7 quantifies GPU performance interference under concurrent gaming workloads; directly relevant to multi-tenant edge SoC contention analysis.
- [[papi-asplos2025]] — server-side counterpart whose dynamic GPU+PIM parallelism scheduling insights extend to mobile SoC characterization.
- [[lp-spec-arxiv2025]] — proposes a concrete LPDDR-PIM mobile-SoC architecture for the LLM target this paper characterizes.
- [[lincoln-hpca2025]] — attacks the same consumer-SoC LPDDR bandwidth wall from the flash-PIM side (HPCA 2025).

introduces *Hetero-layer execution* — no page yet
introduces *Hetero-tensor execution* — no page yet
introduces *weight-centric partition* — no page yet
introduces *fast synchronization via predictable kernel waiting times* — no page yet
- [[cim-centric-llm-mobile-soc]] — **direct methodology template**: HeteroInfer's characterize-each-unit-then-decide-split pattern is adopted wholesale; we extend it to include CIM-as-third-unit on a simulated mobile SoC.
