---
type: source
title: "Step-1 — Edge-Board CNN Characterization Report (Aetina RKC-A02)"
created: 2026-05-23
updated: 2026-05-23
raw_path: raw/notes/metis-step1-cnn-characterization-2026-05-23.pdf
source_kind: note
ingest_level: full
authors: [tundergod]
venue: internal
year: 2026
tags: [metis, rknpu2, mali, aetina-rkc-a02, cnn-edge, bottleneck-characterization, hpca-2027, step-1, phase-0, voyager-sdk]
---

# Step-1 — Edge-Board CNN Characterization Report (Aetina RKC-A02)

D4 deliverable for the Phase-0 Step-1 handoff of [[cnn-dnn-edge-memory-wall-metis-embedded]]. Phase 1a (vendor pipelines, batch=1) 45/45 cells + Phase 1b (direct-tensor harness, batch sweep) 180 cells (123 good + 57 `did_not_run`). Three compute units measured: **Metis Alpha M.2 / RKNPU2 / Mali-G610**. **CPU was not measured** in this campaign (out of scope vs handoff §2; deferred to end-of-campaign per user decision 2026-05-23).

> **Report version note (2026-05-25):** the current PDF is **v3 (23 pages, generated 2026-05-24)**. Major additions over v2: (a) **full Hardware section** — compute-unit + memory-hierarchy tables, with explicit confirmation that ResNet-152 60 MB INT8 > 32 MB L2 → compiler spills weights to host DDR over PCIe (`dpu_constants_home: ddr`); (b) **5-workload detailed reference cards** (paper-quality); (c) **🔥 new "three operational modes" finding** — same MobileNetV2 measured at 431 / 1147 / **1378 FPS** across 3 different ways of using the 4 cores; (d) **major correction to the batched-compile story** — earlier "13/15 fail" was a *wrong-knob* problem, not an SDK limitation; the 3-knob YAML override works on all 5 models; (e) "what does NOT exist in v1.3.1" — intra-frame multi-core parallelism (`cooperative` / `pipeline` modes) is declared in the type system but **not implemented** — meaningful constraint for single-frame latency story. Key claims below reflect v3.

## TL;DR

The bottleneck depends entirely on how you feed the unit. One board, five models, three completely different bottleneck regimes — and on Metis a **7.9× same-hardware throughput gap** (174 → 1378 FPS MobileNetV2) when the AIPU is fed via SDK-managed compile-time batched-multi-core mode (`multicore_mode: batch`, 3-knob YAML override) instead of the vendor pipeline. **Mode 3 wins both throughput and per-frame latency** because it does one PCIe DMA carrying 4 frames per call instead of four small DMAs contending on the link. The 214-TOPS AIPU was not slow; it was starved by per-call DMA overhead.

## Key claims

- **Phase 1a (vendor pipelines, batch=1):** 45/45 cells, 0 failures.
  - **Metis is flat at 157–177 FPS across a 38× GMACs range** (MobileNetV2 0.30 → ResNet-152 11.51) — clean evidence of **host-pipeline-bound** behaviour.
  - **Two independent signals** confirm Metis is *not* compute-bound in Phase 1a: (i) end-to-end FPS flat across 38× compute; (ii) **the AIPU `inference` element itself is flat at 5.4–8.4 ms regardless of model** — ResNet-152 (11.5 GMACs) and MobileNetV2 (0.30 GMACs) post 5.76 vs 5.66 ms, near-identical. That ~5–8 ms is the **fixed per-inference DMA round-trip floor**, not AIPU compute. **The actual AIPU work for these models is sub-millisecond — below the DMA floor.** Phase 1a's "flat across 38× compute" was a real measurement of the wrong thing: it characterised the host CPU's preprocessing pipeline (axtransform-colorconvert0, ~14–21 ms/frame at 640×480), not the AIPU.
  - **RKNPU2 is op-mix sensitive** — EfficientNet-B0 collapses to 22 FPS (depthwise + Swish kernels weak), while other classifiers run 28–105 FPS.
  - **Mali falls cleanly with compute** (compute-bound): 118 FPS MobileNetV2 → 7 FPS ResNet-152, 10–25× slower than Metis.
- **Phase 1b (direct-tensor harness, batch sweep):** 180 cells, 123 good + 57 `did_not_run` (Metis batched compile blocked for most model/batch combos; Mali times out on big batched models).
  - **Metis direct + 4-instance multi-threading at batch=1: 137–1144 FPS, now correlated with model compute.** MobileNetV2 174 → **1144 FPS (6.6×)**; EfficientNet-B0 177 → 550 (3.1×); ResNet-152 173 → 137 (0.8× — vendor was already running it at 4-instance-equivalent throughput).
  - **RKNPU2 latency → throughput transition is dramatic** under batching: MobileNetV2 106 → 681 FPS (6.4×); ResNet-152 50 → 472 (9.5×); EfficientNet-B0 103 → 589 (5.7×). **At batch=32 RKNPU2 beats Phase-1a Metis on every classifier.**
  - **Mali is memory-bandwidth-saturated**: batching gives ~no scaling; big models exceed the 900-s benchmark timeout. Not a viable full-model inference unit on this board at any batch.
- **🔥 v3 correction — batched compile is NOT broken; earlier conclusion used wrong knob.** The previous "13/15 fail" report came from trying batched compile via `input_tensor_shape: [N, 3, H, W]`, which hits a compiler-frontend assertion `Only batch sizes of 1 supported`. The **correct mechanism** is the **three-knob YAML override** in `extra_kwargs.compilation_config`:
  ```yaml
  extra_kwargs:
    compilation_config:
      aipu_cores: 4         # how many cores to compile for
      resources: 1.0        # MUST be ≥ aipu_cores / aipu_cores_max
      multicore_mode: batch # the only working multi-core compile mode
  ```
  All three are required; partial overrides fail with explicit validation errors. This forces the compiler's `axelera.compiler.atex.passes.singlecore_to_batched_multicore` pass — the same pass that fires *automatically* for ResNet-50 / YOLOv8n in vendor defaults but is skipped for MobileNetV2 / EfficientNet-B0 / ResNet-152. **Voyager defaults are graph-shape-dependent and unrelated to whether the graph CAN be batched.** Verified on MobileNetV2: 3-knob override → `compile_config.aipu_cores_used = 4`, `model.json` input shape `[4, 226, 240, 4]`, **1378 FPS** (vs 431 single-core / 1147 user-managed 4-instance).
- **🔥 v3 new — three operational modes for filling 4 AIPU cores** (all measured on MobileNetV2):

  | Mode | Host threads | Instances | AIPU cores | Per-call lat | Per-frame lat | Throughput |
  |---|---|---|---|---|---|---|
  | **(1)** single instance b=1 | 1 | 1 | 1 (25 % util) | 2.32 ms | 2.32 ms | 431 FPS |
  | **(2)** 4-inst b=1 (user-managed) | 4 | 4 | 4 | 3.49 ms (×4 ∥) | 0.87 ms | 1147 FPS (2.66× — **contention**) |
  | **(3)** single inst, compiler-batched=4 (SDK-managed) | 1 | 1 (b=4 input) | 4 (compiler fans out) | 2.90 ms (4 frames) | **0.73 ms** | **1378 FPS (3.20× over (1), +20 % over (2))** |

  **Mode (3) wins both throughput and per-frame latency.** Reason: mode (2) issues 4 small host→device DMAs that contend on PCIe Gen3 ×4 + 4 separate completion-polls in `dma_poll=1` mode; mode (3) issues a single DMA carrying 4 frames + 1 completion. Sub-ms AIPU compute is dominated by per-call overhead → amortising overhead across 4 frames per call is the lever. **Mode (1) still wins for true real-time** (lowest time-to-first-result; mode (3)'s first of 4 frames waits the full 2.90 ms). Three modes = three workload shapes.
- **🔥 v3 new — intra-frame multi-core parallelism does NOT exist in Voyager v1.3.1.** The `multicore_mode` enum has 5 values but only `batch` is real compile-time multi-core. `multiprocess` / `multithread` are host-side replication wrappers (single-core compile). **`cooperative` / `pipeline` are declared in the type system but explicitly rejected at validation as "Unsupported"** — they would have been intra-frame multi-core parallelism (4 cores collaborate on one frame's compute / pipeline layers across cores). **Practical consequence: single-frame latency on Metis can only come down via faster per-call DMA + compute of a single-core model, NOT via splitting that frame across cores.** This is a hard SDK ceiling, not a hardware ceiling.
- **v3 new — ResNet-152 weight spill explicitly confirmed.** 60 MB INT8 weights > 32 MB L2 SRAM → compiler spills weights to host DDR over PCIe (`dpu_constants_home: ddr`). Step-1's observation that ResNet-152 still hits the same ~5.79 ms inference-element time despite the spill implies `double_buffer` / `imc_double_buffer_pipeline` is effectively hiding the streaming behind compute (CNN arithmetic intensity high enough).

## Methodology (per-cell)

- Phase 1a: vendor stock paths — Metis `inference.py` (full GStreamer pipeline: video source → format conversion → resize → normalisation → device inference → post-processing); RKNPU2 `rknn.inference()` on dummy input (no preprocessing in measured path); Mali MNN `MNNV2Basic.out` OpenCL FP16 benchmark, dummy input.
- Phase 1b: custom harness using `axelera.runtime` Python API (Metis: N = 4/batch model instances dispatched from 4 host threads to fill all 4 AIPU cores; pure AIPU compute + DMA, no host preprocessing in measured path); RKNPU2 `rknn.inference()` tight loop with batched dummy input; Mali MNN `MNNV2Basic.out` with batched `inputSize`.
- Common protocol: 20-inference warmup discarded; ≥ 300 inferences and ≥ 30 s steady-state (whichever longer); 3 cold-start repeats per cell; per-unit native precision (Metis/RKNPU2 INT8, Mali FP16 — true INT8 GEMM on Mali GPU not well supported in MNN OpenCL backend); shared calibration data (**COCO 2017 representative subset, 200 images**) across all three INT8 quantisation flows for parity.
- **Power and energy intentionally not reported** (Pre-flight P2 answer): the board has no on-board power instrumentation (no INA, no current sensors). Bottleneck attribution does not need power numbers.

## Bottleneck distribution (D2)

| Cells | Verdict |
|---|---|
| Metis vendor-pipeline × 5 | host-pipeline-bound (Phase 1a) |
| Metis direct b=1 multi-inst × 5 | compute-bound, correlated with GMACs |
| Metis batched (b>1) × 13 | originally recorded as `did_not_run` (wrong-knob path); **v3 corrects: all batchable with 3-knob `compilation_config` override** (mode 3 above) |
| Metis mode 3 (compiler-batched b=4) | 1378 FPS MobileNetV2; +20 % over user-managed mode 2; lowest per-frame latency of all 4-core modes |
| RKNPU2 b=1 × 5 | latency-bound (small fixed per-inference overhead dominates) |
| RKNPU2 b∈{4,16,32} × 15 | throughput-bound; batching scales 5–10× |
| Mali b=1 × 5 | compute-bound |
| Mali batched × 12 | bandwidth-saturated (no scaling) + activation/timeout walls on big models |

## Hardware (v3)

- **Metis Alpha (PCIe card)**: 4 AIPU cores, each with **a 512×512 in-memory matrix-vector unit + a depthwise unit + an LUT for activations**. ~214 TOPS INT8 peak. L1 4 MB per core (activations scratchpad) + L2 32 MB shared (weights home). **No on-card DDR (Alpha)** — "device DDR" = host LPDDR4 via 1 GB PCIe IOMMU window. PCIe Gen3 ×4 ~3.5 GB/s effective.
- **RKNPU2 (on-SoC)**: 3-core NPU @ 1 GHz, 6 TOPS INT8. Shares RK3588's 16 GB LPDDR4 via on-chip AXI fabric — no PCIe in path.
- **Mali-G610 MP4 (on-SoC)**: 4 shader cores, OpenCL 3.0 / Vulkan 1.2; designed for graphics; INT8 NN compute immature; FP16 useful. Shares LPDDR4.
- **Memory hierarchy** (fastest → slowest): Metis L1 SPM (4 MB/core, single-cycle, AIPU core only) → Metis L2 SRAM (32 MB shared, all 4 cores) → Host LPDDR4 16 GB (~40–50 GB/s for CPU/NPU/Mali; Metis sees it via PCIe DMA window ~3.5 GB/s) → eMMC.

**Why these numbers drive the results**:
- Metis 32 MB L2 is the cliff every big batched model hits — ResNet-152 60 MB INT8 already exceeds at b=1, compiler spills to host DDR over PCIe; batched activations on top fill the budget further
- PCIe Gen3 ×4 @ ~3.5 GB/s combined with `dma_poll=1` polling-mode is the "DMA fixed cost" — every Metis device call has a few-ms round-trip floor; dominates AIPU `inference` element time in Phase 1a (~5–8 ms regardless of model)
- RKNPU2 / Mali share LPDDR4 with host, no PCIe to cross — explains RKNPU2 clean batch scaling (no per-call DMA round-trip to amortise) and Mali quickly saturating LPDDR bandwidth on big batches

## Per-unit capability & optimization surface

- **Metis Alpha** — board's compute ceiling by a wide margin once fed properly (**1378 FPS MobileNetV2 via mode 3** vs Phase 1a's 174 = 7.9×). Optimization surface is *how to feed it*: the vendor pipeline (Voyager `inference.py`) is the wrong harness if throughput matters. Three modes for filling 4 cores (see Key claims above) — mode 3 wins both throughput and per-frame latency. **ResNet-152 is the informative counter-case**: at batch=1 single-instance the AIPU has 60M weights to stream per call, so per-call work is substantial and Phase 1a was already near the single-instance compute ceiling for this model (173→137 = 0.8× for user-managed 4-instance). MobileNetV2 is the opposite extreme: light enough that AIPU compute is invisible under DMA overhead, hence 6.6×→7.9× as feeding mode improves.
- **RKNPU2** — best non-Metis option for batched offline classification on this board (b=32 beats Phase-1a Metis on *every* classifier). Has a small fixed per-call overhead (~10 ms equivalent) — no PCIe to cross, purely driver / kernel-launch overhead; batching amortises it cleanly. Optimization surface: maximise batch within memory budget. EfficientNet-B0 batch=1 collapse (22 FPS) persists from the depthwise+Swish kernel weakness; mitigation is HSwish re-export or lean on batching.
- **Mali-G610** — not a viable full-model inference unit at any batch. Its actual role on this board is **preprocessing offload** (OpenCL `resize_cl` element is already in the Voyager pipeline). The realistic Step-2 question for Mali is "how much pipeline ceiling does GPU-side preprocessing buy us?", not full-model throughput.

## Framing implications for Step-2

Step-2 priorities, ordered by leverage:

1. **Productionise the direct-AxRuntime / multi-instance Metis path** — replicate the 1144 FPS MobileNetV2 number in a deployable form (not a benchmark harness). Biggest single throughput lever on the board.
2. **Fix the Metis batched-compile workflow** — via Voyager SDK or pre-batched ONNX upstream. Closes the data gap and may push throughput further.
3. **Mali as preprocessing accelerator** — quantify how much of Phase 1a's host pipeline (colorconvert) Mali can absorb to lift the vendor-pipeline ceiling.
4. **RKNPU2 EfficientNet collapse** — investigate Swish vs HSwish, depthwise kernel coverage. Now lower priority since batching already recovers ~6×.

## Why it might matter

The **7.9× same-hardware gap (174 → 1378 FPS MobileNetV2)** between vendor-default and SDK-managed compile-time batched-multi-core mode (mode 3) is first-hand empirical support for the **discrete-CIM round-trip-tax thesis**: per-call PCIe DMA overhead dominates AIPU latency, and the lever that wins is **amortising that overhead across more frames per call** (mode 3's single DMA / 4 frames). User-managed multi-instance (mode 2) does NOT win because its 4 small DMAs contend on the same link.

Additionally, v3 establishes that **intra-frame multi-core parallelism is not implemented in Voyager v1.3.1** (`cooperative` / `pipeline` modes rejected at validation). This means **single-frame latency on Metis can only come down via faster per-call DMA + compute** — not via splitting a frame across cores. For latency-sensitive scenarios this is a hard SDK ceiling, and the round-trip-tax mitigation story is the only available lever in this software generation.

The "13/15 batched compile fail" earlier conclusion was wrong — it came from using `input_tensor_shape: [N, ...]` instead of the 3-knob `compilation_config` override. **relevance: high** for the HPCA'27 main thrust and for the round-trip-tax research framing now under consideration.

## Connections

- [[cnn-dnn-edge-memory-wall-metis-embedded]] — Phase-0 Step-1 deliverable for this idea; closes Step 1 and re-prioritises Step 2.
- [[system-aetina-rkc-a02]] — empirical capability numbers (board ceiling: Metis ~1144 FPS MobileNetV2 multi-inst; RKNPU2 ~472 FPS ResNet-152 b=32; Mali ≤180 FPS, not viable for full-model).
- [[metis-exp-board-rkc-a02-2026-05-18]] — board audit (Q1–Q4 modifiability matrix); this report instantiates and updates the EfficientNet-B0 baseline.
- [[metis-aipu-nn-v2-2026-05-21]] — direction report; this Step-1 data supplies the empirical baseline the pivot was waiting for.
- [[metis-llm-investigation-desktop-2026-05-19]] — sister desktop-card investigation; different memory regime.
- [[characterizing-mobile-soc-heterogeneous-llm-inference-sosp2025]] — SOSP'25 mobile-SoC heterogeneous characterization; methodological precedent (stage/order/shape-sensitive NPU pathology) — this report is the CNN-on-RK3588+Metis analogue.
- [[system-aetina-rkc-a02]] · [[edge-ai]] · [[compute-in-memory]] · [[memory-centric-computing]]
- **[[cim-centric-llm-mobile-soc]]** — **direct downstream consumer**: this Step-1 dataset (225 cells, 5 CNN × 3 unit × batch sweep + 3-mode comparison) is the **L6 end-to-end CNN cross-validation anchor** for the new High idea's simulator. Phase 0 A1/A2/A3 sub-experiments planned for that idea map directly back to this characterization.
