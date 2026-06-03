---
type: idea
title: "CNN/DNN Edge Memory Wall on Metis (Embedded Board)"
created: 2026-05-22
updated: 2026-05-25
tags: [research-idea, cnn, dnn, sram-cim, metis, edge-ai, memory-wall, hpca-2027, activation-wall, calibration-source]
priority: low
status: calibration-source
demoted: 2026-05-25
demoted_for: cim-centric-llm-mobile-soc
---

> **⚠️ DEMOTED 2026-05-25 — reframed as calibration data source**
>
> This idea was the High-priority main thrust until 2026-05-25, when the user pivoted research direction to **LLM on simulated CIM-enabled heterogeneous mobile SoC**. New High-priority idea: **[[cim-centric-llm-mobile-soc]]**.
>
> **This page is NOT archived.** All Step-1 work and data remain valuable as **L6 end-to-end CNN cross-validation anchor** for the new simulator. Specifically:
> - [[metis-step1-cnn-characterization-2026-05-23]] — full 225-cell Step-1 dataset (5 models × 3 units × batch sweep + 3-mode comparison) is the **only real-silicon end-to-end CNN validation source** for the new simulator
> - A1/A2/A3 sub-experiments planned here map directly to Phase 0 A1/A2/A3 in [[cim-centric-llm-mobile-soc]]
> - The MSI handoff, multi-core handoff, mode 2/3 breakdown handoff defined here remain executable as Phase 0 work for the new idea
>
> **The original HPCA'27 winter cycle deadline (2026-08-01) is also released** with the demotion; the new idea targets HPCA Fall'27 / MICRO'27 / ASPLOS Fall'27 (decided in grill-me 2026-05-25 but specific venue TBD).
>
> Content below remains as historical record + executable calibration plan.

---

# CNN/DNN Edge Memory Wall on Metis (Embedded Board)

**Priority:** Low (demoted 2026-05-25) — **reframed as calibration-source for [[cim-centric-llm-mobile-soc]]**. Previously: HPCA'27 winter cycle main thrust (abstract 2026-08-01) — **deadline released**.
**Workload:** CNN / DNN inference (vision).
**Platform:** Aetina RKC-A02 (RK3588 + Metis Alpha M.2) — see [[system-aetina-rkc-a02]].
**Metrics:** Latency, throughput, L1/L2/DDR traffic, energy, thermal.

## Problem

Activation spill between **L1 (4 MiB/core) → L2 (32 MiB shared) → LPDDR4 (~24 GB/s)** on commercial digital SRAM CIM, in a real edge SoC form factor. CNN/DNN memory wall is **activation-driven** (weight reuse is high; activations flow between layers). The empty cell in the 71-paper PIM/CIM heatmap ([[metis-aipu-nn-v2-2026-05-21]] §5) is exactly "commercial digital SRAM CIM × CNN/DNN at edge form factor," and this idea targets it directly.

## Platform

Aetina RKC-A02:
- Host: Rockchip RK3588 (4×A76 + 4×A55, Mali-G610 MP4, RKNPU2 6 TOPS).
- Memory: 16 GB shared LPDDR4 — also used by CPU / Mali / RKNPU2.
- Accelerator: Axelera Metis **Alpha** M.2 over PCIe Gen3 ×4 (3.9 GB/s host↔device), no on-card DDR.

For the clean baseline measurement the other accelerators (Mali, RKNPU2) are forced idle. Multi-tenant contention is split into a separate idea ([[multi-tenant-heterogeneous-edge-soc-contention]]).

## Decision: Embedded vs Desktop Platform

The embedded board was chosen over the production-class desktop card ([[system-axelera-metis-card]]) as the HPCA'27 main thrust. The trade-off, in HPCA reviewer-style framing:

| Dimension | Desktop (production card + x86 host + PCIe ×16) | Embedded (Aetina RKC-A02, RK3588 + Metis Alpha M.2) |
|---|---|---|
| Memory-wall pressure | PCIe Gen3 ×16 = 15.75 GB/s host↔device; on-card LPDDR4 = 24 GB/s. Host side roomy → bottleneck only inside Metis (L1/L2 → on-card DDR) | PCIe Gen3 ×4 = 3.9 GB/s host↔device; internal still 24 GB/s. Second-tier bottleneck: host LPDDR4 shared by CPU / Mali / RKNPU2 / Metis → genuine edge SoC topology |
| Naturalness of memory-wall narrative | Weak — desktop has wide PCIe, plenty of host RAM, GPU available; "why does Metis starve" sounds contrived | Strong — edge form factor, ~10 W, shared LPDDR4, narrow PCIe. Memory wall is intrinsic to the platform, not an experimental contrivance |
| Metis's standing on the platform | 209 TOPS vs RTX 3090 (187 tok/s vs 15 tok/s on Llama-1B); Metis is a supporting actor; reviewer will ask "why not GPU" | Edge SoC top end (RKNPU2 6 TOPS, Mali ~10 GFLOPS); Metis is the most capable unit for this form factor; measuring its memory wall = measuring this product class's ceiling |
| Reviewer D5/D9 (motivation, significance) | "We benchmark a vendor CIM card" — weak motivation | "Edge memory wall on the leading commercial digital SRAM CIM in real edge SoC" — motivation is intrinsic |
| Reviewer D7 (venue scope) | HPCA workable but invites "why not a PIM benchmark workshop" | HPCA-natural: edge inference architecture; commercial digital SRAM CIM × CNN/DNN is the empty cell in 3-year PIM/CIM heatmap |
| Measurement purity | High — single accelerator, low host noise | Lower by default — but recoverable via baseline-with-others-idle (cgroup cpuset.cpus verified writable) + optional contention ablation |
| Reproducibility | High — M.2 + desktop, easy to source | Medium — Aetina less common but real silicon; measurement framework already in place |
| Thermal envelope | Desktop unimportant (~10 W with no thermal pressure) | Real thermal throttling possible → additional paper angle |
| CNN/DNN spill triggering | Requires batch=32 or high resolution to force L2 spill | Same; plus host LPDDR4 sharing makes second-tier spill easier to trigger |
| Compile-path equivalence | ONNX → Voyager — same | ONNX → Voyager — same; SDK is v1.3.1 on board vs v1.6 on desktop, knob-equivalence smoke test is the first-week go/no-go |
| Decision | **Alt path** (record for future paper) | **HPCA'27 main thrust** |

## Desktop production card — not a companion platform

*Merged 2026-05-22 from `cnn-dnn-memory-wall-metis-desktop`.* The production-class Axelera Metis card (x86 host, PCIe Gen3 ×16, on-card LPDDR4x 1–16 GB — see [[system-axelera-metis-card]]) is **not** a cross-platform companion for this idea. The two platforms have fundamentally different memory architectures — the embedded board exposes a **host-shared LPDDR4** topology while the desktop card has **on-card device memory** — so embedded results cannot be cross-validated against the desktop card, and the desktop card is not a fallback platform either: this work is committed to the on-board **SDK v1.3.1** stack. The desktop card is therefore only relevant as the basis for a possible *standalone* follow-on study (higher-batch / larger-model regimes enabled by on-card LPDDR4x), targeting HPCA'28 / MICRO'27 / ISCA'28 or a TC/TCAD journal extension.

## Experiment Phase 0 — board capability & bottleneck characterization

Before committing to a contribution framing, the first measurement campaign characterizes the board itself: the capability of each compute unit (NPU/CPU/GPU) and where the memory-wall bottleneck appears.

**Compute units to profile** (Aetina RKC-A02):
- **Metis Alpha M.2** (Axelera CIM, primary unit) — via Voyager / ONNX path.
- **RKNPU2** (Rockchip NPU, 6 TOPS) — via RKNN toolkit.
- **CPU** (4×A76 + 4×A55) — ONNX Runtime / TFLite CPU baseline.
- **GPU** (Mali-G610 MP4) — OpenCL / ArmNN baseline.

**What to measure per unit:** latency, throughput, L1/L2/DDR (or host-LPDDR4) traffic, energy, thermal — for the same workload set, so the bottleneck location (compute-bound vs activation-spill vs PCIe host↔device vs shared-LPDDR4 contention) is identified per model.

**Workload set:** edge NN inference models (vision-first). See *Candidate workloads* below — selection pending.

## Candidate workloads (edge NN models)

Survey of NN workloads benchmarked in vault sources, filtered to edge-class vision CNN/DNN (the target of this idea). Selection from this list is pending; the rightmost column flags how well each model exercises the activation memory wall.

### Image classification (primary candidates)

| Model                         | Typical dataset     | Notes / spill behavior                                                                                            | Source                                                             |
| ----------------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| EfficientNet-B0               | ImageNet            | Already runs on this board — **192 FPS, CPU-preprocessing bound at 13.9 ms**; good Phase-0 anchor                 | [[metis-exp-board-rkc-a02-2026-05-18]]                             |
| ResNet-50                     | ImageNet / CIFAR-10 | Canonical CIM benchmark; INT8 quantized; moderate activations, deep enough to spill L2 at higher resolution/batch | [[24-25-in-memory-computing-sram-imc-platform-delta-y1]], NeuroSim |
| ResNet-18 / -34 / -101 / -152 | ImageNet / CIFAR    | Architecture-family sweep — depth scaling vs spill                                                                | [[dnn-neurosim-v1-iedm2019]], [[dnn-neurosim-v2-tcad2021]]         |
| VGG-8 / VGG-16                | CIFAR-10 / ImageNet | Large, dense activations — **strongest L1/L2 spill trigger** among classifiers                                    | [[dnn-neurosim-v1-iedm2019]], [[dnn-neurosim-v2-tcad2021]]         |
| MobileNetV2                   | ImageNet / custom   | Depthwise-separable; low arithmetic intensity → memory-bound even at small size; INT8                             | [[24-25-in-memory-computing-sram-imc-platform-delta-y1-midterm2]]  |

### Object detection / segmentation (high-resolution → spill)

| Model | Typical dataset | Notes | Source |
|---|---|---|---|
| YOLO-family (e.g. YOLOv5/v8) | COCO | High input resolution forces large intermediate activations — natural L2→DDR spill; vendor Voyager zoo includes YOLO | (not in vault sources; vendor/standard edge benchmark) |
| Segmentation (e.g. U-Net / DeepLab) | Cityscapes / COCO | Encoder-decoder skip connections keep activations live across layers — worst case for activation wall | (not in vault sources; standard) |

### Transformer / ViT-edge (secondary — note arithmetic-intensity caveat)

| Model | Task | Caveat | Source |
|---|---|---|---|
| ViT (vision transformer) | ImageNet | Vision transformer keeps CNN-like arithmetic intensity — viable; relevant to ECG-ViT line | [[26-29-listen-to-your-heart-cim-and-accelerator-design-for-ecg]] |
| Llama-3.2-1B INT8 | Token generation | **Decode is memory-bound (~2 ops/byte)** — already characterized on desktop; useful contrast unit, not the main CNN/DNN story | [[metis-llm-investigation-desktop-2026-05-19]] |

## Phase-0 committed workload set

Locked 2026-05-22. Five vision models spanning the arithmetic-intensity / activation-footprint space, plus one memory-bound contrast unit. All exported to ONNX, INT8-quantized, run through the same per-unit profiling on Metis / RKNPU2 / CPU / Mali.

| # | Model | Exact version | Params | Compute (MACs, single image @ native res) | Native input res | Role in the sweep |
|---|---|---|---|---|---|---|
| 1 | **EfficientNet-B0** | timm / torchvision `efficientnet_b0`, ImageNet-1k | 5.3 M | ≈ 0.39 G | 224×224×3 | Board anchor — already runs (192 FPS); validates the pipeline |
| 2 | **ResNet-50** | torchvision `resnet50` v1.5, ImageNet-1k | 25.6 M | ≈ 4.1 G | 224×224×3 | Canonical CIM benchmark; mid arithmetic intensity |
| 3 | **VGG-16** | torchvision `vgg16_bn`, ImageNet-1k | 138.4 M | ≈ 15.5 G | 224×224×3 | Max weight + activation pressure; strongest spill trigger |
| 4 | **MobileNetV2** | torchvision `mobilenet_v2`, width-mult 1.0, ImageNet-1k | 3.5 M | ≈ 0.30 G | 224×224×3 | Depthwise-separable; low intensity → memory-bound even when small |
| 5 | **YOLOv8n** | Ultralytics `yolov8n`, COCO-80 | 3.2 M | ≈ 4.4 G (8.7 GFLOPs) | 640×640×3 | Object detection; high-res input → natural L2→DDR spill |
| C | **Llama-3.2-1B** | INT8, decode | 1.24 B | ≈ 1.24 G / token | seq-len varies | Memory-bound contrast unit (not CNN/DNN main line) |

*Compute convention: "MACs" = multiply-accumulate count; the GFLOPs figure for YOLOv8n is Ultralytics' reported number (≈ 2× MACs). All standard reference values, to be reconfirmed against the actual ONNX graphs on-board.*

**Peak single-image activation footprint** (INT8, largest single intermediate tensor — the quantity that drives L1 4 MiB / L2 32 MiB spill; estimates, to be measured):

| Model | Largest activation tensor | INT8 size | vs L1 (4 MiB) |
|---|---|---|---|
| EfficientNet-B0 | stem 112×112×32 | ≈ 0.38 MiB | fits |
| ResNet-50 | conv1 112×112×64 | ≈ 0.78 MiB | fits |
| MobileNetV2 | first expansion 112×112×96 | ≈ 1.15 MiB | fits |
| YOLOv8n @640 | P1 320×320×16 | ≈ 1.56 MiB | fits |
| VGG-16 | conv1_1 224×224×64 | ≈ 3.06 MiB | borderline |

Single-image tensors mostly fit L1; the wall is reached by **(i) batch** — at batch=32, VGG-16 conv1_1 ≈ 98 MiB overflows L2 → LPDDR4 — and **(ii) resolution** — YOLOv8n at 1280×1280 quadruples every activation. Both are swept axes in Phase-0.

---

# HANDOFF — Phase-0 Step 1: whole-model bottleneck & board capability

> Self-contained experiment spec. Everything needed to execute the first measurement campaign is in this section. Steps 2+ (sweeps, per-layer drill-down, framing decision) depend on Step 1 output and are out of scope here.

## 1. Objective

Characterize the **whole model end-to-end** on the Aetina RKC-A02 board. Two deliverables:

- **(A) Board capability** — how fast and how efficiently each compute unit (Metis / RKNPU2 / CPU / Mali) runs each model.
- **(B) Bottleneck attribution** — for each model×unit cell, *where* the wall-clock time and the energy go: compute, host↔device transfer, CPU pre/post-processing, activation spill, or thermal throttling.

Step 1 does **not** attempt per-layer analysis, knob tuning, or the contribution framing — it produces the data on which those later decisions rest.

## 2. Run matrix

5 vision models × 4 compute units = **20 cells**, plus 1 contrast cell (Llama-3.2-1B on Metis only) = **21 runs**.

|                         | Metis Alpha M.2 | RKNPU2                        | CPU (A76/A55) | Mali-G610 |
| ----------------------- | --------------- | ----------------------------- | ------------- | --------- |
| EfficientNet-B0         | ✓               | ✓                             | ✓             | ✓         |
| ResNet-50               | ✓               | ✓                             | ✓             | ✓         |
| VGG-16                  | ✓               | ✓ (may OOM/fallback — record) | ✓             | ✓         |
| MobileNetV2             | ✓               | ✓                             | ✓             | ✓         |
| YOLOv8n                 | ✓               | ✓                             | ✓             | ✓         |
| Llama-3.2-1B (contrast) | ✓               | —                             | —             | —         |

A cell that will not run is **not** a failure of the campaign — record the reason (unsupported op, OOM, toolchain gap) as a result. Do not force a model onto a unit by altering it.

## 3. Fixed conditions (held constant across all runs)

- **Batch size:** 1.
- **Resolution:** model-native (224×224 classifiers; 640×640 YOLOv8n).
- **Precision:** INT8.
- **Other accelerators idle:** when profiling one unit, the other three are forced idle (CPU governor pinned, `cgroup cpuset.cpus` for CPU isolation — verified writable per [[metis-exp-board-rkc-a02-2026-05-18]]). Multi-tenant contention is a separate idea ([[multi-tenant-heterogeneous-edge-soc-contention]]).
- **Thermal start state:** each run begins from a defined idle-cooled baseline (board temp within X °C of ambient) so thermal results are comparable. Log ambient.
- **Power state:** fixed CPU/GPU/DDR governors (`performance`), recorded.

## 4. Measurement protocol (per cell)

1. **Warmup:** discard first 20 inferences.
2. **Steady state:** measure ≥ 300 inferences (or ≥ 30 s wall-clock, whichever longer).
3. Report **median + p95 + min/max** latency, not just mean. Throughput from steady-state median.
4. **Repeat each cell 3×** (separate cold starts) to expose run-to-run variance; report variance.
5. Log every run as one JSON record (schema in §8) — one append-only results file.

## 5. Metrics (per cell)

| Metric | Definition | How measured |
|---|---|---|
| End-to-end latency | input ready → output ready, single inference | host-side wall clock around the full pipeline |
| Throughput | steady-state inferences·s⁻¹ (FPS) | 1 / median end-to-end latency, batch=1 |
| **Stage breakdown** | time in each of: `preprocess (CPU)` → `H→D transfer (PCIe)` → `device compute` → `D→H transfer` → `postprocess (CPU)` | instrumented timestamps at each stage boundary — **see §6** |
| Power | board-level instantaneous W during steady state | see Pre-flight check P2 |
| Energy / inference | mean W × end-to-end latency | derived |
| Peak temperature | max °C reached during the run | RK3588 thermal sensors (`/sys/class/thermal`); Metis card temp if exposed |
| Throttle | did clocks drop below nominal during the run? y/n + when | clock readback during run |
| Memory traffic | L1 / L2 / DDR (or host-LPDDR4) bytes moved | **if exposed** — see Pre-flight check P1 |

## 6. Harness requirement — stage instrumentation (critical)

The single most important harness property: **preprocessing must be timed separately from inference.** EfficientNet-B0 is already known to be CPU-preprocess bound at 13.9 ms ([[metis-exp-board-rkc-a02-2026-05-18]]); if the harness folds preprocess into the inference number, every bottleneck verdict is wrong.

Required timestamp boundaries, in order:
`t0 input ready → t1 preprocess done → t2 H→D transfer done → t3 device compute done → t4 D→H transfer done → t5 postprocess done`

For CPU/Mali runs the H→D / D→H stages are null (no discrete device) — record as 0 and note it. For Metis the H→D / D→H stages over PCIe Gen3 ×4 are first-class and expected to matter.

**Methodological precedent:** [[characterizing-mobile-soc-heterogeneous-llm-inference-sosp2025]] (SOSP'25) runs essentially this characterization for LLMs on a Snapdragon-class mobile SoC — its NPU pathology vocabulary (stage-/order-/shape-sensitivity), its memory-bandwidth saturation protocol, and its GPU+NPU heterogeneous-execution results are a direct template for this Step-1 profiling and for framing the related work. Borrow its instrumentation discipline; the workload differs (LLM vs CNN/DNN) but the mobile-SoC heterogeneous-unit method is the same.

## 7. Per-unit toolchain

| Unit | Export & quantize path | Runtime | Notes |
|---|---|---|---|
| Metis Alpha M.2 | ONNX → Voyager compile (SDK **v1.3.1**) | AxRuntime | knob-equivalence smoke test (v1.3.1 vs documented behavior) runs in parallel — first-week go/no-go |
| RKNPU2 | ONNX → RKNN toolkit (INT8 quant w/ calibration set) | rknn-runtime | VGG-16 may exceed 6 TOPS NPU comfortably — record fallback |
| CPU (A76/A55) | ONNX (INT8) | ONNX Runtime CPU EP (or TFLite) | thread affinity pinned; record core set used |
| Mali-G610 | ONNX (INT8) | ONNX Runtime OpenCL EP / ArmNN | confirm INT8 OpenCL path is supported per op |

All four INT8 paths use the **same calibration dataset** (small ImageNet / COCO subset, ~200–500 images) so quantization is comparable.

## 8. Deliverables

**D1 — results table** (`model × unit`, one cell each):

| model | unit | lat median/p95 (ms) | FPS | preproc / H→D / compute / D→H / postproc (ms) | power (W) | energy/inf (mJ) | peak °C | throttle | mem traffic | verdict |
|---|---|---|---|---|---|---|---|---|---|---|

**D2 — per-cell bottleneck verdict**, one of: `compute-bound` · `PCIe-transfer-bound` · `preprocess-bound` · `postprocess-bound` · `activation-spill-bound` · `thermal-bound` · `did-not-run (reason)`. Verdict rubric: the stage holding ≥ 50 % of end-to-end latency, cross-checked against memory traffic and clock readback.

**D3 — raw JSON log**, one record per run:
```json
{"model":"", "unit":"", "run_idx":0, "lat_ms":{"median":0,"p95":0,"min":0,"max":0},
 "fps":0, "stages_ms":{"preproc":0,"h2d":0,"compute":0,"d2h":0,"postproc":0},
 "power_w":0, "energy_mj":0, "peak_c":0, "throttled":false,
 "mem_traffic":{"l1":null,"l2":null,"ddr":null}, "ambient_c":0, "notes":""}
```

**D4 — Step-1 findings note:** which unit is the capability ceiling for this board, and the bottleneck distribution across the matrix — this feeds the Step 2 sweep design and the framing decision.

## 9. Pre-flight checks (do these before the 21 runs — they gate result depth)

- **P1 — memory-traffic visibility.** Determine whether Voyager SDK v1.3.1 exposes L1/L2/DDR counters on Metis, and whether RK3588 exposes LPDDR4 traffic counters (DFI / perf events). If **yes**, include `mem_traffic` in every run. If **no**, Step 1 proceeds on stage timing alone and per-layer traffic is deferred to Step 2 (vendor profiler / analytical model). *This check decides how deep the bottleneck attribution can go — do it first.*
- **P2 — power measurement method.** Pick one method and use it for all 21 runs: on-SoC sensors (RK3588) vs external inline power meter (wall / USB-PD meter / shunt on the board's DC input). The Metis M.2 card has no standard per-rail readout — board-level power is the comparable quantity. Energy numbers are only cross-unit-comparable if the method is identical.
- **P3 — quantization parity.** After exporting the four INT8 graphs per model, run an accuracy spot-check on the calibration subset. If top-1 (classifiers) / mAP (YOLO) diverges beyond a small tolerance between units, the units are running different models — fix before benchmarking.
- **P4 — compute-figure reconfirmation.** Recompute params / MACs from the actual on-board ONNX graphs and reconcile against §"Phase-0 committed workload set" reference values.

## 10. Definition of done

Step 1 is complete when: all 21 cells have a result (a measurement or a recorded did-not-run reason); D1–D4 are filled; the three pre-flight checks are resolved and their outcomes recorded. At that point the campaign returns to pick Step 2 sweep priorities and revisit the contribution framing against real data.

---

## Step-1 RESULTS (2026-05-23) — Step 1 closed

Full report: [[metis-step1-cnn-characterization-2026-05-23]] (D4 deliverable). Phase 1a (vendor pipeline, batch=1) 45/45 cells, 0 failures. Phase 1b (direct-tensor harness, batch sweep) 180 cells, 123 good + 57 `did_not_run`. **CPU was not measured in this campaign** — only Metis / RKNPU2 / Mali (gap vs handoff §2; revisit if the CPU floor matters).

**Headline (revised 2026-05-25 per [[metis-step1-cnn-characterization-2026-05-23]] v3):** the bottleneck depends entirely on how the unit is fed. Three completely different regimes per accelerator, **and on Metis three distinct operational modes for filling the 4 AIPU cores — all measured**:

- **Metis** through vendor `inference.py` is **host-pipeline-bound (157–177 FPS flat across a 38× GMACs range)**. Two independent signals confirm: AIPU `inference` element itself is flat at 5.4–8.4 ms regardless of model (ResNet-152: 5.76 ms; MobileNetV2: 5.66 ms — near-identical despite 38× compute difference). That ~5–8 ms is the **fixed per-inference PCIe DMA round-trip floor**; AIPU work is sub-millisecond. Three distinct ways to use 4 cores via direct AxRuntime, measured on MobileNetV2:
  - **Mode 1** (single instance, 1 core): 431 FPS / 2.32 ms — "fed properly but unparallelised" baseline; 2.5× over vendor pipeline alone
  - **Mode 2** (4-instance multi-threading, user-managed): 1147 FPS / 0.87 ms per frame — **only 2.66× over mode 1, not 4×, because 4 small DMAs contend on PCIe Gen3 ×4**
  - **Mode 3** (single instance, compiler-batched=4 via 3-knob YAML, SDK-managed): **1378 FPS / 0.73 ms per frame — wins both, +20 % over mode 2** because one PCIe DMA carries 4 frames per call instead of four small contending DMAs

  **7.9× total gap on the same silicon** (174 → 1378 FPS). The lever is amortising per-call DMA overhead across more frames per call, not adding more host threads.
- **RKNPU2** is latency-bound at batch=1, throughput-scales 5–10× under batching (ResNet-152 50 → 472 FPS at b=32). At b=32 **RKNPU2 beats Phase-1a Metis on every classifier** — sweet spot for offline batched classification.
- **Mali** is memory-bandwidth-saturated for full-model inference: batching gives near-zero scaling and big models exceed the 900-s benchmark timeout. Not a viable full-model unit on this board.

**Toolchain finding — CORRECTED in v3 (2026-05-25):** the earlier "13/15 batched compile fail" conclusion was a **wrong-knob diagnosis, now retracted**. v2 attributed the failure to pre-quantized ONNX shape-dependent ops; the real story is that `input_tensor_shape: [N, ...]` hits a compiler frontend assertion `Only batch sizes of 1 supported`, regardless of graph topology. The **correct mechanism** is the **3-knob YAML override** `extra_kwargs.compilation_config: {aipu_cores: 4, resources: 1.0, multicore_mode: batch}` — which forces the compiler's `singlecore_to_batched_multicore` pass to run on any graph. The two cells that "slipped through" before (ResNet-50, YOLOv8n) had that pass fire automatically because Voyager defaults set `aipu_cores: 4` for their graph shapes; MobileNetV2 / EfficientNet-B0 / ResNet-152 defaulted to `aipu_cores: 1` (Voyager-internal graph heuristic, not a capability question). With the 3-knob override **all 5 models batch cleanly** — verified on MobileNetV2 → 1378 FPS, model.json input `[4, 226, 240, 4]`. **Metis batched compile is NOT broken; the correct knob just isn't obvious.**

**Hard SDK ceiling (v3 new):** intra-frame multi-core parallelism (4 cores collaborate on one frame's compute) is **not implemented in Voyager v1.3.1**. `multicore_mode: cooperative` / `pipeline` are declared in the type system but rejected at validation. **Single-frame latency on Metis can only come down via faster per-call DMA + compute of a single-core model, not via splitting a frame across cores.** For real-time / single-stream scenarios this is THE binding constraint in this SDK generation.

**Pre-flight P2 (power) resolved (negative):** the board has no on-board power instrumentation (no INA, no current sensors). Power and energy intentionally not reported. Bottleneck attribution — the actual objective — does not need power numbers. P1 (memory-traffic counters) and P3 (quant parity) still not reported.

**Step-2 priorities (re-prioritised 2026-05-25 after v3 findings):**

1. **OBSOLETE — multi-core handoff superseded by v3 report.** v3 already executed the 3-mode comparison on MobileNetV2; the multi-core compile path is now characterised. Remaining sub-question: extend mode-3 measurement to the other 4 models (ResNet-50/152, EfficientNet-B0, YOLOv8n) — confirm the 3-knob override unlocks them as the v3 report predicts. ~1.5 h.
2. **Stage breakdown of per-call overhead** — instrumented harness (LD_PRELOAD shim on libze_loader + eBPF kprobes on metis driver) to split the ~5–8 ms DMA floor into: user-space chain / kernel ioctl / DMA descriptor setup / AIPU compute / polling spinwait / output collection. Direct evidence for the round-trip-tax framing. ~1 day.
3. **Productionise mode 3** — turn the 1378 FPS into a deployable application; this is the SDK-managed batched-multi-core path, simplest to ship.
4. **Mali as preprocessing accelerator** — push colorconvert from CPU to Mali OpenCL (or RGA); quantify lift in the vendor-pipeline ceiling.
5. **Memory-usage characterization** — per-model peak resident memory (host LPDDR4 + device 1 GB IOMMU window); mode-2 vs mode-3 memory cost difference (mode 2 has 4 model copies, mode 3 has 1 with batched buffers); batch×model OOM boundary; per-tier traffic (L1 / L2 / DDR) — last item gated by Pre-flight P1.
6. **RKNPU2 EfficientNet collapse** — Swish vs HSwish, depthwise kernel coverage. Low priority.

**Pre-flight P3 (quant parity) implicitly resolved by v3:** shared COCO 2017 200-image calibration subset across all three INT8 flows. P1 (memory traffic counters) still open. P2 (power) resolved as negative (no on-board instrumentation).

**Framing implication (revised 2026-05-25).** The 7.9× gap **with 3-mode decomposition** is direct empirical support for the **discrete-CIM round-trip-tax thesis** (being workshopped in concurrent strategy discussion):
- Mode 1 → Mode 2 (4-instance, user-managed): 431 → 1147 FPS = **2.66×, NOT 4×** — proves user-managed parallelism *cannot* fully escape the round-trip tax because 4 small DMAs contend on the same PCIe link
- Mode 2 → Mode 3 (compiler-batched, SDK-managed): 1147 → 1378 FPS = **+20 %** — proves the lever is *consolidating DMA round-trips*, not adding more host concurrency
- Mode 1 still wins for true real-time single-frame latency (2.32 ms vs mode 3's 2.90 ms per call) — three modes = three workload shapes

Combined with v3's finding that `cooperative` / `pipeline` modes are unimplemented → **the round-trip tax is the only available latency lever in this SDK generation**. This sharpens framing (a) "scheduler beating Voyager defaults" (vendor defaults pick mode 1 for 3/5 models even though mode 3 is strictly better for throughput) and framing (c) "cost-of-openness" (the 3-knob override is undocumented in vendor user-facing docs; required reading `compiler_configs_full.md`).

### Step-2 decisions (2026-05-23, updated 2026-05-25)

- **CPU backfill deferred to the end of the campaign.** Mali is the floor; CPU would be lower still; the matrix hole is acknowledged and accepted for now (do it before submission, not before Step-2). Recordable gap vs handoff §2.
- **Pre-flight P2 resolved (negative):** no on-board power instrumentation. **P3 resolved (positive):** shared COCO 200-image calibration across all units. **P1 (memory counters) still open** — gates Step-2 spill-quantification depth.
- **Multi-core handoff(2026-05-24, given verbatim) is OBSOLETE** — v3 report already executed the 3-mode comparison. The 1378 FPS mode-3 number lands above the multi-core handoff's pre-flight hypothesis. Sub-task remaining: extend to other 4 models, ~1.5 h.
- **Step-2 priority #1 active (user-driven): breakdown of the 174 → 1144 FPS MobileNetV2 gap.** Pull apart `vendor pipe → remove host colorconvert → add 4-instance multi-threading → (if available) batched compile`. Each step a separate measurement; attribute the 6.6× across the three (or four) levers. Output of this breakdown is the dataset that decides framing (a) vs (c).
- User is running this analysis directly; orchestrator on standby for Step-2 handoff writing, related-work scaffolding, or downstream synthesis once breakdown numbers come in.

---

### Step 2 — sweep axes (after Step 1)

- **Batch size:** 1, 2, 4, 8, 16, 32 (classifiers) — walks the L1→L2→LPDDR4 spill boundary.
- **Input resolution:** native, plus 320 / 512 / 1280 for YOLOv8n; 256 / 320 for classifiers.
- **Precision:** INT8 baseline; optional INT4 sweep (smaller activations) per [[awq-lin-2024]] / [[gptq-frantar-2023]].
- **Compute unit:** Metis Alpha M.2 / RKNPU2 / CPU / Mali — same workload, per-unit traffic + energy + thermal.

## Target

- **Abstract**: HPCA'27 — 2026-08-01.
- **Full paper**: HPCA'27 — 2026-08-08 (estimated; confirm).
- **Notification**: HPCA'27 — 2026-11-01 (estimated).

## Connections

- [[sram-imc]]
- [[compute-in-memory]]
- [[memory-centric-computing]]
- [[system-aetina-rkc-a02]]
- [[system-axelera-metis-card]] — alt-platform comparison
- [[metis-aipu-nn-v2-2026-05-21]] — direction report; pivot rationale, two-platform plan
- [[metis-exp-board-rkc-a02-2026-05-18]] — board audit; Q1–Q4 modifiability matrix
- [[cim-weight-changing-large-model]] — adjacent CIM idea (weight-change cost)
- [[multi-layer-computing-analysis-dnn]] — adjacent direction (multi-layer compute analysis)
- [[cnn-dnn-memory-wall-metis-desktop]] — merged into this page 2026-05-22; desktop card retained only as a possible standalone follow-on (different memory architecture, not a companion)
- [[metis-cxl-cim-memory-system]] — CIM + CXL successor
- [[multi-tenant-heterogeneous-edge-soc-contention]] — multi-tenant successor
- [[metis-aipu-full-stack-memory-management]] — archived predecessor
- [[characterizing-mobile-soc-heterogeneous-llm-inference-sosp2025]] — SOSP'25; methodological template for Phase-0 (mobile-SoC GPU/NPU/memory characterization, NPU stage/order/shape pathologies)
- [[metis-step1-cnn-characterization-2026-05-23]] — Step-1 D4 deliverable; closes Phase-0 Step 1 with the 6.6× vendor-vs-direct gap

### Transferable design patterns from PIM/CIM literature

Papers originally studied in a different workload context but whose design ideas are relevant to the CNN/DNN activation-management problem on Metis:

- [[neupims-asplos2024]] — **sub-batch interleaving + dual-row-buffer scheduling**: generalizes to pipelining activation movement across CNN/DNN layers; the idea of overlapping compute and activation transfer maps directly onto Voyager's `double_buffer` / `imc_double_buffer_pipeline` knobs.
- [[ianus-asplos2024]] — **PIM Access Scheduling (PAS) + unified-memory NPU+PIM**: treating the L1 IMC ↔ L2 ↔ LPDDR4 hierarchy as a unified pool with a scheduling policy over it is the right conceptual frame for Metis activation placement — relates to framing (b)/(d) in *Candidate framings*.
- [[awq-lin-2024]] · [[gptq-frantar-2023]] — **activation-aware / one-shot quantization**: INT4/INT8 applies directly to CNN/DNN; quantization × memory-wall interaction (smaller activations fit in L1/L2, reducing spill) is a measurement variable worth sweeping.
- [[cent-asplos2025]] — **CXL + GDDR-PIM hierarchy**: memory-hierarchy composition idea; relevant to [[metis-cxl-cim-memory-system]] sister project where Metis is one tier in a larger CXL-attached CIM stack.
- [[vllm-pagedattention-sosp2023]] — **paged block memory management**: eliminating pre-allocated contiguous activation buffers in favor of demand-allocated blocks is an open question for future Metis/Europa runtime design; not a current Voyager knob, but worth noting as a long-term direction.
