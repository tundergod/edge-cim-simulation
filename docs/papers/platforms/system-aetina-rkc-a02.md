---
type: entity
title: Aetina RKC-A02 Experimental Platform
entity_kind: system
created: 2026-05-18
updated: 2026-05-19
verified: 2026-05-19
power_verified: 2026-05-18
sdk_version: voyager-1.3.1
tags: [edge-ai, ai-accelerator, axelera, rockchip, rknpu, compute-in-memory, benchmarking-platform, llm-inference, alpha-silicon]
---

# Aetina RKC-A02 Experimental Platform

An edge-AI carrier board pairing the Rockchip RK3588 SoC with an Axelera Metis Alpha PCIe AI accelerator card. Two independent accelerator stacks, two separate toolchains, on a single board. Used as the primary hardware platform for AI-acceleration experiments.

---

## Identification

| Field | Value |
|-------|-------|
| Board | Aetina RKC-A02 v0.4.2 (based on Rockchip RK3588-EVB7 LP4 v10 reference) |
| Hostname | `aetina` |
| Class | RK3588 edge-AI carrier + Axelera Metis **M.2** AI accelerator (1 GiB module) |
| Silicon maturity | **Alpha** (Metis card is pre-production; behavior/perf may diverge from production Metis) |

> **Corrected 2026-05-18 (on-device `axdevice`)**: the Metis accelerator is the **Quad-Core Metis M.2** module (form factor `m2`, board controller type `ortles`), routed to the host over PCIe Gen3 ×4 — *not* a 4 GB/16 GB PCIe card. This is a **vision-model platform**.
>
> **Deeper correction 2026-05-19 (disassembly + on-device experiment)**: the "**1 GiB**" reported by `zeDeviceGetMemoryProperties` is **not real on-card DRAM** — this **Alpha** M.2 has **no on-card DDR at all**, only **32 MB L2 SRAM** (BAR2, `0x900000000–0x901FFFFFF`). The "1 GB" is a **PCIe IOMMU window mapping host LPDDR4** to a device-visible address range. ⚠ **This contradicts the vendor M.2 datasheet** ("1 GB dedicated DRAM"): the spec describes *production* Metis M.2; **Alpha silicon differs**. Production Metis has 4–16 GB on-card DDR; this difference is the entire reason LLM compute is impossible here (see *LLM-on-Alpha* below). Cross-device synthesis: [[metis-aipu-llm-architecture-research-fit]].

---

## Compute — Primary AI Accelerator (Axelera Metis)

| Field | Value |
|-------|-------|
| Device | Axelera AI Metis — Alpha, **1 GiB Quad-Core M.2 module** (form factor `m2`, board type `ortles`) |
| PCI ID | `1f9d:1100` at `01:00.0`; class `0x120000` Processing Accelerators |
| PCIe link | Gen3 (8.0 GT/s) × 4 lanes — full width/speed negotiated |
| MSI / DMA mode | `dma_poll=1` (polling DMA) used; interrupt-mode times out at 2 s. **Community-confirmed on this exact platform** — multiple Axelera forum posts report MSI / DMA timeouts on Aetina + Metis M.2 combos (e.g. "Aetina board with M2 Metis: error timeout for querying an inference"). Specific mechanism uncertain (IRQ routing on this carrier BSP, MSI vector handling, or `metis-dkms` interaction). **Re-verify once with `dma_poll=0` + raised `dma_timeout` on our exact box** to confirm we are seeing the same fault community sees, not a stale local config. |
| AIPU | Quad-core AI Processing Unit; **~214 TOPS INT8** (digital in-memory compute); 4 cores @ 800 MHz, mvm 100% |
| On-chip SRAM | per-core 4 MB L1 SPM + **32 MB shared L2 SRAM** (BAR2) — weights home (`pool_l2_const`) |
| **On-card DRAM** | **None** (Alpha silicon, verified 2026-05-19). `zeDeviceGetMemoryProperties` "1 GB" = PCIe IOMMU window into host LPDDR4, *not* chip memory. Production Metis: 4–16 GB on-card DDR |
| Firmware | runtime/firmware **v1.3.1**; flver 1.3.0; board controller bcver 1.4 |
| Thermal | sw throttle 200 °C (rate 12 %, hyst 5 °C); hw throttle 105 °C (hyst 10 °C); PVT warning 95 °C |
| Power | **~8–15 W typical** (vendor app range, 1-chip card); energy efficiency **~15 TOPS/W INT8** |
| Driver | `metis` kernel module via `metis-dkms` v1.0.2 (source `/usr/src/metis-1.0.2`) |
| Device nodes | `/dev/metis-0:1:0`, `/sys/class/metis/`, MSI IRQs 159/160 |
| SDK | Axelera Voyager SDK under `/opt/axelera` (+ `/axelera`); Aetina launcher `/home/aetina/start_axelera_aetina.py` |
| Management | Onboard FTDI UART (`usb-Axelera_AI_Metis_Alpha_PCIe_Card_*`) for console/recovery |

**PCIe bandwidth ceiling**: Gen3 ×4 ≈ 3.9 GB/s usable host↔card — monitor for high-resolution / high-FPS pipelines.

---

## Compute — Host SoC (Rockchip RK3588)

| Field        | Value                                                                                                                        |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| SoC          | Rockchip RK3588 (8 nm)                                                                                                       |
| CPU          | Octa-core big.LITTLE: 4× Cortex-A76 ≤2.3 GHz + 4× Cortex-A55 ≤1.8 GHz; ARMv8.2-A aarch64; FP16 + INT8 dotprod SIMD           |
| Built-in NPU | RKNPU2 — 3-core, **6 TOPS INT8** (INT4/INT8/INT16/FP16); driver v0.8.2 @ 1.0 GHz; runtime `librknnrt` 1.6.0; `rknn_server` active. No vendor per-block power; whole-SoC envelope applies. |
| GPU          | ARM Mali-G610 MP4 (Valhall arch 10.8.6) ≤1.0 GHz; OpenCL / Vulkan / OpenGL ES; `/dev/mali0`. ARM/Rockchip publish no per-block TDP. |
| 2D engine    | RGA 2.1.0 — resize / CSC / crop offload for pre-processing                                                                   |
| Video codec  | Rockchip MPP — H.264/H.265/VP9/AV1 hardware decode + encode                                                                  |

---

## Memory & Storage

| Component | Spec |
|-----------|------|
| RAM | 16 GB LPDDR4 (≈15 GiB usable); **no swap configured** |
| eMMC | 32 GB (mmcblk0, SDHCI ADMA): 14 GB root (8.8 GB free) + 14.8 GB `/userdata` + boot/firmware |
| NVMe/M.2 | None — PCIe lane consumed by Metis card |

> **No swap** is a risk for memory-heavy workloads. Add a swapfile on `/userdata` before heavy model loading if needed.

---

## Software Stack

| Component | Version / State |
|-----------|----------------|
| OS | Ubuntu 22.04.4 LTS (Jammy), aarch64 |
| Kernel | 5.10.110 (Rockchip BSP / Firefly build, Jun 2024) |
| Python | 3.10.12; **no AI packages installed** (no torch / onnx / rknn-lite / numpy) |
| Docker | Present (`docker0` bridge) |
| Voyager SDK | **1.3.1** confirmed working; Docker image `axelera-sdk-ubuntu-2204-arm64:1.3.1`; SDK framework 63 MB; model weights pulled on demand from `media.axelera.ai` |
| RKNN runtime | `librknnrt` 1.6.0; `rknn_server` active (enables remote/PC inference) |
| Bootloader | U-Boot, eMMC boot; verified-boot state: **orange** |

---

## Accelerator Comparison

| | Axelera Metis | RKNPU2 |
|---|---|---|
| Peak INT8 | **~214 TOPS** | 6 TOPS |
| Power | ~8–15 W typical (own PCIe card) | No per-block figure; shares RK3588 SoC budget (whole SoC ~5–6 W light, board ~15–25 W under load) |
| Efficiency | ~15 TOPS/W INT8 (vendor) | not separately published |
| Interface | PCIe Gen3 ×4 | On-SoC |
| Toolchain | Axelera Voyager SDK (ONNX/PyTorch → Metis) | RKNN-Toolkit2 → `librknnrt` |
| Silicon state | Alpha | Production |
| Use case | High-throughput primary target | Low-power fallback / comparison baseline |

---

## Voyager SDK & Model Zoo

**Voyager SDK 1.3.1** — verified working, run non-interactively from Docker image `axelera-sdk-ubuntu-2204-arm64:1.3.1`.

**Deployment note**: the SDK framework is only 63 MB; weights are pulled on demand from `media.axelera.ai`. Extracted to `/userdata/voyager-sdk/` (symlinked from `/home/aetina/voyager-sdk`).

**Detached container** (Claude-controllable, no TTY required):
```bash
# Start
~/start-sdk-bg.sh
# Run commands
docker exec axelera-sdk bash -c 'cd /home/ubuntu/voyager-sdk && source venv/bin/activate && <cmd>'
# Run inference
docker exec axelera-sdk bash -c 'cd /home/ubuntu/voyager-sdk && source venv/bin/activate && python3 inference.py <network> --no-display --show-stats'
```
Required env vars: `-e GID=0 -e USERNAME=root -e SRC_DIR=/home/ubuntu/voyager-sdk` (entrypoint.sh requirement).

**106 deployable networks** across 8 categories. Each model typically has two variants: native framework (`-coco`) and pre-exported `-onnx`.
### Object Detection — YOLO (38)

`yolo11{n,s,m,l,x}-coco` · `yolov3` · `yolov5{n,s,m,l}-v7` · `yolov5s-v5` · `yolov5s-relu` · `yolov7` / `yolov7-tiny` / `yolov7-640x480` · `yolov8{n,s,m,l}` · `yolov9{t,s,m,c}` · `yolox-{s,m}`

### Instance Segmentation (8)
`yolo11{n,l}seg-coco` · `yolov8{n,s,l}seg-coco` (+ onnx variants)

### Pose / Keypoint (8)
`yolo11{n,l}pose-coco` · `yolov8{n,s,l}pose-coco` (+ onnx variants)

### Classification — ImageNet (26)
`resnet{18,34,50,101,152}` · `efficientnet_b0–b4` · `mobilenetv2` · `squeezenet1.0/1.1` (torchvision) + from timm: `mobilenetv4_{small,medium,large,aa_large}` · `resnet10t`

### Semantic Segmentation — mmlab (2)
`unet_fcn_256-cityscapes-onnx` · `unet_fcn_512-cityscapes`

### TensorFlow Object Detection (2)
`ssd-mobilenetv1-coco-poc-onnx` · `ssd-mobilenetv2-coco-poc-onnx`

### Other Vision — torch (5)
`fastdepth-nyudepthv2` (depth) · `lprnet` (license plate) · `real-esrgan-x4plus` (super-resolution) · `retinaface-mobilenet0.25` / `retinaface-resnet50` (face detection)

### LLMs — Metis static graphs (7) — ⚠️ NOT RUNNABLE on this card (proven, see below)

> **No LLM static graph can run on this Alpha M.2.** Not merely a card-size shortfall: Alpha has **no on-card DDR**, so the AIPU compute kernel can only address weights through the 1 GB PCIe-IOMMU device window. The *load* limit was surmounted from user space (model fully loads); the *compute* limit is a closed-firmware hard wall (`-1301`). Listed for reference / future-hardware planning only.

| Model                            | Context  | Cores | Min Metis card |
| -------------------------------- | -------- | ----- | -------------- |
| `phi3-mini-512-static`           | 512 tok  | 1     | 4 GB           |
| `phi3-mini-1024-4core-static`    | 1024 tok | 4     | 16 GB          |
| `phi3-mini-2048-4core-static`    | 2048 tok | 4     | 16 GB          |
| `llama-3-2-1b-1024-4core-static` | 1024 tok | 4     | 4 GB           |
| `llama-3-2-3b-1024-4core-static` | 1024 tok | 4     | 4 GB           |
| `llama-3-1-8b-1024-4core-static` | 1024 tok | 4     | 16 GB          |
| `velvet-2b-1024-4core-static`    | 1024 tok | 4     | 4 GB           |

Naming convention: context length + core count in the slug. `-4core` = all 4 Metis AIPU cores fused. These are pre-compiled static graphs; no Voyager re-compilation needed. Card-memory requirements per Voyager v1.3 `model_zoo.md` (verified 2026-05-18).

### Reference Application Pipelines (12)
Multi-model cascades and parallel runs: `yolov8sseg→yolov8lpose`, `yolov8spose+yolov8n-weapons`, `parallel-yolov8spose-retinaface`, `yolov8n→yolov8s`, barrel/perspective-distortion variants, CES2025 demos, `fruit-demo`. Tutorials: `t1-simplest-onnx`, `cifar10`, `resnet34-caltech101`.

---

## Metis Runtime Health (verified at setup)

All checks healthy. Card never yet driven (clean baseline).

| Check | Result |
|-------|--------|
| PCIe link | Gen3 8.0 GT/s × 4 — full negotiated speed/width. `pcie_set_speed.sh` available to force Gen3 if it ever down-trains. |
| Kernel driver | `metis.ko` v1.0.2 loaded via DKMS (`/lib/modules/5.10.110/updates/dkms/`) |
| sysfs version | `/sys/class/metis/version` = 1.0.2 — matches driver |
| Device node | `/dev/metis-0:1:0`; mode `crw-rw-rw-`, group `axelera` — world-accessible; SDK container can open without privilege escalation |
| PCI state | `1f9d:1100` enabled (`enable=1`), power state **D0** |
| MSI interrupts | 32 vectors allocated (`msi-metis-0:1:0-0..31`); all counts = 0 (card idle, never exercised this boot) |
| Running processes | None — no Metis/Voyager process or container active |

### Driver tunables (defaults; relevant for DMA/latency benchmarking)

| Tunable | Default | Notes |
|---------|---------|-------|
| `dma_poll` | 0 | polling vs interrupt-driven DMA completion |
| `dma_timeout` | 2 | DMA timeout in seconds |
| `irq_timeout` | 1 | IRQ timeout in seconds |
| `single_msi` | 0 | use all 32 MSI vectors (not single-vector mode) |
| `enable_dmabuf_sync` | 1 | explicit DMA-buf synchronisation enabled |

---

## Power Envelope

| Block | Power | Source / confidence |
|-------|-------|---------------------|
| Axelera Metis (1-chip PCIe) | **~8–15 W typical**, ~15 TOPS/W INT8 | Vendor-published "typical application" range — high confidence |
| RK3588 SoC (whole chip) | **~5–6 W** light typical; component blocks not broken out | Vendor/secondary; Rockchip does not publish per-block TDP |
| Full board under load | **~15–25 W** (Orange-Pi-class RK3588 boards: idle ~10 W, load ~14 W) | Third-party board measurements — indicative, this carrier not yet measured |
| RKNPU2 / Mali-G610 individually | **No published per-block power** | ARM/Rockchip do not publish these |

**Implications**:
- Metis is the only block with a trustworthy power spec. Its **~15 TOPS/W** is the headline efficiency number and the reason it is the primary target despite Alpha silicon.
- The Metis card draws its 8–15 W *on top of* the host SoC over PCIe — total platform power ≈ board (~15–25 W loaded) **+** Metis (~8–15 W). Size the PSU / thermals for the sum, not either alone.
- RKNPU2 vs Mali-G610 power can only be obtained by **on-board measurement** (rail/INA sensor or system-power delta while looping a fixed workload). No paper number exists to cite.

> Metis 4-chip PCIe variant (not this board) is 30–58 W for ~856 TOPS — listed only to bound the product family.

---

## Key Experimental Constraints

0. **1 GiB Metis M.2 — vision only**: onboard DRAM is 1 GiB; **no Voyager LLM static graph fits** (needs ≥4 GB card). Plan LLM-on-Metis experiments around a larger module; this card is for vision-model work (YOLO / ResNet / segmentation / pose).
1. **Alpha silicon**: pin Voyager SDK version; avoid upgrading mid-experiment series.
2. **Clean Python**: provision the runtime stack yourself — good for controlled benchmarking, but requires explicit setup before first run.
3. **PCIe bandwidth**: ~3.9 GB/s host↔Metis ceiling; monitor for large activations / high-FPS video pipelines.
4. **No swap**: set up a swapfile on `/userdata` before loading large models.
5. **Storage budget**: root has 8.8 GB free; install AI packages and models to `/userdata`.
6. **`rknn_server` active**: the board can be driven remotely from a PC for RKNPU2 inference and profiling without SSH.

---

## LLM-on-Alpha — conclusive investigation (2026-05-19)

Tested `llama-3-2-1b-1024-4core-static` (smallest LLM, ~1.26 GB total weight pools). The problem splits into **load** and **compute**:

- **Load path — surmounted from user space ✅.** `ze_shim2.c` (LD_PRELOAD): (1) intercept `zeMemAllocDevice`→`zeMemAllocShared` (host LPDDR4, no 1 GB cap; symbol resolution must use `dlopen("libze_loader.so.1", RTLD_NOLOAD)`, **not** `RTLD_NEXT` — SDK dlopens libaxruntime into a LOCAL namespace → segfault otherwise); (2) **stateful copy interception** — `libaxeDriver.so axeShareMemoryExecute` (disasm `0x23dc0`) is a pure PCIe-DMA executor that rejects `shared→shared`, so track shared-pool ranges and `memcpy`+return SUCCESS in `zeCommandListAppendMemoryCopy` instead of appending a DMA command. Result: **all 18 pools (incl. 953 MB) allocate; model fully loads** (`AxInstance initialized`) — previously impossible.
- **Compute path — closed-firmware hard wall ❌.** Inference immediately fails: `axeLoadAndExecuteQueueElf: Wait kernel failed -1301`. Disassembly proved the entire wait chain (`triton_pcie_wait_kernel`→…→`axl_read32` raw PCIe read) is pure status relay; `-1301` is a raw value the closed RISC-V firmware writes at kernel setup. **Not a timeout** (`LIBUIO_TRITON_KRN_TV_SEC` 2→600 s: identical immediate failure). The `.axnet` queue ELF's baked-in weight addresses must resolve inside the 1 GB device window; host-shared addresses (`0x7f…`) are invalid to the compute kernel. No host-side lever — fixing it needs the closed RISC-V firmware or closed Axelera compiler.

**Eliminated routes**: change `model.json "memory"` (disasm `from_json` `0x36ee0` — only l1/l2/ddr accepted, no shared/host); LD_PRELOAD `get_ddr_size()` (only fools the guard, alloc still OOMs); pure alloc-layer interception (hits the shared→shared DMA wall); raising kernel-wait timeout (not a timeout). Reverse-engineering record was in `hacks/ze_shim/` (deleted with raw notes; reproducible from the owner's `metis-exp-board` repo).

> **Root truth**: load is a software limit (beaten); compute is a silicon/firmware limit (not beatable from user space). LLMs whose weights exceed 1 GB are out of scope for Metis Alpha. Investigation **closed**.

## Baseline vision performance (EfficientNet_b0, 2026-05-18)

Vision works well. EfficientNet_b0-imagenet, fakevideo 640×480, `dma_poll=1` (the **only** working DMA mode — board MSI is unreliable → interrupt mode 2 s-timeouts; polling ~0.1% CPU): end-to-end **~190 FPS** (`--show-stats`) / **502 FPS** quick run; Metis device inference alone **192 FPS** (5,199 µs/frame). **The bottleneck is host pre-processing, not the AIPU**: `colorconvert` is **13,939 µs** (CPU, `libtransform_colorconvert.so`) ≫ Metis 5,199 µs → AIPU is starved. Highest-value fix = RGA hardware colorconvert offload (est. ~1,400 µs, 10×). EfficientNet_b0 too small for thermal stress (Δ < 3 °C SoC; Metis core 47 °C nominal); Tier-2 thermal needs yolo11x-class sustained load. Offline compile (EfficientNet_b0) ≈ 643 s cold.

## Empirical compute ceiling (Step-1 campaign, 2026-05-23)

Full data: [[metis-step1-cnn-characterization-2026-05-23]] (45-cell Phase 1a + 180-cell Phase 1b).

| Unit | Phase 1a (vendor pipe, b=1) | Phase 1b best | Regime |
|---|---|---|---|
| **Metis Alpha** | 156–177 FPS flat across 38× GMACs | **1378 FPS MobileNetV2 mode-3 (SDK-managed batched multi-core)**; 1147 FPS mode-2 (user-managed 4-instance); 431 FPS mode-1 (single instance); 682 FPS ResNet-50 b=4; 137 FPS ResNet-152 b=1 | Vendor pipe = host-pipeline-bound (AIPU starved). Mode 3 wins both throughput and per-frame latency — one PCIe DMA / 4 frames beats 4 contending DMAs. **7.9× total gap on same silicon.** Intra-frame parallelism (`cooperative` / `pipeline`) **not implemented in v1.3.1** — single-frame latency only comes down via faster single-core per-call. |
| **RKNPU2** | 13–105 FPS (op-mix sensitive; EfficientNet-B0 collapses to 22 — depthwise+Swish) | **681** FPS MobileNetV2 b=32; **472** FPS ResNet-152 b=32; **589** FPS EfficientNet-B0 b=32 | Latency-bound at b=1, throughput-scales 5–10× with batching. **At b=32 beats Phase-1a Metis on every classifier.** |
| **Mali-G610** | 7–118 FPS (compute-bound; 10–25× slower than Metis) | ~180 FPS plateau (MobileNetV2); big models timeout >900 s | Bandwidth-saturated; batching does not help. **Not a viable full-model unit at any batch.** Realistic role = preprocessing offload. |
| **CPU** | not measured in this campaign | — | Gap vs Phase-0 §2; revisit if the CPU floor matters. |

**Toolchain note (corrected 2026-05-25 in [[metis-step1-cnn-characterization-2026-05-23]] v3):** the earlier "batched compile broken" diagnosis was a *wrong-knob* problem. The working path is the **3-knob YAML override** in `extra_kwargs.compilation_config`:
```yaml
extra_kwargs:
  compilation_config:
    aipu_cores: 4
    resources: 1.0
    multicore_mode: batch
```
All three required; this forces the compiler's `singlecore_to_batched_multicore` pass. **All 5 Phase-0 models batch cleanly via this path.** ResNet-50 and YOLOv8n hit this automatically by Voyager defaults; MobileNetV2 / EfficientNet-B0 / ResNet-152 default to `aipu_cores: 1` (Voyager-internal graph heuristic) and need the explicit override. **Intra-frame multi-core (`cooperative` / `pipeline`) NOT implemented in v1.3.1.**

## Modifiability map (depth analysis, all on-board verified)

| Layer | Verdict |
|---|---|
| Driver / PCIe (runtime, easy) | `dma_poll`/`dma_timeout`/`single_msi`/`enable_dmabuf_sync`/`dma_dual_channel` tunable; `dma_poll=1` mandatory on this board |
| Compiler / quant (compile-time, medium) | `dpu_constants_home` l2↔ddr, `double_buffer`, `l2_constraint`, tiling, `quantization_scheme`, `inter_operator_async`, `max_tiling_attempts` |
| Runtime pipeline (medium) | `dmabuf_inputs/outputs` zero-copy, `num_children` threads, custom pre/post `.so`, C++ Level-Zero direct calls |
| **Q1 shared-mem mgmt** | **Partial** — Level-Zero C++ custom allocator writable, `DmaBufAllocator` Python accessible; **no `zeMemRealloc`** (free+alloc only); `model.json` cannot select shared (only l1/l2/ddr) |
| **Q2 big.LITTLE scheduling** | **Fully modifiable** — `taskset`/`chrt`/cpufreq/cgroup-v2 `cpuset.cpus`/`sched_util_clamp_min`/EAS all verified writable |
| **Q3 L1 SPM policy** | **Not a cache** — 4 MB software-managed scratchpad, runtime-locked; needs closed Axelera compiler (the `cache.h` API controls only the RISC-V mgmt core's 64 KB SPM, *not* the AIPU L1) |
| **Q4 L2 SRAM policy** | **Not a cache** — 32 MB (4×8 MB banks), RISC-V-firmware-managed; no runtime evict/prefetch API; only `dpu_constants_home` / `l2_constraint` compile-time knobs; multi-model L2 swap only via manual `zeMemFree`+`zeMemAllocDevice` |
| L2 internal / RISC-V firmware | **Not possible** — closed source |

## Connections

- [[edge-ai]] — primary research domain
- [[compute-in-memory]] — Metis uses digital in-memory compute architecture
- [[on-device-llm-inference]] — LLM compute impossible here (Alpha firmware wall); vision-only
- [[sram-imc]] — contrasting IMC architecture (SRAM vs Metis digital CIM)
- [[system-axelera-metis-card]] — the *other* Metis device (production-class card); LLM runs there but hits the 24 GB/s memory wall
- [[metis-aipu-llm-architecture-research-fit]] — unified synthesis: is Metis AIPU viable for LLM architecture research?
- [[metis-aipu-full-stack-memory-management]] — idea directly impacted: the memory-management levers are closed firmware/compiler
- [[system-sram-imc-platform]] — the Delta robotics platform; different IMC paradigm, useful for cross-platform comparison
- [[edge-ai-systems]] — research project group most relevant to this platform
- [[characterizing-mobile-soc-heterogeneous-llm-inference-sosp2025]] — SOSP'25; characterizes systolic-array NPU pathologies (stage/order/shape sensitivity) on Snapdragon 8 Gen 3 — directly applicable to this board's RKNPU2 systolic-array NPU
- [[metis-step1-cnn-characterization-2026-05-23]] — Step-1 empirical capability + bottleneck campaign on this board (5 models × 3 units, 225 cells)
- **[[cim-centric-llm-mobile-soc]]** — **calibration source platform** for the new High research direction (CIM-centric LLM inference on simulated heterogeneous mobile SoC). This board feeds Phase 0 characterization for CIM compute primitives, PCIe behavior, RKNPU2 / Mali / CPU matmul + LLM-support op micro-benchmarks.
