---
type: source
title: "Metis Experimental Board (Aetina RKC-A02 + Metis Alpha M.2): Architecture and Modifiability Map"
created: 2026-05-22
updated: 2026-05-22
raw_path: raw/notes/metis-exp-board-rkc-a02-2026-05-18.html
source_kind: note
ingest_level: full
authors: [Wen-Sheng Lim]
year: 2026
tags: [metis, aetina, rk3588, edge-ai, sram-cim, modifiability, big-little]
---

# Metis Experimental Board (Aetina RKC-A02 + Metis Alpha M.2): Architecture and Modifiability Map

## TL;DR

Architecture-and-modifiability characterization (2026-05-18) of the Aetina RKC-A02 embedded board (RK3588 + Metis **Alpha** M.2). Measures a clean vision baseline (EfficientNet_b0 ~190 FPS end-to-end with CPU colorconvert as the bottleneck), confirms the LLM-compute closed-firmware wall (`-1301`) is silicon-limited not software-limited, and enumerates Q1–Q4 deep modifiability findings that determine which research directions are reachable on this platform.

## Key claims

- **Platform topology** (§A): Aetina RKC-A02 = Rockchip RK3588 (4×Cortex-A76 + 4×Cortex-A55, Mali-G610 MP4 GPU, RKNPU2 6 TOPS) + 16 GB shared LPDDR4 + Axelera Metis **Alpha** M.2 (PCIe Gen3 ×4, **no on-card DDR**, only a 1 GB PCIe IOMMU window into host LPDDR4).
- **Vision baseline measured** (§B, Table B1): EfficientNet_b0 ~190 FPS end-to-end; Metis inference alone 5.2 ms (192 FPS); CPU colorconvert 13.9 ms — **CPU pre-processing is the bottleneck, not the AIPU**. RKNPU2 fully idle.
- **LLM compute is a closed-firmware hard wall** (§C, locator `-1301`): model **load** can be surmounted from user space via `ze_shim2` LD_PRELOAD; **compute** dead-ends at the proprietary RISC-V firmware. The error is silicon-limited (not a timeout, not a software bug).
- **Q1 (shared memory): partial** (§D.1). DMA-buf available; Mali sysfs writable; `ze_shim2` proves user-space model-load redirection works *for vision*; firmware-side compute path is sealed.
- **Q2 (big.LITTLE): fully modifiable, verified** (§D.2). cgroup v2 `cpuset.cpus` and `sched_util_clamp_min/max` are writable; A76/A55 pinning and frequency clamping behave as documented.
- **Q3 (L1 SRAM): not a cache, software-managed scratchpad** (§D.3). 4 MiB per core; layout fully decided by the Voyager compiler; no runtime evict/prefetch/realloc API.
- **Q4 (L2 SRAM): compile-time only** (§D.4). 32 MiB shared; placement controlled by `dpu_constants_home` and `l2_constraint` knobs at compile time; locked at runtime.

## Motivation

Before committing to a Metis research thrust, the team needed a **modifiability audit**: which architectural levers (shared memory routing, big.LITTLE scheduling, L1 scratchpad management, L2 placement) are actually reachable from user space on this specific embedded SKU, and which are sealed behind closed firmware or compiler? The audit determines which research directions are *implementable* on this board vs which would require vendor cooperation.

## Method

- Vision baseline: stock EfficientNet_b0 pipeline (Voyager SDK, INT8), measured end-to-end FPS and per-stage latency on a real workload. RKNPU2 monitored via sysfs to confirm idle.
- LLM compute probe: attempt `axruntime` LLM model invocation; observe `-1301` after successful `ze_shim2`-assisted load; confirm not a timeout or environment issue by exhaustive elimination.
- Modifiability Q1–Q4: each lever probed independently — DMA-buf import/export tests; cgroup v2 pin and clamp checks under load; Voyager IR inspection for L1/L2 placement; documentation cross-reference for any runtime API.

## Results

- Vision pipeline shipping-grade and well-characterized; CPU colorconvert is the obvious first systems-work target (RGA offload, big.LITTLE pinning, RKNPU2 co-scheduling).
- **LLM compute on Alpha M.2 is permanently blocked** until Axelera opens firmware or ships a different SKU.
- Big.LITTLE is the strongest user-space lever on this board — fully tunable, verified, and matters for multi-tenant contention studies.
- L1/L2 SRAM are *compile-time* surfaces only; any "full-stack memory management" claim must be reframed as compiler-policy work or cross-platform.

## Contributions / What's reusable

- The Q1–Q4 modifiability matrix is the canonical reference for "what is reachable on RKC-A02 + Alpha M.2" — directly feeds the four successor ideas.
- The vision baseline (190 FPS EfficientNet_b0, 13.9 ms colorconvert bottleneck) is a reproducible starting point for any CNN/DNN edge memory-wall measurement on this platform.
- The `ze_shim2` LD_PRELOAD trick (cleanly separating "software limit, beatable" from "silicon limit, not") is reusable for any future Axelera SKU triage.

## Limitations / open questions

- Alpha M.2 (no on-card DDR) is not production Metis; some closed-firmware behaviors may differ on the production card.
- The CPU colorconvert bottleneck means vision FPS is not a pure measurement of AIPU memory pressure — CNN/DNN memory-wall studies need a workload that forces L1/L2 → LPDDR4 spill (high batch, high resolution).
- SDK v1.3.1 on the embedded board vs v1.6 on the desktop: compile-knob equivalence not yet verified.
- Whether the `-1301` wall is Alpha-specific or also present on production Metis is unresolved (would constrain any custom LLM memory scheme on a 16 GB card too).

## Connections

- [[sram-imc]]
- [[compute-in-memory]]
- [[memory-centric-computing]]
- [[system-aetina-rkc-a02]] — entity page for this board
- [[system-axelera-metis-card]] — comparison: production card vs Alpha M.2
- [[metis-aipu-nn-v2-2026-05-21]] — the v2 direction report that ingests this finding
- [[metis-llm-investigation-desktop-2026-05-19]] — sister LLM campaign on the desktop card
- [[metis-aipu-llm-architecture-research-fit]] — synthesis of the LLM-research closure
- [[cnn-dnn-edge-memory-wall-metis-embedded]] — HPCA'27 main thrust idea built on this audit
- [[multi-tenant-heterogeneous-edge-soc-contention]] — uses Q2 big.LITTLE finding directly
- [[cim-weight-changing-large-model]] — adjacent measurement (weight-change cost on Metis)
