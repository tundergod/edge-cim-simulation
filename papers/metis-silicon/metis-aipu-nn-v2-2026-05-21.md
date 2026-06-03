---
type: source
title: "Metis AIPU Research Direction (v2): Pivot to CNN/DNN Edge Memory Wall"
created: 2026-05-22
updated: 2026-05-22
raw_path: raw/notes/metis-aipu-nn-v2-2026-05-21.pdf
source_kind: note
ingest_level: full
authors: [Wen-Sheng Lim]
year: 2026
tags: [metis, sram-cim, memory-wall, cnn, dnn, edge-ai, research-strategy]
---

# Metis AIPU Research Direction (v2): Pivot to CNN/DNN Edge Memory Wall

## TL;DR

Internal v2 research-direction report (2026-05-21) that pivots the Metis AIPU work axis from "LLM memory wall" to **CNN/DNN edge memory wall**, after the LLM path was ruled non-viable for architecture research ([[metis-aipu-llm-architecture-research-fit]]). Documents the 5-tier Metis memory hierarchy, the open Voyager compile knobs, a 71-paper PIM/CIM literature heatmap that identifies "commercial digital SRAM CIM × CNN/DNN" as the most empty cell, and a 9-cell research direction map naming CNN/DNN edge memory wall as the sweet spot. Proposes a two-platform plan (desktop = clean measurement; embedded = edge framework + multi-tenant) that yields two papers from one infrastructure.

## Key claims

- **5-tier Metis memory hierarchy** (§3): L0 IMC ~4 MiB per AIPU, L1 4 MiB per core (SW-managed scratchpad), L2 32 MiB shared on-chip SRAM, on-card / on-board LPDDR4x (1–16 GB depending on SKU), host RAM over PCIe Gen3 ×16 (or ×4 on embedded).
- **Voyager exposes compile-time knobs** but no runtime memory API (§4): `tiling_depth`, `multicore_mode`, `dpu_constants_home`, `l2_constraint`, `double_buffer`, `enable_buffer_promotion` — all decided at compile, locked at runtime.
- **71-paper PIM/CIM literature heatmap** (Appendix M, §5): commercial digital SRAM CIM × CNN/DNN at edge form factor is the most under-occupied cell; analog CIM and ReRAM dominate the published space.
- **9-cell research direction map** (§6): rows = {LLM, CNN/DNN, multi-tenant}, columns = {desktop, embedded, simulation}; **CNN/DNN × embedded** is the identified sweet spot for HPCA'27 main thrust.
- **Two-platform plan** (§7): desktop = clean memory-wall measurement (single accelerator, low noise); embedded = edge framework + multi-tenant contention; one shared measurement codebase, two papers.
- **LLM is appendix-only** (Appendix K): the 24.23 GB/s decode wall and the prefill-bound result are preserved as a measured reference point, not the main story.

## Motivation

The LLM-on-Metis line hit three independent walls (closed RISC-V RTOS firmware, precompiled-only LLM artifacts, hard 24 GB/s decode memory wall — all vendor-only fixes) and was closed for architecture research ([[metis-aipu-llm-architecture-research-fit]]). The Metis hardware is still capable silicon (~214 TOPS INT8, 15 TOPS/W); the question this report answers is *what research story does that capability fit, given the closed stack and the open Voyager compile surface?*

## Method

Triangulation across three inputs: (a) on-device measurement campaigns ([[metis-exp-board-rkc-a02-2026-05-18]], [[metis-llm-investigation-desktop-2026-05-19]]); (b) Voyager SDK documentation and compile-knob enumeration; (c) 71-paper PIM/CIM literature survey across the last ~3 years of top venues (ISCA, MICRO, HPCA, ASPLOS, ISSCC, VLSI), built into a categorical heatmap to surface the empty cells.

## Results

- **CNN/DNN edge memory wall is the right axis**: activation-driven (weight reuse is high, activations flow between layers — opposite of LLM decode which is weight-streaming-driven). Spills L1 (4 MiB/core) → L2 (32 MiB shared) → LPDDR4 (~24 GB/s) under realistic batch / resolution.
- **Embedded board chosen as HPCA'27 main thrust**: edge form factor makes the memory wall *intrinsic* to the platform, not an experimental contrivance — strong reviewer D5/D9.
- **Desktop becomes alt path**: clean measurement, but the memory-wall narrative is weaker (wide PCIe, GPU available, Metis is a supporting actor).
- **Two-platform plan yields two papers from one infrastructure**: shared ONNX→Voyager pipeline + measurement harness; embedded paper = HPCA'27, desktop paper = follow-on / cross-platform validation.

## Contributions / What's reusable

- The 5-tier hierarchy mental model is the reference picture used by every downstream Metis paper and idea in the vault.
- The 71-paper heatmap (Appendix M) and 9-cell direction map (§6) are reusable scaffolding for positioning future Metis-related submissions.
- The two-platform plan formalizes the "embedded vs desktop" decision that all four successor ideas inherit.
- Appendices J (Thetis/Europa generations), K (LLM data), L (simulator landscape), N (URL list) are bibliography for future work.

## Limitations / open questions

- Embedded board runs Voyager SDK v1.3.1 while desktop runs v1.6 — the compile-knob equivalence (`dpu_constants_home`, `l2_constraint`, `tiling_depth`) is a first-week go/no-go smoke test.
- The heatmap is a snapshot to 2026-05; needs refresh before HPCA'27 submission.
- Multi-tenant heterogeneous edge SoC contention is split out as a separate idea ([[multi-tenant-heterogeneous-edge-soc-contention]]) — not yet planned as a thrust.
- No public LLM compiler path for Metis → LLM appendix data is what we have; cannot be extended without vendor cooperation.

## Connections

- [[sram-imc]]
- [[compute-in-memory]]
- [[memory-centric-computing]]
- [[system-aetina-rkc-a02]] — embedded platform (HPCA'27 main thrust)
- [[system-axelera-metis-card]] — desktop platform (alt path)
- [[metis-exp-board-rkc-a02-2026-05-18]] — embedded measurement campaign
- [[metis-llm-investigation-desktop-2026-05-19]] — desktop LLM measurement (now appendix-only)
- [[metis-aipu-llm-architecture-research-fit]] — closed LLM thread (resolved 2026-05-22)
- [[cnn-dnn-edge-memory-wall-metis-embedded]] — HPCA'27 main thrust (successor idea)
- [[cnn-dnn-memory-wall-metis-desktop]] — alt path (successor idea)
- [[metis-cxl-cim-memory-system]] — CIM + CXL angle (successor idea)
- [[multi-tenant-heterogeneous-edge-soc-contention]] — multi-tenant systems angle (successor idea)
- [[cim-weight-changing-large-model]] — adjacent CIM idea (weight-change cost)
- [[multi-layer-computing-analysis-dnn]] — adjacent direction (multi-layer compute analysis)
