---
type: source
title: "Toward Full-System Heterogeneous Simulation: Merging gem5-SALAM with Mainline gem5"
created: 2026-05-25
updated: 2026-05-25
tags: [simulator, gem5, salam, accelerator-simulation, heterogeneous-simulation, soc, llvm, cycle-accurate, edge-ai, mobile-inference]
raw_path: https://www.gem5.org/2025/07/30/gem5AccHetSimBlog.html
source_kind: note
ingest_level: weak
authors: [Matthew Sinclair, Samuel Rogers, Joshua Slycord, Hamed Tabkhi]
venue: gem5.org-blog + ISCA-2025-workshop-poster
year: 2025
---

# gem5+SALAM (2025): Full-System Heterogeneous SoC Simulation

## TL;DR

gem5-SALAM — originally a standalone fork targeting programmable hardware accelerators via LLVM IR execution — was merged into mainline gem5 in mid-2025 and presented at ISCA 2025. The merge elevates custom accelerators to first-class components (alongside CPUs and GPUs) with cycle-accurate datapath modeling, memory-mapped scratchpads, DMA engines, and a hardware-profile generator that automates accelerator construction from timing specs. There is no mention of compute-in-memory, analog arrays, or NVM in the framework; gem5-SALAM models digital, LLVM-compiled accelerator datapaths, not resistive or capacitive analog compute fabrics.

## What gem5+SALAM provides

- **Cycle-accurate accelerator modeling** via the `LLVMInterface`: LLVM IR kernels execute with per-instruction cycle timing derived from user-supplied hardware profiles; this is "execute-in-execute" (not trace-replay), so timing is sensitive to runtime data.
- **Memory hierarchy**: memory-mapped scratchpads, DMA engines, stream buffers; the `AccCluster` abstraction groups accelerators with their local memories, reflecting real SoC modularity (e.g., separate clusters for vision, NPU, DSP blocks).
- **`CommInterface`**: software-visible control registers and interrupt signaling, enabling full-stack OS + accelerator co-simulation (firmware → driver → kernel).
- **Hardware profile generator**: automates accelerator datapath construction from timing specs; replaces hand-crafted SimObjects.
- **CACTI-SALAM**: scratchpad memory timing + energy estimation, refactored from the original tool.
- **Full-system scope (post-merge)**: CPUs, GPUs, and accelerators in one gem5 instance — supports studies of performance interference, resource arbitration, and synchronization across heterogeneous compute engines.
- **Workloads targeted**: real-time vision, mobile inference, AR/VR, edge computing, and (noted for future work) multi-GHz accelerators with advanced cooling.
- **ISA coverage**: ARM at merge time; extension to additional ISAs noted as a next step.

## What's still missing for CIM modeling

- **No analog array model.** gem5-SALAM assumes digital, LLVM-compilable compute. CIM's defining characteristic — analog matrix-vector multiplication inside resistive/capacitive arrays — has no representation. There is no ADC/DAC model, no device-variation or write-noise model, no charge-domain compute primitive.
- **No NVM / emerging-device substrate.** RRAM, PCM, FeFET, and SRAM-CIM-specific array behaviors (e.g., bitline accumulation, activation sparsity in analog domain) are out of scope. The framework is device-agnostic only in the digital sense.
- **No weight-stationary data-flow model.** CIM accelerators are fundamentally weight-stationary; SALAM's LLVM IR execution model naturally implies a more instruction-streaming paradigm incompatible with the tiled analog-MVM execution style.
- **No circuit-level accuracy.** [[dnn-neurosim-v1-iedm2019|DNN+NeuroSim V1]] (the incumbent CIM simulator) operates at device → circuit → algorithm hierarchy with nonideal-device noise models. gem5-SALAM operates at architecture level; the two layers do not yet connect.
- **Energy model is scratchpad-centric.** CACTI-SALAM estimates SRAM scratchpad energy; there is no analog-array energy model (no charge accumulation, no conversion cost).
- **Gap for integration:** To use gem5+SALAM as CIM-SoC substrate, one would need to build a custom `SimObject` implementing the analog-array timing and energy, then couple it with CACTI-SALAM for SRAM peripherals. This is nontrivial but the AccCluster abstraction and hardware profile generator provide structural scaffolding.

## Why it might matter

The [[cim-llm-mobile-soc-simulator]] idea (High priority, HOME) is exploring "path (b): extend an existing simulator" as the build strategy. gem5+SALAM is the strongest candidate for that path: it is the only mainline gem5 extension that gives full-system SoC simulation (CPU + NPU/CIM accelerator + memory hierarchy) without a custom gem5 fork, and its AccCluster abstraction maps naturally onto a heterogeneous mobile SoC with a CIM block alongside CPU cores. The critical gap is the analog layer — but the structural scaffolding (hardware profile generator, CommInterface, CACTI-SALAM) reduces from-scratch work significantly compared to building on vanilla gem5 or choosing [[dnn-neurosim-v1-iedm2019|DNN+NeuroSim]] (which is inference-only and has no CPU/system simulation). Cross-validation plan: use real Aetina+Metis data from [[cnn-dnn-edge-memory-wall-metis-embedded]] to calibrate the digital side of the simulator, then add CIM analog modeling on top.

**relevance: high** — directly evaluates a top candidate for the priority-1 research idea.

## Connections

- [[compute-in-memory]] — gem5-SALAM does not natively model CIM analog arrays; adding a CIM SimObject is the key extension needed
- [[in-memory-computing]] — broader substrate context; gem5-SALAM sits on the digital side of the IMC design space
- [[memory-centric-computing]] — SoC memory hierarchy modeling is a gem5-SALAM strength; relevant to memory-wall research direction
- [[edge-ai]] — target application domain: mobile inference, AR/VR, real-time vision workloads
- [[on-device-llm-inference]] — the [[cim-llm-mobile-soc-simulator]] idea targets LLM inference on CIM-equipped SoCs; gem5-SALAM is the prospective substrate
- [[dnn-neurosim-v1-iedm2019]] — the incumbent CIM simulator; complementary (device→circuit layer) but lacks system/CPU simulation; combining both layers is possible in principle but requires custom integration work
- [[dnn-neurosim-v2-tcad2021]] — extends NeuroSim V1 with training; same architectural gap w.r.t. system simulation
- [[cnn-dnn-edge-memory-wall-metis-embedded]] — calibration data source (real Aetina+Metis measurements) for the planned simulator
- [[cim-centric-llm-mobile-soc]] — **simulator path (b) candidate that was assessed and not chosen** (chose hybrid path (c) — NeuroSim + Ramulator2 + custom system layer instead). Retreat candidate if NeuroSim hybrid fails at M1.
