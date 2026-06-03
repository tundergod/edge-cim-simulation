---
type: source
title: "DNN+NeuroSim V2.0: An End-to-End Benchmarking Framework for Compute-in-Memory Accelerators for On-Chip Training"
created: 2026-05-18
updated: 2026-05-18
tags: [cim-simulator, compute-in-memory, on-chip-training, neurosim, benchmarking]
raw_path: raw/papers/dnn-neurosim-v2-tcad2021.pdf
source_kind: paper
ingest_level: full
authors: [Xiaochen Peng, Shanshi Huang, Hongwu Jiang, Anni Lu, Shimeng Yu]
venue: IEEE TCAD (Vol. 40, No. 11)
year: 2021
extends: "[[dnn-neurosim-v1-iedm2019]]"
---

# DNN+NeuroSim V2.0: End-to-End CIM Benchmarking for On-Chip Training

**TL;DR.** Extends [[dnn-neurosim-v1-iedm2019|DNN+NeuroSim V1]] from inference-only to **on-chip training**. Adds modeling of analog-eNVM non-ideal device properties for training — nonlinearity/asymmetry of weight update, device-to-device and cycle-to-cycle variation — plus peripheral circuits for error/weight-gradient backprop. Benchmarks SRAM and eNVM (RRAM, FeFET) for VGG-8/CIFAR-10, revealing the synaptic-device specs that matter for in-situ training. Open source (DNN_NeuroSim_V2.0). (Georgia Tech, Shimeng Yu.)

> Lineage: **extends** V1 (inference) with training support; current canonical CIM-training benchmarking tool, not superseded.

## Key claims

- Adds on-chip *training* evaluation to the NeuroSim framework (abstract).
- Models nonlinearity/asymmetry, device-to-device + cycle-to-cycle variation of analog weight update (abstract).
- Implements peripheral circuits for error/weight-gradient computation in backprop (abstract).
- Identifies the synaptic-device specs critical for in-situ training accuracy and hardware efficiency (abstract).

## Method

Python wrapper (PyTorch) + NeuroSim core extended with backprop datapath; nonideal-device training models; benchmarks SRAM/RRAM/FeFET on VGG-8/CIFAR-10 for area/energy/throughput + training accuracy.

## Results

Quantifies how device nonidealities degrade on-chip training; reveals which synaptic-device parameters dominate training accuracy and efficiency.

## Contributions

First open end-to-end CIM **training** benchmarking framework with calibrated nonideal-device models — completes the V1 inference / V2 training tool pair widely accepted at architecture/EDA venues.

## Limitations / open questions

- DNN training oriented; HDC online retraining or long-sequence encoding is not native — adaptation required for the HDC encoding study.
- Simulation only (as intended for a benchmarking framework) — accepted at TCAD/architecture venues, which is exactly why it underwrites the vault's D4 platform-credibility argument.

## Connections

- [[incremental-in-storage-hdc-index]] — DNN+NeuroSim (V1/V2) is the named venue-accepted CIM simulator backing that idea's D4.
- [[dnn-neurosim-v1-iedm2019]] — the inference-only predecessor this extends.
- [[neurosim-validation-frontiers2021]] — silicon validation applicable to the NeuroSim lineage (V1 framework); <1% chip-level error; the D4 provenance reference for any CIM simulator claim using NeuroSim.
- [[cim-weight-changing-large-model]], [[metis-aipu-full-stack-memory-management]] — relevant if weight-change/training cost is modeled in simulation.
- [[compute-in-memory]]
- [[in-memory-computing]]
