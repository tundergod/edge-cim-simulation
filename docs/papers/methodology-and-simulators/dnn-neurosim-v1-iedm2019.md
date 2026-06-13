---
type: source
title: "DNN+NeuroSim: An End-to-End Benchmarking Framework for Compute-in-Memory Accelerators with Versatile Device Technologies"
created: 2026-05-18
updated: 2026-05-18
tags: [cim-simulator, compute-in-memory, benchmarking, neurosim, reram]
raw_path: raw/papers/dnn-neurosim-v1-iedm2019.pdf
source_kind: paper
ingest_level: full
authors: [Xiaochen Peng, Shanshi Huang, Yandong Luo, Xiaoyu Sun, Shimeng Yu]
venue: IEDM
year: 2019
---

# DNN+NeuroSim (V1): End-to-End Benchmarking Framework for CIM Accelerators

**TL;DR.** The de-facto standard CIM simulator. A PyTorch/TensorFlow Python wrapper around the NeuroSim hardware macro model, giving hierarchical device→circuit→algorithm benchmarking of CIM accelerators. Models analog-synaptic-device reliability + ADC quantization effects on inference accuracy; benchmarks SRAM and emerging devices (RRAM, PCM, FeFET, ECRAM) from VGG to ResNet, CIFAR to ImageNet. Open source (github.com/neurosim/DNN_NeuroSim_V1.0). (Georgia Tech, Shimeng Yu.)

> Lineage: **foundational tool, not superseded** — succeeded by [[dnn-neurosim-v2-tcad2021|DNN+NeuroSim V2.0]] (adds on-chip training) and later V1.x inference revisions; V1 remains the canonical inference-engine benchmarking baseline.

## Key claims

- A hierarchical (device→circuit→chip→algorithm) CIM benchmarking framework is needed beyond the prior 2-layer-MNIST MLP+NeuroSim (§I).
- Captures analog synaptic-device non-idealities + ADC quantization impact on inference accuracy (abstract).
- Supports versatile devices: SRAM, RRAM, PCM, FeFET, ECRAM; flexible topologies VGG/ResNet; CIFAR→ImageNet (abstract).
- SPICE-calibrated component models (PTM nodes) → credible chip-level area/latency/energy/throughput (§II).

## Method

Python wrapper interfaces NeuroSim macro model with PyTorch/TensorFlow; auto algorithm→hardware mapping; unrolls synaptic/activation traces, partitions to chip floorplan; outputs hardware-constrained accuracy + area/latency/energy.

## Results

Benchmarks across device technologies and networks; quantifies benefit of high on-state resistance and three-terminal synapses for inference; demonstrates end-to-end accuracy-vs-hardware trade-offs.

## Contributions

Established the open, SPICE-calibrated, end-to-end CIM benchmarking methodology now widely accepted at DAC/ICCAD/DATE/MICRO — the vault's named "venue-accepted simulator" for CIM-HDC ([[incremental-in-storage-hdc-index]] D4).

## Limitations / open questions

- DNN/CNN-oriented; HDC permute/n-gram recurrence is not a built-in workload — adapting it for the long-sequence HDC encoding study requires custom mapping.
- Inference only (V1); training modeled in [[dnn-neurosim-v2-tcad2021|V2.0]].

## Connections

- [[incremental-in-storage-hdc-index]] — listed in papers-to-ingest as "DNN+NeuroSim"; the de-facto CIM simulator underwriting that idea's D4 (platform credibility) argument.
- [[dnn-neurosim-v2-tcad2021]] — direct successor adding on-chip training.
- [[neurosim-validation-frontiers2021]] — silicon validation of this framework against 40-nm TSMC RRAM macro; <1% chip-level error; upgrades V1 to silicon-validated status.
- [[cim-weight-changing-large-model]], [[metis-aipu-full-stack-memory-management]] — candidate simulation tool for CIM weight-management studies.
- [[compute-in-memory]]
- [[in-memory-computing]]
- [[gem5-salam-merge-2025]] — system-simulation counterpart: digital SoC + accelerator scaffolding (no analog model); NeuroSim provides the CIM tile model, gem5-SALAM the system layer.
