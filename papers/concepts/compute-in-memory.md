---
type: concept
title: "Compute-in-memory"
created: 2026-05-07
updated: 2026-05-07
tags: [memory-systems, accelerators]
aliases: [CIM, computing-in-memory]
---

Compute-in-memory places arithmetic inside memory arrays or tightly alongside memory to reduce data movement. In this vault it appears in SRAM robotic manipulation, ReRAM/3D NAND AI/ML acceleration, ECG reasoning, and skyrmion racetrack memory systems.

Related sources: [[24-25-in-memory-computing-sram-imc-platform-delta-y1-proposal]], [[24-25-in-memory-computing-sram-imc-platform-delta-y1-final]], [[24-27-holistic-design-of-stretchable-oects-and-brain-inspired-circuits-for-neuromorphic-computing-nstc-afosr]], [[25-26-in-memory-computing-sram-imc-platform-delta-y2-proposal]], [[25-26-in-memory-computing-sram-imc-platform-delta-y2-midterm2]], [[25-28-design-and-optimization-for-aiml-with-in-memory-computing-nstc]], [[26-29-enabling-high-performance-data-centric-computing-on-skyrmionic-racetrack-memories-nstc-bmbf]], [[26-29-listen-to-your-heart-cim-and-accelerator-design-for-ecg-data-reasoning-on-the-edge-nstc]], [[26-27-collision-avoidance-and-synchronized-control-for-dual-robotic-arms-with-in-memory-computing-delta-y3-proposal]].

## Vault papers in this area

- **In-memory HDC** [[karunaratne-inmemory-hdc-2020]]: PCM crossbar CIM for HDC — in-memory XOR binding (Item Memory) and dot-product similarity search (Associative Memory); 6× energy reduction vs CMOS baseline.
- **Streaming Encoding** [[streaming-encoding-hdc-2022]]: hash-based HDC encoding mapped to ReRAM CIM operations; 1000× speedup over CPU; eliminates Item Memory storage overhead.
- **DeCoHD** [[decohd-2025]]: class-axis HDC decomposition with streaming inference; bind–bundle–dot pipeline is compatible with in/near-memory accelerators; 277× energy gain vs CPU.
- **ReHDC** [[rehdc-tcad2024]]: ReRAM-PIM HDC unifying encode + compare in one engine (analog accum + digital XOR crossbars).
- **Tri-HD** [[tri-hd-tcad2024]]: first full-pipeline ReRAM-PIM HDC for nonbinary data; PIM-friendly distance metric; segmented bitlines.
- **DNN+NeuroSim V1** [[dnn-neurosim-v1-iedm2019]]: the de-facto venue-accepted CIM benchmarking simulator (device→circuit→algorithm); inference engine.
- **DNN+NeuroSim V2.0** [[dnn-neurosim-v2-tcad2021]]: extends V1 with on-chip training + nonideal-device update models.
- **NeuroSim Validation** [[neurosim-validation-frontiers2021]]: silicon validation of the NeuroSim framework against 40-nm TSMC RRAM macro; seven PDK-calibration factors achieve <1% chip-level error; the methodology template for D4 simulator credibility claims.
- **UPMEM PIM Case Study** [[pim-case-study-atc2021]]: first real off-the-shelf UPMEM evaluation (ATC'21); bandwidth scales with memory size; data-copy and DPU-speed limitations; HDC 13× speedup — baseline for PIM-path work.
- **HPIM** [[hpim-arxiv2025]]: SRAM-PIM substrate for attention ops in a heterogeneous SRAM-PIM+HBM-PIM LLM accelerator (arXiv 2025).
- **gem5-SALAM** [[gem5-salam-merge-2025]]: mainline gem5 heterogeneous accelerator simulation; provides digital SoC scaffolding but no analog-array model — must be extended for CIM.

## Connections

[[memory-centric-computing]] · [[in-memory-computing]] · [[processing-in-memory-llm]] · [[mi-llm-multiplier-free-pim-tc2026]] (multiplier-free LUT LLM inference on real UPMEM near-bank PIM; most directly relevant real-hardware CIM-for-LLM result) · [[sieve-moe-pim-arxiv2026]] (dynamic GPU/PIM expert partitioning for MoE on HBM-PIM; simulation) · [[duplex-moe-pim-isca2024]] (Op/B-based hot/cold expert split across xPU + Logic-PIM for MoE; ISCA 2024, simulation) · [[pim-dl-asplos2024]] (LUT-NN on real commodity DRAM-PIM via eLUT-NN + Auto-Tuner; DNN-scoped foundation of the LUT-on-PIM thread; ASPLOS 2024)
