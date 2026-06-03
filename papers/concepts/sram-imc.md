---
type: concept
title: "SRAM In-memory Computing"
created: 2026-05-07
updated: 2026-05-07
tags: [sram, compute-in-memory, edge-ai]
aliases: [SRAM-IMC, SRAM-CIM]
---

SRAM-IMC is treated as a practical, manufacturable compute-in-memory option for low-power edge inference, tactile processing, collision models, and always-on wearable sensing.

Related sources: [[24-25-in-memory-computing-sram-imc-platform-delta-y1-proposal]], [[24-25-in-memory-computing-sram-imc-platform-delta-y1-final]], [[24-25-in-memory-computing-sram-imc-platform-delta-y1-midterm1]], [[24-25-in-memory-computing-sram-imc-platform-delta-y1-midterm2]], [[25-26-in-memory-computing-sram-imc-platform-delta-y2-proposal]], [[25-26-in-memory-computing-sram-imc-platform-delta-y2-midterm1]], [[25-26-in-memory-computing-sram-imc-platform-delta-y2-midterm2]], [[26-31-wise-always-on-wearable-intelligent-systems-nstc-ssf]], [[26-27-collision-avoidance-and-synchronized-control-for-dual-robotic-arms-with-in-memory-computing-delta-y3-proposal]].

## Vault papers in this area

- **NeuroSim Validation** [[neurosim-validation-frontiers2021]]: 7-nm SRAM is the reference baseline technology in the post-validation system benchmark (Table 7); highest compute density (TOPS/mm²) among compared technologies.

## Random-weight / power-up connection

- **Weight Agnostic NNs** [[weight-agnostic-nn-2019]] (Gaier & Ha, NeurIPS 2019): NEAT-based NAS finds architectures that work with any random shared weight — relevant because SRAM-IMC platforms have non-deterministic power-up states that could serve as a random weight source (see [[sram-power-up-random-weight-neural-network]]).

## LLM-related SRAM-IMC

- **HPIM** [[hpim-arxiv2025]]: digital SRAM-IMC architecture (16 macro groups × 8 FP16 multipliers/macro) used as the attention-side PIM in a heterogeneous SRAM-PIM + HBM-PIM LLM accelerator (arXiv 2025).
