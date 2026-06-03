---
type: concept
title: "In-memory Computing"
created: 2026-05-07
updated: 2026-05-07
tags: [memory-systems, ai-ml]
aliases: [IMC]
---

In-memory computing reduces data movement by placing computation in memory or storage-adjacent structures. It is one of the vault's main cross-cutting themes, spanning NVM, SRAM-CIM, edge AI, and computational storage.

Related sources: [[19-22-design-and-optimization-of-multi-stream-bit-alterable-flash-storage-for-next-generation-in-memory-computing-nstc]], [[19-21-efficient-pcm-based-computing-systems-for-neural-network-training-nstc]], [[24-25-in-memory-computing-sram-imc-platform-delta-y1-proposal]], [[25-28-design-and-optimization-for-aiml-with-in-memory-computing-nstc]], [[25-28-transparent-ecosystems-for-edge-ai-from-smart-glasses-to-phones-and-edge-servers-nstc]], [[26-29-integrating-computational-memory-hierarchies-for-efficient-memory-centric-computing-nstc-bmbf]], [[26-27-collision-avoidance-and-synchronized-control-for-dual-robotic-arms-with-in-memory-computing-delta-y3-proposal]].

## Vault papers in this area

- **In-memory HDC** [[karunaratne-inmemory-hdc-2020]]: full HDC inference (IM + AM) in PCM crossbars; no data movement to CPU; 6× energy reduction.
- **Streaming Encoding** [[streaming-encoding-hdc-2022]]: hash-based HDC encoding eliminates the Item Memory; ReRAM CIM implementation achieves 1000× speedup; first HDC accelerator for both numeric and categorical inputs.
- **DeCoHD** [[decohd-2025]]: streaming HDC inference with O(D) peak memory; class-axis decomposition reduces Associative Memory from O(CD) to O(LD); compatible with in/near-memory accelerator designs.
- **ReHDC** [[rehdc-tcad2024]]: ReRAM-PIM HDC, unified encode+compare engine; static-encode CIM-HDC is occupied.
- **Tri-HD** [[tri-hd-tcad2024]]: full-pipeline ReRAM-PIM HDC (Rosing); PIM-friendly distance metric.
- **DNN+NeuroSim V1/V2** [[dnn-neurosim-v1-iedm2019]] / [[dnn-neurosim-v2-tcad2021]]: de-facto CIM benchmarking simulators (inference / +training).
- **NeuroSim Validation** [[neurosim-validation-frontiers2021]]: first silicon validation of NeuroSim against 40-nm TSMC RRAM CIM macro; chip-level error <1% after 7-factor PDK calibration — establishes silicon-validated status for the NeuroSim lineage.
- **HPIM** [[hpim-arxiv2025]]: heterogeneous SRAM-PIM + HBM-PIM accelerator for LLM inference (arXiv 2025); simulator-only.
- **PAPI** [[papi-asplos2025]]: heterogeneous GPU + PIM (FC-PIM + Attn-PIM) for LLM decode with dynamic kernel scheduling (ASPLOS 2025).
- **gem5-SALAM** [[gem5-salam-merge-2025]]: simulator infrastructure — full-system heterogeneous accelerator modeling merged into mainline gem5 (mid-2025); does NOT provide CIM analog-array primitives.

## Connections

[[memory-centric-computing]] · [[compute-in-memory]] · [[processing-in-memory-llm]] · [[mi-llm-multiplier-free-pim-tc2026]] (multiplier-free LUT LLM inference on real commodity UPMEM near-bank PIM; real-hardware in-memory computing for LLM inference; IEEE TC 2026) · [[pim-dl-asplos2024]] (LUT-NN on real commodity DRAM-PIM via eLUT-NN + Auto-Tuner; DNN-scoped foundation of the LUT-on-PIM thread; ASPLOS 2024)
