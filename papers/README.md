# Literature & Real-Silicon Notes

Curated paper notes and real-silicon investigation reports for the CIM-centric LLM-on-mobile-SoC simulator. These are structured reading notes (TL;DR, claims, method, simulator/D4 posture, "why it matters to us") — not the original PDFs. Cross-references between notes use `[[wikilink]]` syntax (Obsidian origin); resolve them by filename within `papers/`.

Start with [hpim-arxiv2025](pim-llm-accelerators/hpim-arxiv2025.md) (closest competitor), [characterizing-mobile-soc-heterogeneous-llm-inference-sosp2025](methodology-and-simulators/characterizing-mobile-soc-heterogeneous-llm-inference-sosp2025.md) (HeteroInfer — our methodology template), and the two `metis-silicon/` investigation reports (the calibration ground truth).

---

## pim-llm-accelerators/ — PIM/CIM LLM accelerators (prior art & competitors)
- [hpim-arxiv2025](pim-llm-accelerators/hpim-arxiv2025.md) — **Closest competitor.** Heterogeneous SRAM-PIM + HBM-PIM, single-batch, simulator-only/FP16-only (its D4 gap = our strength)
- [papi-asplos2025](pim-llm-accelerators/papi-asplos2025.md) — Dynamic GPU+PIM decoding parallelism (FC-PIM/Attn-PIM split)
- [specpim-asplos2024](pim-llm-accelerators/specpim-asplos2024.md) — Speculative inference on PIM, architecture-dataflow co-exploration
- [neupims-asplos2024](pim-llm-accelerators/neupims-asplos2024.md) — NPU-PIM heterogeneous, batched LLM
- [ianus-asplos2024](pim-llm-accelerators/ianus-asplos2024.md) — NPU-PIM unified memory system
- [cent-asplos2025](pim-llm-accelerators/cent-asplos2025.md) — CXL-enabled GPU-free all-PIM LLM inference
- [lp-spec-arxiv2025](pim-llm-accelerators/lp-spec-arxiv2025.md) — LPDDR-PIM mobile speculative inference (mobile sibling)
- [cxl-pnm-lpddr-hpca2024](pim-llm-accelerators/cxl-pnm-lpddr-hpca2024.md) — LPDDR-based CXL-PNM transformer inference
- [lincoln-hpca2025](pim-llm-accelerators/lincoln-hpca2025.md) — 50–100B LLM on consumer devices via compute-enabled flash
- [cambricon-llm-micro2024](pim-llm-accelerators/cambricon-llm-micro2024.md) — Chiplet NPU + flash-PIM, on-device 70B
- [duplex-moe-pim-isca2024](pim-llm-accelerators/duplex-moe-pim-isca2024.md) — MoE + GQA + continuous batching device
- [l3-dimm-pim-longcontext-arxiv2025](pim-llm-accelerators/l3-dimm-pim-longcontext-arxiv2025.md) — DIMM-PIM long-context coordination
- [pimphony-lolpim-longcontext-hpca2026](pim-llm-accelerators/pimphony-lolpim-longcontext-hpca2026.md) — Bandwidth/capacity for PIM long-context
- [repa-kvcache-pim-asplos2026](pim-llm-accelerators/repa-kvcache-pim-asplos2026.md) — Reconfigurable PIM for KV-cache offload+processing
- [starc-sparse-attention-pim-arxiv2025](pim-llm-accelerators/starc-sparse-attention-pim-arxiv2025.md) — Sparse-attention remapping for PIM decoding
- [mi-llm-multiplier-free-pim-tc2026](pim-llm-accelerators/mi-llm-multiplier-free-pim-tc2026.md) — Multiplier-free LLM on commodity PIM
- [pim-llm-pgemmlib-cgo2025](pim-llm-accelerators/pim-llm-pgemmlib-cgo2025.md) — GEMM library + target-aware opts on real PIM
- [context-aware-moe-cxl-ndp-arxiv2025](pim-llm-accelerators/context-aware-moe-cxl-ndp-arxiv2025.md) — Context-aware MoE on CXL GPU-NDP
- [sieve-moe-pim-arxiv2026](pim-llm-accelerators/sieve-moe-pim-arxiv2026.md) — Dynamic expert-aware PIM for evolving MoE

## methodology-and-simulators/ — simulation & validation methodology
- [characterizing-mobile-soc-heterogeneous-llm-inference-sosp2025](methodology-and-simulators/characterizing-mobile-soc-heterogeneous-llm-inference-sosp2025.md) — **HeteroInfer.** The characterization-driven methodology template (per-unit characteristic curves decide the op split)
- [neurosim-validation-frontiers2021](methodology-and-simulators/neurosim-validation-frontiers2021.md) — CIM-simulator validation against silicon (the D4 pattern; <1% chip error citation)
- [dnn-neurosim-v1-iedm2019](methodology-and-simulators/dnn-neurosim-v1-iedm2019.md) — CIM benchmarking framework (inference)
- [dnn-neurosim-v2-tcad2021](methodology-and-simulators/dnn-neurosim-v2-tcad2021.md) — CIM benchmarking framework (on-chip training)
- [gem5-salam-merge-2025](methodology-and-simulators/gem5-salam-merge-2025.md) — Full-system heterogeneous simulation (option assessed, not chosen)

## on-device-llm/ — on-device / mobile LLM inference systems
- [powerinfer2-smartphone-2024](on-device-llm/powerinfer2-smartphone-2024.md) — Fast LLM inference on a smartphone
- [fast-ondevice-llm-npu-asplos2025](on-device-llm/fast-ondevice-llm-npu-asplos2025.md) — On-device LLM with NPU acceleration
- [llm-in-a-flash-apple-2023](on-device-llm/llm-in-a-flash-apple-2023.md) — LLM inference with limited memory (flash offload)
- [kvswap-ondevice-2025](on-device-llm/kvswap-ondevice-2025.md) — On-device long-context via KV-cache swapping

## metis-silicon/ — our real-silicon investigations (calibration ground truth)
- [metis-exp-board-rkc-a02-2026-05-18](metis-silicon/metis-exp-board-rkc-a02-2026-05-18.md) — Aetina board architecture + modifiability map; the `-1301` LLM wall
- [metis-llm-investigation-desktop-2026-05-19](metis-silicon/metis-llm-investigation-desktop-2026-05-19.md) — Production card LLM bottleneck (~24 GB/s decode memory wall) — **L4 anchor**
- [metis-step1-cnn-characterization-2026-05-23](metis-silicon/metis-step1-cnn-characterization-2026-05-23.md) — 225-cell CNN characterization (5 models × 3 units) — **L6 anchor**
- [metis-aipu-nn-v2-2026-05-21](metis-silicon/metis-aipu-nn-v2-2026-05-21.md) — Research-direction synthesis report

## platforms/ — hardware platform entity pages
- [system-aetina-rkc-a02](platforms/system-aetina-rkc-a02.md) — RK3588 + Metis Alpha M.2 (Phase 0 Machine 1)
- [system-axelera-metis-card](platforms/system-axelera-metis-card.md) — Production Metis card + RTX 3090 host (Phase 0 Machine 2)

## ideas/ — project idea pages
- [cim-centric-llm-mobile-soc](ideas/cim-centric-llm-mobile-soc.md) — This project's source idea page
- [cnn-dnn-edge-memory-wall-metis-embedded](ideas/cnn-dnn-edge-memory-wall-metis-embedded.md) — Calibration-source idea (Step-1 data feeds L6)

## concepts/ — background concept notes
CIM/PIM and LLM-systems primers: [compute-in-memory](concepts/compute-in-memory.md), [in-memory-computing](concepts/in-memory-computing.md), [sram-imc](concepts/sram-imc.md), [processing-in-memory-llm](concepts/processing-in-memory-llm.md), [memory-centric-computing](concepts/memory-centric-computing.md), [on-device-llm-inference](concepts/on-device-llm-inference.md), [llm-serving](concepts/llm-serving.md), [llm-weight-quantization](concepts/llm-weight-quantization.md), [kv-cache-management](concepts/kv-cache-management.md), [speculative-decoding](concepts/speculative-decoding.md)
