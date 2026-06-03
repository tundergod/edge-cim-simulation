# CONTEXT

Domain glossary for this repo. When naming a concept in an issue, plan, test, or proposal, use the term as defined here. Authoritative detail lives in [overall.md](overall.md) and [docs/voyager-sdk.md](docs/voyager-sdk.md).

## Glossary

- **CIM** — Compute-in-Memory. The study unit; on Metis it is **D-IMC** (Digital In-Memory Compute), INT8.
- **AIPU** — Axelera's quad-core Metis accelerator (the CIM device).
- **Metis Alpha** — pre-production M.2 on the Aetina board; no on-card DRAM; cannot compute LLM (`-1301` wall). Used for CIM/PCIe/NPU/GPU/CPU micro-benchmarks.
- **Metis Card** — production card that runs precompiled INT8 LLMs; decode is a ~24 GB/s on-card-DRAM memory wall. Source of the L4 anchor.
- **prefill / decode** — the two LLM inference phases. prefill = compute-bound (GEMM); decode = memory-bound (GEMV). The CIM compute-bound↔memory-bound split is the core axis.
- **unit** — a compute unit in the simulated SoC: CIM / NPU (RKNPU2) / GPU (Mali) / CPU (A76).
- **M1–M8** — simulator modules (M1 CIM tile, M2 memory, M3 event engine, M4 NPU/GPU/CPU, M5 workload/trace gen, M6 scheduler/mapper, M7 energy, M8 thermal-optional). See `overall.md`.
- **L1–L6** — cross-validation layers (what the simulator is checked against). See `overall.md`.
- **op inventory** — the distinct op types + shape parametrization a model executes (Phase 0.1 output).
- **trace** — the ordered per-token op×shape stream for a (model, prefill_len, decode_len) run.
- **workload** — the actual inputs (tasks/datasets + length profiles), not just the model. Layer A: ShareGPT / GSM8K / LongBench-TriviaQA / HumanEval. Layer B: synthetic (prefill × decode) sweep.
- **Phase 0.1 / 0.2 / 0.3 / 1 / 2** — trace+op gen / board measurement (no temp) / thermal / per-component equation-fit & validation / simulator integration.

Architectural decisions, once made, are recorded as ADRs under `docs/adr/` (created lazily).
