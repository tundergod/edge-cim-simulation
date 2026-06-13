# edge-cim-simulation

A **real-silicon-calibrated simulator** of LLM inference on a **CIM-enabled heterogeneous mobile
SoC** — studying Compute-in-Memory (CIM) as a first-class compute unit alongside NPU, GPU, and CPU
(INT8, batch=1, prefill + decode).

The simulated target is a CIM accelerator on PCIe + MMIO unified host memory (capacity is a tunable
parameter, not a hard-coded value), calibrated against **two real Axelera Metis boards**:

- **Metis Alpha** (Aetina RKC-A02) — CIM / PCIe / NPU / GPU / CPU micro-benchmarks.
- **Production Metis Card** — end-to-end INT8 LLM behaviour under on-card DRAM (the L4 anchor).

## Design principle — CIM-centric

The system is designed around CIM's constraints *first* (strong at weight-stationary GEMV, weak at
dynamic attention, tile alignment to channel×64, weight-residency limits, host↔device transfer
cost); GPU / NPU / CPU are the support layer. The op→unit split is decided by **characterization
measurements, not assumed**.

## Scope

| Axis | Choice |
| --- | --- |
| Models | Llama-3 + Qwen-2.5, 1B–8B (Llama-3.2-1B / 3B, Llama-3.1-8B, Qwen-2.5-7B; 13B stretch) |
| Phase / precision | Prefill + decode, end-to-end; INT8 on CIM |
| Mixed precision | Per-unit native precision (CIM INT8, NPU INT8/16/FP16, Mali FP16). The CIM-MLP(INT8) × GPU-attention(FP16) boundary is a core research question |
| Context / batch | 2K baseline (8K stretch); batch = 1 |

Workloads span the prefill-heavy ↔ decode-heavy spectrum (ShareGPT, GSM8K, LongBench-TriviaQA,
HumanEval) plus a synthetic length sweep. Out of scope: training, MoE/sparse, FP32, multi-batch —
see [OVERALL.md](OVERALL.md).

## Status

- **Done:** Phase 0.1–0.3 (trace / op-profile / on-board measurement) and **Phase 1.1–1.3**
  (per-component modelling + validation: CIM, memory, CPU, GPU, NPU, plus Ramulator2 / ONNXim
  heavy-sim drop-ins behind a frozen `engine=` interface).
- **Reinforcement** (補強 — *not* new phases): 1.4–1.6b are folded into 1.1 / 1.2 / 1.3 (CIM Card
  re-measurement, re-reviews, the ScaleSim third NPU engine, and an honest NPU-characteristic
  measurement).
- **Phase 0.4 (thermal):** Metis Card measured; Aetina pending repair.
- **Next — Phase 2:** integrate the M3 event engine + M6 scheduler into an end-to-end
  prefill + decode simulator.

**Consolidated report:** [`docs/report/phase1-site/`](docs/report/phase1-site/) — a hand-coded
multi-page site where every number is injected from committed JSON (the build fails on any
unresolved placeholder) and every figure is regenerated from data. Known measurement gaps (no
RKNPU2 silicon for the NPU, issue #13; prefill / multi-tile) are labelled honestly.

## Honesty discipline

The simulator is calibrated against real silicon for *some* units and not others, so the repo is
strict about provenance: each value is tagged `[MEASURED]` / `[GAP]` / calibrated / simulated /
assumption / borrowed, and report numbers flow from committed JSON rather than prose. **Results must
never confirm their own assumptions** — no circular reasoning, no manufactured cross-source
agreement, no validation language without ground truth. See CLAUDE.md § *Honesty discipline*.

## Repository layout

```
CLAUDE.md  CONTEXT.md  OVERALL.md  README.md  LOG.md  requirements.phase0.txt
docs/              papers/ plans/ adr/ agents/ figures/ report/(phase1-site) + *-findings.md  voyager-sdk.md
simulator/         specs/ (swappable hardware json) · models/ (M1/M2/M4/M7 + engine.py + params/)
                   · engines/ (external heavy-sim caches: ONNXim / ScaleSim / Ramulator2)
characterization/  on-board measurement scripts (aetina/  metis_card/)
measurements/      silicon measurements + op inventory/profile (aetina/  metis_card/  op_inventory/  op_profile/)
traces/            per-token op×shape streams (per model, workload)
validation/        contracts/(m*.yaml acceptance criteria)  reports/(phase*)  validators
tools/             analysis/ (fits)  plotting/ (one script per figure)  report/  trace_export/  onnxim/  ramulator2/
tests/             pytest
```

## Build & test

```sh
pip install -r requirements.phase0.txt              # dependencies
python -m pytest tests/                              # unit tests
python docs/report/phase1-site/build.py --strict    # build the report — fails on any unresolved {{key}} or stale figure
python tools/plotting/site_npu.py                    # (e.g.) regenerate a figure group from committed data
```

## Key docs

- **[OVERALL.md](OVERALL.md)** — project brief: goal, phases (0.1→2), modules M1–M8, validation
  layers L1–L6, workloads, open risks.
- **[CONTEXT.md](CONTEXT.md)** — domain glossary + a directory-level **repo index** (where
  everything lives; consult it before grepping).
- **[CLAUDE.md](CLAUDE.md)** — working guidelines (per-phase workflow, simplicity, honesty discipline).
- **[docs/adr/](docs/adr/)** — architecture decision records (ADR-0001…0007).
- **[docs/voyager-sdk.md](docs/voyager-sdk.md)** — how to measure Metis (Voyager SDK reference,
  tagged `[DOC]` / `[FORUM]` / `[MEASURED]` / `[GAP]`).
- **[docs/papers/](docs/papers/)** — curated literature + real-silicon notes (16 papers).

## External references

- Voyager SDK — <https://github.com/axelera-ai-hub/voyager-sdk>
- Axelera community (Metis M.2) — <https://community.axelera.ai/metis-m-2-3>
