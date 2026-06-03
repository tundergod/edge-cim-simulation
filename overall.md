# CIM-Centric LLM Inference on Heterogeneous Mobile SoC — Simulator

> **Status: preliminary project brief.** The plans, module breakdown, and architecture below are an *initial* design, not a contract — revise freely as characterization data and implementation realities dictate. The fixed points are the **research goal** (a real-silicon-calibrated simulator of LLM inference on a CIM-enabled heterogeneous mobile SoC) and the **two real boards** we calibrate against.

**Workload:** LLM inference (Llama-3 + Qwen-2.5 families, 1B–8B with 13B stretch, prefill + decode).
**Platform:** **Simulated** CIM-enabled heterogeneous mobile SoC (CIM on PCIe with 16 GB MMIO unified host memory), anchored on real Metis Alpha (Aetina RKC-A02) + production Metis Card silicon for cross-validation. Future-platform anchor: **Axelera Europa** (2026 H1 — 128 MB L2 SRAM + LPDDR5 + unified architecture, ~629 TOPS @ 45 W).
**Contribution:** TBD post-characterization; expected **CIM-centric** — CIM is fast and power-efficient but constrained, so the system is designed around CIM's limits first.

---

## Why a simulator

We want to study LLM on a CIM-mobile SoC, but no such silicon exists and the real Metis cards can't be the study object directly: Metis Alpha can't compute LLM at all (closed-firmware `-1301` wall + no on-card DRAM, only a 1 GB IOMMU window), and the production Metis Card runs LLM only through a closed, precompiled-only toolchain. The simulator side-steps this by making the **thing we study** a *simulated* CIM-mobile SoC, with real Metis silicon as **calibration ground truth**:
- **Metis Alpha (Aetina):** CIM compute primitives + PCIe behavior on CNN/matmul workloads.
- **Production Metis Card:** real LLM behavior at INT8 on the on-card-DRAM topology.

Real-silicon details: [papers/metis-silicon/](papers/metis-silicon/). SDK measurement surface: [voyager-sdk.md](voyager-sdk.md).

## Problem

Commercial discrete CIM accelerators (Axelera Metis, Hailo-8/10, Mythic, Untether) are exclusively packaged as PCIe / M.2 add-in cards. Mobile SoCs separately integrate NPU + GPU + CPU around shared LPDDR. The combination — **CIM as a peer compute unit on a heterogeneous mobile SoC with unified memory access** — is the **architecture trajectory** (Axelera Europa 2026 H1 confirms: 128 MB L2 SRAM + LPDDR5 + "unified architecture" + integrated decoder) but **literature has near-zero work** on running LLM on this combination.

Adjacent prior art exists in fragments (notes in [papers/pim-llm-accelerators/](papers/pim-llm-accelerators/)):
- HBM-PIM / GDDR-PIM LLM accelerators (NeuPIMs, IANUS, AttAcc, CENT, HPIM, PAPI) — server-tier; not mobile, not SRAM-CIM
- LPDDR-PIM mobile (LP-Spec) — LPDDR-internal PIM; not discrete CIM
- Compute-enabled flash (Lincoln, Cambricon-LLM) — flash substrate; not SRAM-CIM
- HeteroInfer (SOSP'25) — characterizes GPU+NPU mobile heterogeneous LLM; no CIM
- Our own Metis measurements ([metis-llm-investigation](papers/metis-silicon/metis-llm-investigation-desktop-2026-05-19.md), [metis-step1-cnn-characterization](papers/metis-silicon/metis-step1-cnn-characterization-2026-05-23.md)) — real silicon but vendor-closed for LLM compute extension

**The gap is precisely (discrete-CIM × heterogeneous-mobile-SoC × LLM × real-silicon-calibrated)** — a near-empty cell in 2024–2026 top-venue literature.

## Position

**CIM-centric design discipline:** the system is **designed around CIM's constraints first** (fast at weight-stationary GEMV, weak at dynamic attention, tile-aligned to channel-multiple-of-64, weight residency limits, host-device round-trip tax). GPU / NPU / CPU are **support layers** for ops CIM can't or shouldn't do (attention with dynamic K, RMSNorm, sampling, KV-cache management, mixed-precision boundaries).

**The specific op-level split is left to characterization** — we don't pre-commit to "CIM does MLP, others do attention" (what HPIM and most existing work assume); measured per-unit characteristic curves decide the split. This is the HeteroInfer methodological pattern applied to a new platform class.

**Differentiators from concurrent work:**
- **HPIM (closest competitor — [paper note](papers/pim-llm-accelerators/hpim-arxiv2025.md)):** heterogeneous SRAM-PIM + HBM-PIM, simulator-only, FP16-only, no energy/area, no mobile. We do SRAM-CIM + GPU + NPU + CPU on mobile-SoC with real-silicon calibration, INT8 (Metis), mixed-precision (CIM INT8 × GPU FP16).
- **PAPI:** dynamic GPU+PIM scheduling, server-tier, FC-PIM/Attn-PIM split. We're mobile + CIM-centric (not GPU-centric) + characterization-driven, not pre-committed split.
- **LP-Spec:** LPDDR-PIM mobile, NPU+PIM, speculative decode. We're CIM (not LPDDR-PIM) + general decode (not speculative-only).
- **Lincoln:** flash-PIM 50–100B consumer. Different substrate (flash vs SRAM-CIM), different regime (very large vs 1–13B), same on-device LLM aspiration.

## Platform assumption

**Stance: operational + descriptive.** Treat "CIM-on-PCIe + MMIO unified host memory" as the simulator's given platform assumption.

- **Descriptive evidence:** Metis Alpha already implements MMIO unification at 1 GB (IOMMU window mapping host LPDDR4 visible to the CIM core); Voyager already uses `dma-buf` + `cl_khr_external_memory_dma_buf` for copy-free buffer sharing.
- **Forward-looking anchor:** Axelera Europa (2026 H1) — 128 MB on-chip L2 SRAM, LPDDR5 256-bit @ ~200 GB/s, "unified architecture" with RISC-V vector cores + HEVC265 decoder.
- **Conservative size:** 16 GB unified, chosen to fit 7B INT8 + KV cache + activations; below Axelera M.2 Max (already 16 GB on-card).
- **Community signal:** Axelera forum confirms request-acknowledged for larger unified memory (no committed roadmap). Note: **no unified host-device memory exists on *current* Metis** — this is a forward assumption, not a present capability.

Bridging assumption written into the Method section: *"CIM-tile compute timing is invariant across memory-substrate change; only data-movement timing changes. Production Metis Card vendor LLM measurements validate CIM + on-card-DRAM topology; our simulator substitutes the on-card DRAM with a host-LPDDR + PCIe model parameterized from Metis Alpha measurements."*

## Workload scope

| Axis | Selection |
| --- | --- |
| **Model families** | **Llama-3 + Qwen-2.5** |
| **Model sizes** | **1B / 3B / 7-8B** (Llama-3.2-1B, Phi-3-mini-class, Llama-3.1-8B / Qwen-2.5-7B). Stretch: **13B** |
| **Phases** | **Prefill + Decode end-to-end** |
| **Precision** | **INT8 on Metis CIM** |
| **🔥 Mixed-precision** | **Main research surface, not ablation.** Units have different native precisions (CIM INT8, NPU INT8/INT16/FP16, Mali FP16, CPU any). Precision-boundary management between **CIM-MLP (INT8) and GPU-attention (FP16)** is a structurally novel problem. |
| **Context length** | 2K base + 8K stretch |
| **Batch** | **batch=1, AIPU Mode 1** (single-instance). Simulator keeps a `batch` hook for extensibility. |

## Phase 0 — Real-board Characterization

**Phase 0 runs first on the production Metis Card + Aetina; Simulation Implementation only starts once Phase 0 measurements are committed to the repo.** This phase produces all ground-truth data the simulator validates against (L1–L6 below), so cross-validation feasibility is *settled in Phase 0*, not deferred.

**Approach: decompose into chip-level invariants + workload-level translations.** Phase 0 splits across two machines, two independent agent handoffs. Each machine's section is self-contained (tooling, targets, output files, sweep matrix).

### Machine 1 — Aetina RKC-A02 (RK3588 + Metis Alpha)

**Agent role:** characterize the heterogeneous SoC's four compute units (CIM Alpha, RKNPU2, Mali, CPU) + the host-device PCIe boundary. Commits to `measurements/aetina/`.

**Required:** Voyager SDK v1.3.1 (Metis Alpha), RKNN toolkit (RKNPU2), OpenCL driver (Mali), `perf`, `eBPF`/`bpftrace`, `taskset`, `chrt`.

| Unit | What to measure | Output file |
| --- | --- | --- |
| **A. Metis Alpha** | (A1) CIM tile micro-benchmark via single-op ONNX differential method — sweep conv shape: in/out ch, H/W. (A2) `dpu_constants_home: l2` vs `ddr` timing diff → SRAM vs PCIe traffic ratio. (A3) Mode 1 per-call DMA stage breakdown (eBPF + LD_PRELOAD). (A4) LLM-relevant matmul micro-benchmark | `metis_alpha_{cnn_proxy,matmul,pcie}.json` |
| **C. RKNPU2** | Matmul micro-benchmark across LLM-relevant shapes (hidden 2048/4096/8192, seq 1/256/2048), INT8 + INT16 + FP16, batch sweep | `rknpu2_matmul.json` |
| **D. Mali** | Self-written OpenCL matmul kernel (avoid framework noise), FP16 primary + FP32 reference | `mali_matmul.json` |
| **E. CPU (RK3588 A76)** | On-target micro-benchmark of LLM CPU-support ops (sampling, RoPE control, KV-cache append/evict, token/quantization boundary). `taskset -c 4-7 chrt -f 50`; `clock_gettime` + `perf stat` | `cpu_ops.json` |

Per-unit characteristic sweep (HeteroInfer Fig 2-4 style): tile/channel size mismatch (ch ∈ {64…1024}), batch (Metis fixed 1; RKNPU2/Mali 1/4/16/32), weight residency (`l2` vs `ddr`), op-type sensitivity, precision, sequence-dim variation (hidden × seq, seq 1→2048).

**Deliverables:** four `measurements/aetina/*.json`; `variance_profile.json`; `characterization/aetina/README.md` (script→file mapping); final report `docs/phase0-aetina-findings.md` (SDK surprises, op-coverage gaps, simulator recommendations).

### Machine 2 — Ubuntu + production Metis Card

**Agent role:** capture the real CIM-running-LLM anchor. Commits to `measurements/metis_card/`.

**Required:** Voyager SDK (v1.6) with LLM precompiled artifacts (Llama-3 family), Python LLM benchmarking harness.

| Unit | What to measure | Output file |
| --- | --- | --- |
| **B. Metis Card** | Vendor pre-compiled INT8 LLM: Llama-3.2-1B / 3B / 8B end-to-end tok/s + per-token latency + 4-core utilization + context-length sweep (2K / 4K / 8K) via `axllm --show-stats` | `vendor_llm_int8.json` |

**Deliverables:** `vendor_llm_int8.json`; `variance_profile.json`; `characterization/metis_card/README.md`; final report `docs/phase0-metis-card-findings.md` (per-model tok/s, scaling with context, vendor-SDK behaviors).

### Measurement protocol — two-stage sampling

- **Stage 0** (variance, half-day): per unit, representative ops, cold-start repeats × per-run iterations; compute Coefficient of Variation (CoV); persist `measurements/{unit}/variance_profile.json`.
- **Stage 1** (production): use the Stage-0-derived sample plan across the full op × shape × precision × batch sweep; document sample size in `validation/contracts/m{M}.yaml`.

### Cross-validation matrix (L1–L6) produced by Phase 0

| L | Simulator validates against | Phase 0 data source |
| --- | --- | --- |
| L1 | CIM tile per-op latency | A1 + A4 + Stage sweep on Metis Alpha (trace-driven lookup primary; NeuroSim physics cross-check optional) |
| L2 | DRAM / PCIe round-trip | A2 + A3 (PCIe + DMA mode timing) |
| L3 | NPU / GPU per-op | C + D matmul micro-benchmarks |
| L4 | End-to-end LLM (INT8) | B Metis Card vendor INT8 LLM tok/s. Caveat: on-card DRAM ≠ simulator's host-MMIO topology (bridging assumption explicit in Method) |
| L5 | Sensitivity (±20% any parameter) | computed against L1–L4 during sim runs — not a separate measurement |
| **L6** | End-to-end CNN | [metis-step1-cnn-characterization](papers/metis-silicon/metis-step1-cnn-characterization-2026-05-23.md) — 225 cells already captured; **reused, no re-measurement** |

**Roofline as validation visualization** (L1/L3/L6): per unit, plot measured-roofline vs simulator-predicted-roofline; knee-location + slope + observation-point match is the 2D consistency check on compute + memory primitives simultaneously. Data points extracted during Phase 0 from the same runs.

**Mixed-precision validation:** only the simplest direct case (e.g. CIM INT8 + GPU FP16 split on one model). Mixed-precision is the method/contribution; its validation is simplified by design.

### Phase 0 success criteria

Done when **both** machines have committed: all `measurements/aetina/*` (4 files) + `measurements/metis_card/*` (1 file); `variance_profile.json` per machine; per-machine `characterization/{aetina,metis_card}/README.md`; per-machine `docs/phase0-{aetina,metis_card}-findings.md`; and a combined `docs/phase0-L1-L6-mapping.md`. After both reports commit, **Simulation Implementation kicks off** on the Metis Card machine.

---

## Simulation Implementation

**Prerequisite: Phase 0 measurements committed.** Operational handoff for building the simulator on top of Phase 0 ground truth. Designed to be offloaded to a (mostly) autonomous agent after initial setup.

### Approach — autoresearch pattern, minimal infrastructure

**1 LLM agent, 0 external orchestrator, Python validator script.** The agent itself drives the loop. Sketch:
```
while not all_modules_passed:
    read program.md, HANDOFF.md, log.jsonl, validation/contracts/, measurements/
    pick next module M (per dependency graph + log state)
    modify simulator/modules/m{M}.py
    invoke: python simulator/runner.py --module M
    read validation_result.json
    if passed:        log; run regression on Mi<M; if ok, advance
    elif retryable:   log; diagnose; modify; retry
    elif stuck:       flag for human; skip; advance
    update HANDOFF.md (state for next session)
```

### Simulator architecture (6 boxes + data flow)

Layered, modular, event-driven. Each box has one upstream input, one validation contract, and one measurement source (or composition of upstream measurements) — so the agent can iterate and validate one box at a time.

```
① Workload generator (M5)
   HuggingFace model → torch.onnx.export → per-token op DAG
   Llama-3 / Qwen-2.5 1B–8B (13B stretch); prefill+decode; batch=1; ONNXim-aligned trace
            │ op stream + tensor metadata
            ▼
② Scheduler / Mapper (M6)  ── CONTRIBUTION LAYER ──
   Per-op: unit (CIM/NPU/GPU/CPU) + precision + memory placement + dataflow
   + pipeline + precision-boundary insertion + resource-constraint check
   Plugin interface: baseline strategy ↔ proposed strategy
            │ (op, target_unit, precision) tuples
            ▼
③ Per-unit timing models (M1 + M4)   event-driven, parallel
   CIM (Metis): trace-driven lookup from Metis Alpha measurements
   NPU (RKNPU2): ONNXim fork + lookup override for divergent shapes
   GPU (Mali): trace-driven lookup from Mali OpenCL measurements
   CPU: ARM A76 instruction-count model
   (`gpu_backend` plugin slot reserved for Accel-Sim)
            │ memory access pattern + latency
            ▼
④ Memory hierarchy (M2 + M3)
   Metis L1 SPM (4 MB×core) ─ L2 SRAM (32 MB shared)
   Host LPDDR5 — Ramulator2 backend; PCIe Gen3 ×4 DMA model (BW + latency + setup)
   TLB-miss penalty (parameterized, default 0); on-SoC LPDDR shared by RKNPU2/Mali/CPU
            │ per-op time + memory access counts
            ▼
⑤ Energy estimation (M7)
   Metis CIM: vendor 15 TOPS/W × utilization; CPU A76: ARM datasheet × activity
   RKNPU2/Mali: INA-delta OR tech-node-derived; Memory: per-access JEDEC; PCIe: per-byte spec
            │ per-inference latency + energy
            ▼
⑥ Output + Inline Validation
   End-to-end latency / throughput / energy-per-inference; per-op timeline + roofline
   Inline comparator: predicted vs measured per box → writes validation_result.json
```

**Implementation language:** Python (event loop self-written; Ramulator2 via Python bindings; ONNXim fork as subprocess).

### Repo structure (target)

```
edge-cim-simulation/
├── overall.md                  # this brief
├── voyager-sdk.md              # SDK characterization reference (all agents)
├── README.md
├── program.md                  # agent's primary instructions (template below)
├── HANDOFF.md                  # cross-session state
├── log.jsonl                   # per-iteration log (append-only)
├── papers/                     # literature + real-silicon notes (this commit)
├── simulator/
│   ├── modules/                # M1–M7 (m1_cim_tile.py … m7_energy.py)
│   ├── runner.py               # entry: python runner.py --module M
│   ├── validator.py            # compares output to measurements
│   └── lib/
├── measurements/               # ground truth, version-controlled
│   ├── aetina/                 # metis_alpha_{cnn_proxy,matmul,pcie}, rknpu2_matmul, mali_matmul, cpu_ops
│   └── metis_card/             # vendor_llm_int8.json
├── characterization/           # scripts to (re-)capture measurements (aetina/, metis_card/)
├── validation/contracts/       # per-module validation spec YAML (m1.yaml …)
├── tools/                      # analysis/, plotting/ (roofline), trace_export/
├── tests/
└── docs/                       # human-facing (architecture, protocol, phase0 findings)
```

### Two-machine division — three agent handoffs

| Phase | Machine | Agent role |
| --- | --- | --- |
| Phase 0 — Aetina | Aetina RKC-A02 | A/C/D/E measurements → `measurements/aetina/*` + findings |
| Phase 0 — Metis Card | Ubuntu + Metis Card | B measurement → `measurements/metis_card/*` + findings |
| Sim Implementation | Ubuntu + Metis Card | autoresearch loop: M1–M7. Pulls Phase 0 data; iterates against `validation/contracts/*`. May SSH to Aetina for re-captures |

The two Phase 0 agents run in parallel (independent machines/domains). Sim Implementation starts only after both Phase 0 reports commit. Sync via git push/pull on the shared repo.

### SSH access — Sim agent → Aetina (re-capture path only)

During Sim Implementation, if a shape/precision/config isn't in `measurements/aetina/`, trigger a remote re-capture:
```bash
ssh aetina 'cd ~/repo/characterization/aetina && ./run_metis_matmul.py --config tier1'
rsync aetina:~/repo/measurements/aetina/ ./measurements/aetina/
git add measurements/aetina/ && git commit -m "char: new shapes" && git push
```
One-time setup: ed25519 key, `ssh-copy-id`, `~/.ssh/config` Host entry. All code dev happens on the Metis Card machine; Aetina is a measurement workhorse driven remotely.

### program.md template (sketch)

```markdown
# Project: CIM-Centric LLM Inference Simulator
## Goal
Implement and validate a simulator for LLM inference on a CIM-enabled heterogeneous
mobile SoC. Iterate one module at a time until all pass against measurements/ ground truth.
## How to work
1. Read HANDOFF.md (else log.jsonl tail) for current state.
2. Read validation/contracts/m{M}.yaml for the target module.
3. Read relevant measurements/*.json as ground truth.
4. Edit simulator/modules/m{M}.py; run `python simulator/runner.py --module m{M}`.
5. Read validator output. If passed: regression `--up-to m{M}`; if ok, advance.
   If not: append analysis to log.jsonl, hypothesize, modify, retry (max 20/session).
## Module dependency graph
M1(CIM tile)←metis_alpha_*  ·  M2(memory)←metis_alpha_pcie+Ramulator2  ·  M3(event engine)←M1+M2
M4(NPU/GPU/CPU)←rknpu2+mali+cpu_ops  ·  M5(workload)←HF+torch.onnx.export  ·  M6(scheduler)←M3+M4+M5  ·  M7(energy)←M1..M6
## End-of-session: update HANDOFF.md (module, last status, blockers, next steps).
```

### Validation contract template (per module)

```yaml
module: m1_cim_tile
measurement_sources:
  - measurements/aetina/metis_alpha_cnn_proxy.json
  - measurements/aetina/metis_alpha_matmul.json
acceptance_criteria:
  - {type: median_op_error, threshold: 10%}        # guidance, not hard commitment
  - {type: roofline_shape_match, metric: knee_position_drift, threshold: 15%}
  - {type: sanity, rules: [no_nan_or_inf, monotonic_with_op_size, latency_positive]}
sample_strategy: {cold_starts: 3, iterations_per_run: 300, budget_seconds: 30}
```

### Modules

| M | Module | Primary measurement source | Notes |
| --- | --- | --- | --- |
| M1 | CIM tile timing | Metis Alpha CNN + matmul micro-benchmark | Trace-driven lookup primary; NeuroSim optional cross-check |
| M2 | Memory hierarchy | Ramulator2 LPDDR5 + Metis Alpha PCIe DMA | TLB-miss penalty parameterized (default 0) |
| M3 | Event-driven engine | M1 + M2 | Python event loop; orchestrates op stream through units + memory |
| M4 | NPU / GPU / CPU traces | RKNPU2 matmul, Mali OpenCL matmul, CPU A76 | NPU = ONNXim fork + lookup override; GPU/CPU = lookup only |
| M5 | LLM workload generator | HF Llama-3 / Qwen-2.5 → torch.onnx.export → per-token op DAG | Trace format aligned with ONNXim input |
| M6 | Scheduler / Mapper | M3 + M4 + M5 | Plugin: op→unit + memory + dataflow + pipeline + precision-boundary insertion. **Contribution lives here.** |
| M7 | Energy estimation | Vendor specs (Metis 15 TOPS/W), ARM datasheet, INA-delta or tech-node | Spec-based + activity-factor estimation |

## Open risks

1. **NeuroSim integration overhead exceeds estimate** — M1 retreat: pure trace-driven lookup of Metis Alpha measurements. NeuroSim drops from required to optional; the validation-methodology citation (NeuroSim <1% chip error) remains valid even without using its code.
2. **Bridging assumption: Metis Card on-card DRAM ≠ simulator's host-MMIO topology** — L4 anchors "CIM + on-card-DRAM"; simulator substitutes host-LPDDR + PCIe. Sensitivity sub-experiment under both topologies required.
3. **HPIM publishes at a top venue first** (closest competitor, [paper note](papers/pim-llm-accelerators/hpim-arxiv2025.md)). Differentiators (mobile-SoC, real-silicon calibration, mixed-precision, characterization-driven split) hold even if HPIM lands first.
4. **Agent autonomy at simulator-dev scale untested** — first M1 iteration is the real test. Mitigation: retreat to manual dev if the agent fails to converge after N sessions.
5. **HuggingFace ONNX export quality** — `torch.onnx.export(Llama-3 / Qwen-2.5)` is notoriously messy (custom ops, dynamic shapes). M5 may need manual post-processing or a different extraction tool. Verify before relying on it.
6. **Ramulator2 LPDDR5 + PIM-like extension coverage** — our LPDDR5-PIM-like usage is not Ramulator2's default; may need custom plug-ins. Budget for it in M2.
7. **ONNXim RKNPU2 fit** — ONNXim models generic systolic NPU; RKNPU2 has Rockchip-specific behaviors (op-mix sensitivity, depthwise+Swish weakness — Step-1 data). Plan B: lookup-table override (already in M4 design).
8. **SSH availability of Aetina** — must be reachable throughout sim dev. If offline, flag blocker and proceed on cached measurements.

## Out of scope (v1 paper)

- **INT4 on Metis CIM** — Voyager public docs don't expose user-controlled INT4. Future work if the SDK opens or a vendor INT4 artifact appears.
- **AIPU Mode 2 (4-instance) / Mode 3 (compiler-batched)** — require 4× weight footprint or static-shape batched compile; don't fit single-batch dynamic-shape LLM on 16 GB unified. Future work for server-like batched scenarios.
- **batch > 1** — mobile single-batch is paper scope. Simulator keeps a `batch` hook.
- **NVIDIA GPU baseline (Accel-Sim) / Jetson Orin / Nano** — interface-modular extension to M4; `gpu_backend` plugin slot reserved. Future generalization study.
- **Thermal modeling** — device-dependent, not generalizable; boards lack on-board power instrumentation.
- **Energy as measurement** — replaced by spec-based estimation (M7), same instrumentation reason.
- **Intra-frame multi-core CIM parallelism** — `cooperative` / `pipeline` modes not implemented in Voyager v1.3.1. Future SDK may enable.

---

## References (in this repo)

- **Closest competitor:** [HPIM](papers/pim-llm-accelerators/hpim-arxiv2025.md)
- **Direct prior art (PIM-LLM accelerators):** [papers/pim-llm-accelerators/](papers/pim-llm-accelerators/) — PAPI, SpecPIM, NeuPIMs, IANUS, CENT, LP-Spec, CXL-PNM, Lincoln, Cambricon-LLM, and more
- **Methodology / simulators:** [papers/methodology-and-simulators/](papers/methodology-and-simulators/) — HeteroInfer (SOSP'25, characterization pattern), NeuroSim validation (D4 pattern), DNN-NeuroSim v1/v2, gem5-SALAM (assessed, not chosen)
- **On-device LLM:** [papers/on-device-llm/](papers/on-device-llm/) — PowerInfer-2, fast-on-device-LLM-NPU, LLM-in-a-flash, KVSwap
- **Real-silicon calibration sources:** [papers/metis-silicon/](papers/metis-silicon/) — Step-1 CNN characterization (L6), Metis Card LLM investigation (L4), Aetina board audit, AIPU NN-v2 direction report
- **Platforms:** [papers/platforms/](papers/platforms/) — Aetina RKC-A02, Axelera Metis Card
- **Concepts:** [papers/concepts/](papers/concepts/) — CIM / PIM-LLM / SRAM-IMC / on-device-LLM / KV-cache / quantization / speculative-decoding
- **Calibration-source idea:** [cnn-dnn-edge-memory-wall-metis-embedded](papers/ideas/cnn-dnn-edge-memory-wall-metis-embedded.md)
- **SDK measurement surface:** [voyager-sdk.md](voyager-sdk.md)
