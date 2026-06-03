---
type: idea
title: "CIM-Centric LLM Inference on Heterogeneous Mobile SoC"
created: 2026-05-25
updated: 2026-05-26
tags: [research-idea, llm, cim, sram-cim, mobile-soc, heterogeneous-computing, simulator, mixed-precision, axelera, europa, hpca, asplos, micro]
priority: high
status: active
---

# CIM-Centric LLM Inference on Heterogeneous Mobile SoC

**Priority:** High — new top-3 idea (replaces [[cnn-dnn-edge-memory-wall-metis-embedded]] in High slot, which is reframed as calibration source).
**Workload:** LLM inference (Llama-3 + Qwen-2.5 families, 1B–8B with 13B stretch, prefill + decode).
**Platform:** **Simulated** CIM-enabled heterogeneous mobile SoC (CIM on PCIe with 16 GB MMIO unified host memory) — anchored on real Metis Alpha (Aetina RKC-A02) + Metis Card silicon for cross-validation. Future-platform anchor: **Axelera Europa** (2026 H1, 128 MB L2 SRAM + LPDDR5 + unified architecture, 629 TOPS @ 45 W).
**Contribution:** TBD post-characterization; expected to be **CIM-centric** (CIM is fast + power-efficient with limits, so system design works around CIM's constraints first).

---

## Origin

Pivoted from [[cnn-dnn-edge-memory-wall-metis-embedded]] on 2026-05-25 after grill-me session. Old idea (CNN/DNN edge memory wall on real Aetina + Metis Alpha) is reframed as **calibration data source** for this new idea — Step-1 measurement campaign data feeds simulator validation; the pivot is workload-and-platform but the real-silicon characterization remains directly usable.

The pivot resolves a contradiction: we wanted LLM but Metis Alpha can't compute LLM (firmware -1301 wall + 1 GB IOMMU window). The simulator strategy **side-steps this** by making the "thing we study" a simulated CIM-mobile-SoC, with real Metis silicon serving as calibration ground truth (Metis Alpha: CIM compute primitives + PCIe behavior on CNN workloads; Metis Card: real LLM behavior at INT8 on on-card DRAM topology).

## Problem

Commercial discrete CIM accelerators (Axelera Metis, Hailo-8/10, Mythic, Untether) are exclusively packaged as PCIe / M.2 add-in cards. Mobile SoCs separately integrate NPU + GPU + CPU around shared LPDDR memory. The combination — **CIM as a peer compute unit on a heterogeneous mobile SoC with unified memory access** — is the **architecture trajectory** (Axelera Europa 2026 H1 confirms: 128 MB L2 SRAM + LPDDR5 + "unified architecture" + integrated decoder) but **literature has zero work** on how to run LLM on this combination.

Adjacent prior art exists in fragments:
- HBM-PIM / GDDR-PIM LLM accelerators (NeuPIMs, IANUS, AttAcc, CENT, HPIM, PAPI) — server-tier; not mobile, not SRAM-CIM
- LPDDR-PIM mobile (LP-Spec) — uses LPDDR-internal PIM; not discrete CIM
- Compute-enabled flash (Lincoln, Cambricon-LLM) — flash substrate; not SRAM-CIM
- HeteroInfer (SOSP'25) — characterizes GPU+NPU mobile heterogeneous LLM; no CIM
- Metis own measurements ([[metis-llm-investigation-desktop-2026-05-19]], [[metis-step1-cnn-characterization-2026-05-23]]) — real silicon but vendor-closed for LLM compute extension

**The gap is precisely (discrete-CIM × heterogeneous-mobile-SoC × LLM × real-silicon-calibrated)** — a near-empty cell in 2024-2026 top-venue literature.

## Position

**CIM-centric design discipline**: the system is **designed around CIM's constraints first** (e.g., fast at weight-stationary GEMV, weak at dynamic attention, tile-aligned to 512×512 multiples, weight residency limits, host-device round-trip tax). GPU / NPU / CPU are **support layers** that handle ops CIM can't or shouldn't (attention with dynamic K, RMSNorm, sampling, KV cache management, mixed-precision boundaries).

**The specific op-level split is left to characterization** — we don't pre-commit to "CIM does MLP, others do attention" (which is what HPIM and most existing work assume); we let measured per-unit characteristic curves decide the split. This is the HeteroInfer methodological pattern applied to a new platform class.

**Differentiator from concurrent work**:
- **HPIM (closest competitor)**: heterogeneous SRAM-PIM + HBM-PIM, simulator-only, FP16-only, no energy/area, no mobile. We do SRAM-CIM + GPU + NPU + CPU on mobile-SoC with real-silicon calibration, INT8 (Metis), mixed-precision (CIM INT8 × GPU FP16).
- **PAPI**: dynamic GPU+PIM scheduling, server-tier, FC-PIM/Attn-PIM split. We're mobile + CIM-centric (not GPU-centric) + characterization-driven not pre-committed split.
- **LP-Spec**: LPDDR-PIM mobile, NPU+PIM, speculative decode. We're CIM (not LPDDR-PIM) + general decode (not speculative-only).
- **Lincoln**: flash-PIM 50–100B consumer. Different substrate (flash vs SRAM-CIM), different model regime (very large vs 1–13B), but same on-device LLM aspiration.

## Platform assumption

**Stance: (iii) operational + (i) descriptive.** Treat "CIM-on-PCIe + MMIO unified host memory" as the simulator's given platform assumption.

- **Descriptive evidence**: Metis Alpha already implements MMIO unification at 1 GB (IOMMU window mapping host LPDDR4 visible to CIM core); Voyager SDK already uses `dma-buf` + `cl_khr_external_memory_dma_buf` for buffer sharing without copy
- **Forward-looking anchor**: **Axelera Europa (2026 H1)** — 128 MB on-chip L2 SRAM, LPDDR5 256-bit @ 200 GB/s, integrated "unified architecture" with 16 RISC-V vector cores + HEVC265 decoder
- **Conservative size**: 16 GB unified is chosen to fit 7B INT8 + KV cache + activations; below Axelera M.2 Max (already 16 GB on-card)
- **Community signal**: Axelera forum confirms request-acknowledged for larger unified memory (no committed roadmap)

The bridging assumption we will write into Method section: "CIM-tile compute timing is invariant across memory substrate change; only data-movement timing changes. Metis Card vendor LLM measurements validate CIM + on-card DRAM topology; our simulator substitutes the on-card DRAM with a host-LPDDR + PCIe model parameterized from Metis Alpha measurements."

## Workload scope 

| Axis                   | Selection                                                                                                                                                                                                                                               |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Model families**     | **Llama-3 + Qwen-2.5**                                                                                                                                                                                                                                  |
| **Model sizes**        | **1B / 3B / 7-8B** (Llama-3.2-1B, Phi-3-mini-class, Llama-3.1-8B / Qwen-2.5-7B). Stretch: **13B**                                                                                                                                                       |
| **Phases**             | **Prefill + Decode end-to-end**                                                                                                                                                                                                                         |
| **Precision**          | **INT8 on Metis CIM**                                                                                                                                                                                                                                   |
| **🔥 Mixed-precision** | **Main research surface, not ablation.** Units have different native precisions (CIM INT8, NPU INT8/INT16/FP16, Mali FP16, CPU any). Precision-boundary management between **CIM-MLP (INT8) and GPU-attention (FP16)** is a structurally novel problem. |
| **Context length**     | 2K base + 8K stretch                                                                                                                                                                                                                                    |
| **Batch**              | **batch=1, AIPU Mode 1** (single-instance). Simulator interface keeps a `batch` hook for extensibility.                                                                                                                                                 |

## Phase 0 — Real-board Characterization

**Phase 0 runs first on Metis Card + Aetina; Simulation Implementation only starts once Phase 0 measurements are committed to the repo.** This phase produces all ground-truth data the simulator validates against (L1–L6 below), so cross-validation feasibility is *settled in Phase 0*, not deferred to simulator completion.

**Approach (α): decomposition into chip-level invariants + workload-level translations.**

Phase 0 splits across **two machines, two independent agent handoffs**. Each machine's section below is self-contained for its own agent: required tooling, measurement targets, output files, sweep matrix.

### Machine 1 — Aetina RKC-A02 (RK3588 + Metis Alpha)

**Agent role**: characterize the heterogeneous mobile SoC's four compute units (CIM Alpha, RKNPU2, Mali, CPU) + the host-device PCIe boundary. Commits output to `measurements/aetina/` and pushes.

**Required on this machine**: Voyager SDK v1.3.1 (Metis Alpha), RKNN toolkit (RKNPU2), OpenCL driver (Mali), `perf`, `eBPF`/`bpftrace`, `taskset`, `chrt`.

#### Per-unit measurement plan (Aetina)

| Unit                        | What to measure                                                                                                                                                                                                                                                                          | Output file                                |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| **A. Metis Alpha**          | (A1) CIM tile micro-benchmark via single-op ONNX differential method — sweep conv shape: in_ch/out_ch/H/W. (A2) `dpu_constants_home: l2` vs `ddr` timing diff → SRAM vs PCIe traffic ratio. (A3) Mode 1 per-call DMA stage breakdown (eBPF + LD_PRELOAD). (A4) LLM-relevant matmul micro-benchmark | `measurements/aetina/metis_alpha_{cnn_proxy,matmul,pcie}.json` |
| **C. RKNPU2**               | Matmul micro-benchmark across LLM-relevant shapes (hidden 2048/4096/8192, seq 1/256/2048), INT8 + INT16 + FP16, batch sweep                                                                                                                                                              | `measurements/aetina/rknpu2_matmul.json`   |
| **D. Mali**                 | Self-written OpenCL matmul kernel (avoid framework noise) across same shape sweep, FP16 primary + FP32 reference                                                                                                                                                                          | `measurements/aetina/mali_matmul.json`     |
| **E. CPU (RK3588 A76)**     | Direct on-target micro-benchmark of LLM CPU-support ops (sampling, RoPE control, KV cache append/evict, token boundary, quantization boundary). `taskset -c 4-7 chrt -f 50`. Wall-clock `clock_gettime` + `perf stat -e instructions,cycles`                                              | `measurements/aetina/cpu_ops.json`         |

#### Per-unit characteristic sweep (HeteroInfer Fig 2-4 style)

| Dimension | Metis CIM (A) | RKNPU2 (C) | Mali (D) |
|---|---|---|---|
| Tile / channel size mismatch | sweep ch in {64, 128, 256, 384, 512, 768, 1024} | same | same |
| Batch size | **1 (Mode 1)** | 1 / 4 / 16 / 32 (Step-1 done) | same (Step-1 done) |
| Weight residency / locality | `l2` vs `ddr` home | weight size vs SRAM | same |
| Op type sensitivity | matmul / conv / depthwise / sigmoid-via-LUT | matmul / depthwise+Swish / activation | same |
| Precision | **INT8** | INT8/INT16/FP16 | FP16/FP32 |
| Sequence dim variation | hidden × seq matrix, seq 1 → 2048 | same | same |

#### Aetina agent deliverables

- All four `measurements/aetina/*.json` files committed
- `measurements/aetina/variance_profile.json` (Stage 0 output, per unit)
- `characterization/aetina/README.md` documents which script produced which file + invocation params
- Final report (markdown) appended to `docs/phase0-aetina-findings.md`: SDK behavior surprises (e.g., INT4 verification result, any unsupported ops), op-coverage gaps, recommendations for simulator

### Machine 2 — Ubuntu 24.04.4 + i9-12900K + Metis Card

**Agent role**: capture real CIM-running-LLM anchor on the production Metis card. Commits output to `measurements/metis_card/` and pushes.

**Required on this machine**: Voyager SDK with LLM precompiled artifacts (Llama-3 family), Python LLM benchmarking harness.

#### Per-unit measurement plan (Metis Card)

| Unit                          | What to measure                                                                                                                                                              | Output file                                 |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **B. Metis Card (production)**| Vendor pre-compiled INT8 LLM: **Llama-3.2-1B / 3B / 8B** end-to-end tok/s + per-token latency + 4-core utilization + context-length sweep (2K / 4K / 8K)                     | `measurements/metis_card/vendor_llm_int8.json` |

#### Ubuntu+Metis Card agent deliverables

- `measurements/metis_card/vendor_llm_int8.json` committed
- `measurements/metis_card/variance_profile.json` (Stage 0 output)
- `characterization/metis_card/README.md` documents the harness invocation
- Final report appended to `docs/phase0-metis-card-findings.md`: per-model tok/s, scaling with context, any vendor SDK behaviors worth flagging

### Measurement protocol — two-stage sampling

**Stage 0** (variance characterization, half-day): per unit, pick representative ops; run with cold-start repeats and per-run iterations; compute Coefficient of Variation (CoV); persist in `measurements/{unit}/variance_profile.json`.

**Stage 1** (production characterization): use Stage-0-derived sample plan; apply to full op × shape × precision × batch sweep; document the chosen sample size in `validation/contracts/m{M}.yaml`.

### Cross-validation data produced by Phase 0 (the L1–L6 matrix)

Phase 0 deliverables are the ground-truth datasets the simulator validates against. **All six rows are achievable in Phase 0** — cross-validation feasibility is verified by completing this phase, not gambled on simulator success.

| L      | What simulator validates against | Phase 0 data source                                                                                                          |
| ------ | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| L1     | CIM tile per-op latency          | A1 + A4 + Stage-0/1 sweep on Metis Alpha (trace-driven lookup primary; NeuroSim physics cross-check optional)                |
| L2     | DRAM / PCIe round-trip           | A2 + A3 on Metis Alpha PCIe + DMA mode timing                                                                                |
| L3     | NPU / GPU per-op                 | C + D matmul micro-benchmarks across LLM-relevant shapes                                                                     |
| L4     | End-to-end LLM (INT8)            | B Metis Card vendor INT8 LLM tok/s (Llama-3.2-1B / 3B / 8B). Caveat: on-card DRAM ≠ simulator's host-MMIO topology (bridging assumption explicit in paper Method) |
| L5     | Sensitivity (±20% any parameter) | (computed against L1–L4 datasets during simulator runs — not a separate measurement)                                          |
| **L6** | **End-to-end CNN (Step-1 data)** | [[metis-step1-cnn-characterization-2026-05-23]] — 5 CNN × 3 unit, 225 cells already captured;**reused, no re-measurement needed** |

**Roofline as validation visualization** (used by L1 / L3 / L6): for each unit, plot measured-roofline and (later) simulator-predicted-roofline; shape match (knee location + slope + observation point distribution) is the 2D consistency check on compute + memory primitives simultaneously. Roofline data points are extracted during Phase 0 from the same measurement runs (no extra captures needed).

**Mixed-precision validation**: only the simplest direct case (e.g., CIM INT8 + GPU FP16 split on one model). Mixed-precision is the method/contribution; validation simplified by design.

### Phase 0 success criteria (both machines)

Phase 0 is "done" when **both** machines have committed:
- All measurement files in `measurements/aetina/` (4 files) and `measurements/metis_card/` (1 file)
- `variance_profile.json` per machine
- Per-machine `characterization/{aetina,metis_card}/README.md`
- Per-machine final report under `docs/phase0-{aetina,metis_card}-findings.md`
- A combined `docs/phase0-L1-L6-mapping.md` documenting which file feeds which validation row (so the Sim Implementation agent can find what it needs)

After both Phase 0 reports commit, **Simulation Implementation kicks off** on the Ubuntu+Metis Card machine (see next section).

---

## Simulation Implementation

**Prerequisite: Phase 0 measurements committed to the repo.** This section is the operational handoff for building the simulator on top of Phase 0 ground truth. Covers agent workflow, repo structure, two-machine division, SSH access, skill set, and program.md/validation-contract templates. **Designed to be offloaded to a server-side agent** with minimal human intervention after initial setup.

### Approach — Karpathy autoresearch pattern, minimal infrastructure

**1 LLM agent, 0 external orchestrator, Python validator script.** Adopted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (83k stars, MIT). Confirmed mechanics: agent itself drives the loop — no separate orchestrator. User launches Claude Code with permissions disabled, prompts agent to read `program.md`, agent iterates autonomously.

Agent loop:
```
while not all_modules_passed:
    read program.md, HANDOFF.md, log.jsonl, validation/contracts/, measurements/
    pick next module M (per dependency graph + log state)
    modify simulator/modules/m{M}.py
    invoke: python simulator/runner.py --module M
    read validation_result.json
    if passed:
        log; trigger regression on Mi<M; if regression ok, advance
    elif retryable failure:
        log; diagnose internally; modify; retry
    elif stuck (attempts >= MAX):
        flag for human review; skip; advance
    update HANDOFF.md (state for next agent session)
```

No background process orchestrates this. The agent is the orchestrator.

### Simulator architecture (6 boxes + data flow)

Layered, modular, event-driven. Each box has a measurement source (or upstream box) and a single validation contract. Implementation correspondence to the M1–M7 modules listed in `### Modules` below.

```
┌──────────────────────────────────────────────────────────────────┐
│ ① Workload generator                            (M5)              │
│   HuggingFace model → torch.onnx.export → per-token op DAG       │
│   • Llama-3 / Qwen-2.5 families, 1B–8B (13B stretch)             │
│   • Prefill + Decode, batch=1                                     │
│   • Trace format aligned with ONNXim input                        │
└────────────────────────────┬─────────────────────────────────────┘
                             │  op stream + tensor metadata
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ ② Scheduler / Mapper        (M6)   ── CONTRIBUTION LAYER ──       │
│   Per-op decision: unit (CIM / NPU / GPU / CPU) + precision +     │
│   memory placement + dataflow + pipeline + precision-boundary     │
│   insertion + resource constraint check                           │
│   Plugin interface: baseline strategy ↔ proposed strategy         │
└────────────────────────────┬─────────────────────────────────────┘
                             │  (op, target_unit, precision) tuples
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ ③ Per-unit timing models      (M1 + M4)   event-driven, parallel │
│   ┌──────────────┐ ┌──────────────────┐ ┌──────────────┐ ┌──────┐│
│   │ CIM (Metis)  │ │ NPU (RKNPU2)     │ │ GPU (Mali)   │ │ CPU  ││
│   │ trace-driven │ │ ONNXim fork +    │ │ trace-driven │ │ ARM  ││
│   │ lookup from  │ │ lookup override  │ │ lookup from  │ │ A76  ││
│   │ Metis Alpha  │ │ for divergent    │ │ Mali OpenCL  │ │ inst-││
│   │ measurements │ │ shapes           │ │ measurements │ │ count││
│   └──────────────┘ └──────────────────┘ └──────────────┘ └──────┘│
│   (`gpu_backend` plugin slot reserved for Accel-Sim integration) │
└────────────────────────────┬─────────────────────────────────────┘
                             │  memory access pattern + latency
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ ④ Memory hierarchy            (M2 + M3)                          │
│   Metis L1 SPM (4 MB × core) ─ Metis L2 SRAM (32 MB shared)      │
│                                       │                          │
│   ┌───────────────────────────────────┴──────────────────────┐   │
│   │ Host LPDDR5 — Ramulator2 backend                         │   │
│   │ PCIe Gen3 ×4 DMA model: BW + latency + setup overhead    │   │
│   │ TLB-miss penalty (parameterized; default 0)              │   │
│   └──────────────────────────────────────────────────────────┘   │
│   On-SoC LPDDR shared by RKNPU2 / Mali / CPU                     │
└────────────────────────────┬─────────────────────────────────────┘
                             │  per-op time + memory access counts
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ ⑤ Energy estimation layer    (M7)                                │
│   Metis CIM:  vendor 15 TOPS/W × utilization                     │
│   CPU A76:    ARM datasheet × activity factor                    │
│   RKNPU2/Mali: INA-delta measurement OR tech-node-derived         │
│   Memory:     per-access energy from JEDEC                       │
│   PCIe:       per-byte energy from PCIe Gen3 spec                │
└────────────────────────────┬─────────────────────────────────────┘
                             │  per-inference latency + energy
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ ⑥ Output + Inline Validation                                     │
│   End-to-end latency / throughput / energy-per-inference         │
│   Per-op timeline + roofline plot                                │
│   Inline comparator: predicted vs measured per box → writes      │
│   `validation_result.json`                                       │
└──────────────────────────────────────────────────────────────────┘
```

**Layering rationale**: each box has one upstream input + one validation contract + one measurement source (or composition of upstream measurements). Modular = the Sim Implementation agent can iterate one box at a time and validate independently before integration.

**Implementation language**: Python (event loop self-written; Ramulator2 wrapped via Python bindings; ONNXim fork integrated as subprocess).

### Repo structure — single private GitHub repo

Repo slug: `cim-llm-mobile-soc-simulator` (private until paper acceptance). Cloned on both machines, synced via git push/pull.

```
cim-llm-mobile-soc-simulator/
├── README.md
├── program.md                  # agent's primary instructions (see template below)
├── HANDOFF.md                  # current state for cross-session continuity
├── log.jsonl                   # agent's per-iteration log (append-only)
│
├── simulator/
│   ├── modules/                # M1-M7 — agent edits these
│   │   ├── m1_cim_tile.py
│   │   ├── m2_memory.py
│   │   ├── m3_event_engine.py
│   │   ├── m4_unit_traces.py
│   │   ├── m5_workload_gen.py
│   │   ├── m6_scheduler.py
│   │   └── m7_energy.py
│   ├── runner.py               # entry: python runner.py --module M
│   ├── validator.py            # invoked by runner; compares output to measurements
│   └── lib/                    # shared helpers (data IO, plotting, common types)
│
├── measurements/                # ground truth, version-controlled
│   ├── aetina/                  # captured on Aetina board (via SSH characterization)
│   │   ├── metis_alpha_cnn_proxy.json
│   │   ├── metis_alpha_matmul.json
│   │   ├── metis_alpha_pcie.json
│   │   ├── rknpu2_matmul.json
│   │   ├── mali_matmul.json
│   │   └── cpu_ops.json
│   └── metis_card/              # captured on Ubuntu+Metis Card box
│       └── vendor_llm_int8.json
│
├── characterization/             # scripts to (re-)capture measurements
│   ├── aetina/                   # runs on Aetina; invoked from Ubuntu via SSH
│   │   ├── README.md
│   │   ├── run_metis_cnn_proxy.sh
│   │   ├── run_metis_matmul.py
│   │   ├── run_pcie_dma.c
│   │   ├── run_rknpu2_matmul.py
│   │   ├── run_mali_matmul/
│   │   └── run_cpu_ops/
│   └── metis_card/               # runs on Ubuntu+Metis Card
│       ├── README.md
│       └── run_vendor_llm.sh
│
├── validation/
│   └── contracts/                # per-module validation spec YAML
│       ├── m1.yaml
│       ├── m2.yaml
│       └── ...
│
├── tools/
│   ├── analysis/
│   ├── plotting/                 # roofline + figure generation
│   └── trace_export/             # HuggingFace LLM → ONNXim-aligned trace
│
├── tests/                        # unit tests (tdd skill writes here)
│
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── measurement_request.md
│   │   ├── module_implementation.md
│   │   └── validation_failure.md
│   └── workflows/                # CI for regression (optional)
│
└── docs/                         # human-facing (NOT for agent loop)
    ├── architecture.md
    ├── measurement-protocol.md
    └── validation-strategy.md
```

### Two-machine division — three agent handoffs

| Phase | Machine | Agent role |
|---|---|---|
| **Phase 0 — Aetina agent** | Aetina RKC-A02 | Runs locally on Aetina: A/C/D/E measurements (per Phase 0 § Machine 1). Commits `measurements/aetina/*` + `docs/phase0-aetina-findings.md`. |
| **Phase 0 — Metis Card agent** | Ubuntu + Metis Card | Runs locally: B measurement (per Phase 0 § Machine 2). Commits `measurements/metis_card/*` + `docs/phase0-metis-card-findings.md`. |
| **Sim Implementation — main agent** | Ubuntu + Metis Card | Karpathy autoresearch loop: M1–M7 simulator dev. Pulls Phase 0 measurements from repo; iterates against `validation/contracts/*`. May SSH back to Aetina (see below) only for additional re-captures during dev. |

Three independent handoffs; each agent gets a self-contained brief. **Phase 0 two agents can run in parallel** (different machines, independent measurement domains, no real-time coupling). **Sim Implementation agent starts only after both Phase 0 reports commit.**

**Sync mechanism**: git push/pull on the shared `cim-llm-mobile-soc-simulator` repo. No real-time coordination needed.

### SSH access — Sim Implementation agent → Aetina (re-capture path only)

Phase 0 agents run locally on their own machine; no SSH between them. During Sim Implementation, if the main agent on Ubuntu+Metis Card needs a shape/precision/config not in `measurements/aetina/`, it triggers a remote re-capture via SSH.

One-time human setup before the Sim Implementation agent runs:

```bash
# On Ubuntu+Metis Card box, as the user running the agent:
ssh-keygen -t ed25519 -f ~/.ssh/aetina_agent -N ""
ssh-copy-id -i ~/.ssh/aetina_agent.pub aetina@<aetina-ip-or-hostname>

# Add to ~/.ssh/config:
cat >> ~/.ssh/config <<EOF
Host aetina
    HostName <aetina-ip-or-hostname>
    User aetina
    IdentityFile ~/.ssh/aetina_agent
    ServerAliveInterval 60
EOF

# Test connectivity:
ssh aetina 'uname -a && ls /home/aetina/'
```

Once SSH key is in place, agent can:
- `ssh aetina 'cd ~/repo && git pull'` — sync repo on Aetina
- `ssh aetina 'cd ~/repo/characterization/aetina && ./run_metis_matmul.py --config tier1'` — trigger remote characterization
- `rsync aetina:~/repo/measurements/aetina/ ./measurements/aetina/` — pull results
- `git add measurements/aetina/ && git commit -m "characterization: new matmul shapes" && git push` — version

All code dev happens on Ubuntu; Aetina is purely a measurement workhorse driven remotely.

### Skills to install (record)

**Install once, before agent first run**:

```bash
# Matt Pocock's skill collection — install all 14
npx skills@latest add mattpocock/skills
```

**Skills we use (subset of mattpocock + local)**:

| Skill | Purpose in our workflow | Source |
|---|---|---|
| `tdd` | Red-green-refactor when implementing modules | local (already installed) + matt's |
| `diagnose` | Structured root-cause analysis when validation fails | local + matt's |
| `to-issues` | Break module work into GitHub issues | local + matt's |
| **`handoff`** | **Compress conversation into HANDOFF.md across sessions — critical for overnight continuity** | matt's only ⭐ |
| **`prototype`** | Throwaway prototypes when trying simulator approach variants | matt's only ⭐ |
| `caveman` | Compressed token mode for long agentic runs (~75 % saving) | matt's only |
| `grill-with-docs` | Pre-implementation alignment before each major module | matt's |
| `improve-codebase-architecture` | Mid/late refactor pass | matt's |
| `write-a-skill` | If we need a custom meta-skill later | matt's |

Skill discovery / version updates: `npx skills check` and `npx skills update` periodically.

### program.md template (agent's primary instruction file)

To live at repo root. Drives agent autonomy. Sketch:

```markdown
# Project: CIM-Centric LLM Inference Simulator

## Your goal
Implement and validate a simulator for LLM inference on a CIM-enabled heterogeneous
mobile SoC. Iterate one module at a time until all pass validation against
measurements/ ground truth.

## How to work
1. Read HANDOFF.md for current state. If empty/unclear, read log.jsonl tail.
2. Read validation/contracts/m{M}.yaml for the module you're working on.
3. Read relevant measurements/*.json as ground truth.
4. Edit simulator/modules/m{M}.py.
5. Run: `python simulator/runner.py --module m{M}`.
6. Read simulator/validator_output.json.
7. If passed: trigger regression with `python simulator/runner.py --regression --up-to m{M}`.
   If regression passed, advance to next module (per dependency graph below).
   If regression failed, fix the regressing earlier module first.
8. If not passed:
   - Append failure analysis to log.jsonl.
   - Form a hypothesis about what to change.
   - Modify and retry.
   - Max attempts per module per session: 20 (then update HANDOFF.md noting blocker, exit).

## Module dependency graph
M1 (CIM tile) ← measurements/aetina/metis_alpha_*
M2 (memory)   ← measurements/aetina/metis_alpha_pcie + Ramulator2 LPDDR5
M3 (event engine) ← M1 + M2
M4 (NPU/GPU/CPU traces) ← measurements/aetina/rknpu2_matmul + mali_matmul + cpu_ops
M5 (LLM workload generator) ← HuggingFace + torch.onnx.export
M6 (scheduler/mapper) ← M3 + M4 + M5
M7 (energy estimation) ← M1..M6

## When to re-capture measurements (ssh into aetina)
If the simulator needs a shape/precision/config not present in measurements/aetina/,
run the appropriate characterization script via:
    ssh aetina 'cd ~/repo/characterization/aetina && ./run_X.py --params Y'
    rsync aetina:~/repo/measurements/aetina/ ./measurements/aetina/
    git add measurements/aetina/ && git commit -m "char: ..." && git push

## When stuck — use these skills
- diagnose: when validation fails repeatedly
- prototype: when unsure about a design approach
- grill-with-docs: before starting a complex module

## End-of-session protocol
Always update HANDOFF.md with:
- Module currently working on
- Last validation status
- Open blockers
- Next steps
Use `handoff` skill to format this concisely.
```

### Validation contract template (YAML per module)

```yaml
# validation/contracts/m{M}.yaml
module: m1_cim_tile
measurement_sources:
  - measurements/aetina/metis_alpha_cnn_proxy.json
  - measurements/aetina/metis_alpha_matmul.json
acceptance_criteria:
  - type: median_op_error
    threshold: 10%      # guidance, not hard commitment
  - type: roofline_shape_match
    metric: knee_position_drift
    threshold: 15%
  - type: sanity
    rules:
      - no_nan_or_inf
      - monotonic_with_op_size
      - latency_positive
sample_strategy:
  cold_starts: 3       # may be lowered after Stage 0 variance characterization
  iterations_per_run: 300
  budget_seconds: 30
```

### Modules (no time allocation per grill-me decision)

| M   | Module                 | Primary measurement source                                                                        | Notes                                                                                                            |
| --- | ---------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| M1  | CIM tile timing        | Metis Alpha CNN + matmul micro-benchmark                                                          | Trace-driven lookup primary; NeuroSim optional cross-check                                                       |
| M2  | Memory hierarchy       | Ramulator2 LPDDR5 + Metis Alpha PCIe DMA                                                          | TLB-miss penalty parameterized (default 0)                                                                       |
| M3  | Event-driven engine    | (depends on M1+M2)                                                                                | Python event loop; orchestrates op stream through units + memory                                                 |
| M4  | NPU / GPU / CPU traces | RKNPU2 matmul, Mali matmul (OpenCL), CPU on RK3588 A76                                            | NPU = ONNXim fork primary + lookup override; GPU/CPU = lookup only                                               |
| M5  | LLM workload generator | HuggingFace Llama-3 / Qwen-2.5 → torch.onnx.export → per-token op DAG                             | Trace format aligned with ONNXim input                                                                           |
| M6  | Scheduler / Mapper     | (depends on M3+M4+M5)                                                                             | Plugin: op→unit + memory mgmt + dataflow + pipeline + precision-boundary insertion. **Contribution lives here.** |
| M7  | Energy estimation      | Vendor specs (Metis 15 TOPS/W), ARM datasheet (CPU), INA-delta or tech-node-derived (RKNPU2/Mali) | Spec-based + activity-factor estimation                                                                          |

## Open risks

1. **NeuroSim integration overhead exceeds estimate** — M1 retreat: pure trace-driven lookup of Metis Alpha measurements (path "use real Metis numbers directly"). NeuroSim then drops from required to optional; methodology citation (NeuroSim Validation paper, <1% chip error) remains valid even without using its code.
2. **Bridging assumption: Metis Card on-card DRAM ≠ simulator's host-MMIO topology** — L4 anchors validate "CIM + on-card-DRAM"; simulator substitutes host-LPDDR + PCIe model. Sensitivity sub-experiment showing simulator behavior under both topologies required.
3. **HPIM publishes at a top venue before us** (arXiv Sep 2025, Beihang) — closest competitor. Differentiators (mobile-SoC, real-silicon calibration, mixed-precision, characterization-driven not pre-committed split) hold even if HPIM lands first.
4. **Agent autonomy at simulator-dev scale untested** — Karpathy autoresearch proves overnight agent loops work for LLM-training (narrow domain, well-defined metric). Simulator development with multi-source measurement validation is more complex; first M1 iteration is the real test of whether the workflow scales. Mitigation: M1 retreat to manual dev if agent fails to converge after N sessions.
5. **HuggingFace ONNX export quality** — `torch.onnx.export(Llama-3 / Qwen-2.5)` is notoriously messy (custom ops, dynamic shape handling). M5 (workload generator) may require manual post-processing or a different extraction tool (e.g., transformers ONNX-export pipeline). Verify before relying on it in M5.
6. **Ramulator2 LPDDR5 + PIM-like extension coverage** — Ramulator2 is modular but our LPDDR5-PIM-like usage is not its default. May need custom plug-ins; M2 budget should include this.
7. **ONNXim RKNPU2 fit** — ONNXim models generic systolic NPU; RKNPU2 has Rockchip-specific behaviors (op-mix sensitivity, depthwise+Swish weakness — see Step-1 data). Plan B: lookup-table override for shapes where ONNXim diverges from RKNPU2 measurements (already in M4 design).
8. **SSH availability of Aetina** — Aetina must be reachable from Ubuntu+Metis Card box throughout simulator dev. If offline, agent flags blocker and proceeds on cached measurements only (limits ability to re-capture).

## Out of scope (with reasons)

Things explicitly excluded from v1 paper scope. Each entry: what's out, why, where it could go in future work.

- **INT4 on Metis CIM** — Voyager public docs don't expose user-controlled INT4. Future work if SDK opens or vendor INT4 artifact appears.
- **AIPU Mode 2 (4-instance) / Mode 3 (compiler-batched)** — both require 4× weight footprint or static-shape batched compile, which don't fit single-batch dynamic-shape LLM on 16 GB unified. Future work for server-like batched scenarios.
- **batch > 1** — see above; mobile single-batch is paper scope. Simulator interface keeps `batch` hook.
- **NVIDIA GPU baseline (Accel-Sim)** — interface-modular extension to M4. Future work for generalization study; `gpu_backend` plugin slot reserved.
- **Jetson Orin / Nano (Accel-Sim DIY config)** — same plugin slot. Future work for edge-GPU generalization.
- **Thermal modeling** — device-dependent, not generalizable; Aetina + Metis Card lack on-board power instrumentation.
- **Energy as measurement** — replaced by spec-based estimation (M7) for the same instrumentation reason.
- **Intra-frame multi-core CIM parallelism** — `cooperative` / `pipeline` modes not implemented in Voyager v1.3.1. Future SDK extensions may enable.

## Relationship to existing vault ideas

- **[[cnn-dnn-edge-memory-wall-metis-embedded]]** — **demoted from High to Low, reframed as calibration source**. Its Step-1 data (225 cells, 5 CNN × 3 unit × batch sweep + 3-mode comparison) is L6 validation anchor; its A1/A2/A3 sub-experiments map directly to Phase 0 A1/A2/A3 here.
- **[[multi-tenant-heterogeneous-edge-soc-contention]]** — sibling on Aetina platform; multi-tenant contention is a sibling research line, not subsumed. May share characterization data.
- **[[llm-test-time-memory]]** — sibling LLM-PIM idea (test-time parametric memory writes on CIM/PIM). HPIM's intra-token pipeline overlap mechanism is directly applicable to test-time memory write scheduling there. Different contribution scope (test-time training vs inference-only).
- **[[long-context-llm-cxl-optimization]]** — sibling LLM-system idea on CXL tier. Different memory regime (CXL not on-SoC); CXL-PNM (HPCA'24) is shared reference.
- **[[moe-upmem-inference]]** — sibling MoE-PIM idea on UPMEM. Different substrate (DRAM-PIM not SRAM-CIM); shares "real-silicon calibration of PIM" methodological discipline.

## Connections

- Closest competitor: [[hpim-arxiv2025]]
- Direct prior art: [[papi-asplos2025]] · [[specpim-asplos2024]] · [[neupims-asplos2024]] · [[ianus-asplos2024]] · [[cent-asplos2025]] · [[lp-spec-arxiv2025]] · [[cxl-pnm-lpddr-hpca2024]] · [[lincoln-hpca2025]] · [[cambricon-llm-micro2024]]
- Methodology template: [[characterizing-mobile-soc-heterogeneous-llm-inference-sosp2025]] (HeteroInfer characterization pattern) · [[neurosim-validation-frontiers2021]] (D4 validation pattern)
- Simulator option assessed (not chosen): [[gem5-salam-merge-2025]]
- Concepts: [[processing-in-memory-llm]] · [[compute-in-memory]] · [[in-memory-computing]] · [[sram-imc]] · [[memory-centric-computing]] · [[on-device-llm-inference]] · [[llm-serving]] · [[llm-weight-quantization]] · [[speculative-decoding]] · [[kv-cache-management]]
- Real-silicon source pages (calibration data): [[metis-step1-cnn-characterization-2026-05-23]] (Step-1 data) · [[metis-llm-investigation-desktop-2026-05-19]] (Metis Card LLM data) · [[metis-exp-board-rkc-a02-2026-05-18]] (Aetina board audit) · [[metis-aipu-nn-v2-2026-05-21]] (direction report)
- Platforms: [[system-aetina-rkc-a02]] · [[system-axelera-metis-card]]
- Calibration source idea: [[cnn-dnn-edge-memory-wall-metis-embedded]]
