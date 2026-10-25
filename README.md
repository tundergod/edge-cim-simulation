# edge-cim-simulation

A **real-silicon-calibrated simulator** of LLM inference on a **CIM-enabled heterogeneous mobile
SoC** — studying Compute-in-Memory (CIM) as a first-class compute unit alongside NPU, GPU, and CPU
(INT8, batch=1, prefill + decode).

The simulated target is a SRAM-based CIM accelerator whose **memory topology** (on-card DRAM / host
PCIe / integrated-SoC LPDDR5) and **compute geometry** are swappable spec files, calibrated against
**two real Axelera Metis boards**:

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
| Models | Llama-3 + Qwen-2.5, 1B–8B (Llama-3.2-1B / 3B, Llama-3.1-8B, Qwen-2.5-7B; Qwen-2.5-14B as an *extrapolation-only* point) |
| Phase / precision | Prefill + decode, end-to-end; INT8 on CIM |
| Mixed precision | Per-unit native precision (CIM INT8, NPU INT8/16/FP16, Mali FP16). The CIM-MLP(INT8) × GPU-attention(FP16) boundary is a core research question |
| Context / batch | 2K baseline (8K stretch); batch = 1 |

Out of scope: training, MoE/sparse, FP32, multi-batch — see [OVERALL.md](OVERALL.md) § 範圍外.

## Status — the end-to-end simulator is built and validated

- **Phase 0.1–0.3** ✅ trace / op-profile / on-board silicon measurement. **Phase 0.4** 🔶 thermal
  (Metis Card measured; Aetina pending repair).
- **Phase 1.1–1.3** ✅ per-component modelling + validation (CIM, memory, CPU, GPU, NPU), plus
  Ramulator2 / ONNXim / ScaleSim heavy-sim drop-ins behind a frozen `engine=` interface. (1.4–1.6b
  are *reinforcement* folded into their parent sub-phase, not new phases.)
- **Phase 3 (2026-10-23)** 🔶 **heterogeneous tiled scheduling** on SoC profiles (ADR-0008/0009 + revisions): any op tile on
  any of CPU / GPU / NPU / SRAM-CIM, HeteroInfer-anchored P1 profile, single-unit + HeteroInfer baselines; **all SIMULATED**
  (no heterogeneous silicon; P1 is borrowed from published numbers). See `docs/adr/0008-*.md`, `0009-*.md`.
- **Phase 2.1–2.5** ✅ the integrated **per-token event-driven simulator** (`simulator/runtime/`):
  - **2.1** — M5 workload op-DAG + M3 event engine + AllCim scheduler; decode reproduces the L4
    silicon anchor within 15% (1B 10.7% / 3B 6.5% / 8B 3.1%) from independent per-op pricing, and a
    memory-only ablation *fails* the gate (non-circular evidence the CIM-compute term is load-bearing).
  - **2.2a/b** — value-flow op-DAG + memory-domain routing + per-op provenance; a `Scheduler` ABC with
    an `AllCim` and a `CimHetero` (CIM-INT8 matmul × GPU-FP16 attention, precision-boundary conversions).
  - **2.3** — topology A/B/C swap (card / host-PCIe / edge-LPDDR5), ±20% sensitivity, a genuine
    leave-8B-out hold-out, and a 14B size-extrapolation — each labelled measured / simulated /
    counterfactual / extrapolated.
  - **2.4** — gate/test hardening (fail-loud validator exit codes, coverage for fail-loud branches,
    honesty-framing corrections).
  - **2.5** — the CIM **compute engine** is now spec-driven (swap the accelerator geometry per topology
    via `cim_compute_params`); a non-Metis geometry is auto-labelled `simulated` (no silicon).

**Consolidated report:** [`docs/report/phase1-site/`](docs/report/phase1-site/) — a hand-coded
multi-page site where every number is injected from committed JSON (the build fails on any unresolved
placeholder) and every figure is regenerated from data. Known measurement gaps (no RKNPU2 silicon for
the NPU, issue #13; the weight-vs-all-traffic double-count, issue #67) are labelled honestly.

## Quickstart

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.phase0.txt
.venv/bin/pytest tests/                 # 160 unit tests
```

> `HF_TOKEN` is only needed to *regenerate* op inventories/traces from Hugging Face. The committed
> traces (`traces/fixture/*.json.gz`) let you run every simulation and validator without it.

### Run one end-to-end simulation

The simulator is a Python API: build a `SimConfig`, call `run(cfg)`, read the metrics dict. There is
no separate CLI — the config *is* the interface (a JSON file or a dict). Example config:
[`simulator/runtime/configs/l4_repro.json`](simulator/runtime/configs/l4_repro.json).

```python
from simulator.runtime.config import SimConfig
from simulator.runtime.runner import run

cfg = SimConfig.from_json("simulator/runtime/configs/l4_repro.json")   # or SimConfig.from_dict({...})
m = run(cfg)
print(m["tok_s"], m["ttft_s_reported_not_gated"], m["energy_per_token_J"])
print(cfg.is_calibrated_anchor(), cfg.provenance)
```

Running the committed L4 config (Llama-3.1-8B, on-card-DRAM topology) gives these key metrics
(`m[...]` fields, rounded):

```
tok_s 2.7841   TTFT_s 0.4074   energy_J/tok 0.362854   memory_eff_BW_GBs 24.2
is_calibrated_anchor True   provenance []       # a silicon-anchored config: no provenance flags
```

Swap the topology in a dict and the provenance layer flags it honestly:

```python
cfg = SimConfig.from_dict({
    "workload":  {"model": "llama-3.2-1b", "context": 1024},
    "platform":  {"topology": "cim_topo_edge", "memory_spec": "mem_lpddr5"},
    "scheduler": {"policy": "all_cim"}})
run(cfg)["tok_s"]     # 17.0354  (eff_BW 29.97 GB/s)
cfg.provenance        # ["simulated: topology 'cim_topo_edge' ...", "simulated: memory_spec 'mem_lpddr5' ..."]
```

### The knobs

| Knob | Values | Notes |
| --- | --- | --- |
| `workload.model` | `llama-3.2-1b` / `llama-3.2-3b` / `llama-3.1-8b` / `qwen2.5-7b` | `qwen2.5-14b` is extrapolation-only (labelled, never validated) |
| `platform.topology` | `cim_topo_card` (L4 anchor, on-card LPDDR4x) / `cim_topo_alpha` (host PCIe, no on-card DRAM — counterfactual) / `cim_topo_edge` (integrated SoC LPDDR5×NoC — simulated) | single bandwidth source; `memory_spec` is a consistency tag |
| `scheduler.policy` | `all_cim` (silicon-gated) / `cim_hetero` (CIM-INT8 × GPU-FP16, simulated) / **Phase 3 (ADR-0009, all SIMULATED, need a `soc_profile` for P1):** `gpu_only` / `npu_only` / `cim_only` / `cpu_only` (single-engine baselines, syncs charged) / `heteroinfer` (faithful HeteroInfer §4) / `tiled_hetero` (every op tiled across the enabled units) | `all_cim` is the only silicon anchor; `all_cim`/`cim_hetero` charge no sync (`sync_mode: n/a-legacy`) and are not fair baselines for the Phase-3 family — compare against `cim_only` |
| `platform.soc_profile` | unset (legacy Metis unit set) / `sd8gen3_cim` (P1: **borrowed** HeteroInfer numbers + CIM) / `rk3588_cim` (P2: RK3588-measured units + CIM, simulated edge) | fixes the CIM topology and the memory wall (43 GB/s single stream, 60 knee on P1); rejects `bw_efficiency`; never a calibrated anchor |
| `scheduler.host` | `gpu` / `npu` / `cim` / `auto` (tiled_hetero) | graph-owning unit (merges land on it, a non-host single placement pays the hand-back sync); `auto` = the portfolio (per-host solver plans + every single-unit plan, engine-ranked) → `plan_source` says which won (`solver:<host>` / `single:<unit>`) |
| `scheduler.allow_cpu_compute` | false (default = HeteroInfer's unit set, CPU control-plane only) / true ("ours": CPU as a compute unit) | the CPU GEMM kernel model is SIMULATED (no silicon; fitted end-to-end to one published ratio) — the largest lever in matrix v1 rests on it |
| `scheduler.sync_mode`, `sync_us_override`, `partition_grid` | `fast` (10 µs decode / 30 prefill, HeteroInfer §4.3) / `clfinish` (400) / `none`; an explicit µs; ratio grid (default 8) | per-boundary sync is a SERIAL pre-op delay (F1); literature-anchored, no hetero-sync silicon |
| `platform.cim_n_cores` | int ≥ 1 (profile default 4 = the Metis geometry) | scalable axis: tile width + prefill throughput ∝ n (F12); other counts EXTRAPOLATED |
| `platform.npu_scale` | float > 0 (default 1.0) | multiplies NPU compute throughput only; ≠ 1 EXTRAPOLATED |
| `platform.cpu_gemv_efficiency` | (0, 1] (profile value 0.128 after the round-2 fit) | sensitivity knob on the CPU M=1 kernel efficiency (an ASSUMPTION); above the GEMM efficiency lifts both |
| `platform.cim_weight_bytes_per_param` | (0, 4] (profile 1.0 = INT8 crossbar weights as stored; 0.5 = W4 storage unpacked on load) | a DESIGN OPTION with no silicon; bytes only within the resident capacity (beyond it the re-programmed bytes scale with it, ADR-0008 R3); decides the CIM-vs-W4 competitors question (matrix v1 H2) |
| `platform.memory_capacity_GB`, `platform.bw_efficiency`, `tunables.*` | forward-looking sweep knobs | any off-anchor value is auto-tagged in `cfg.provenance` |
| topology spec `cim_compute_params` | a repo-relative params path | the **CIM compute-engine swap surface**; a non-Metis geometry is tagged `simulated` |

Every config that leaves the silicon-calibrated envelope carries a `provenance` list explaining why,
and `is_calibrated_anchor()` returns `False`. Only `cim_topo_card` + AllCim is a calibrated anchor.

### Run the validation gates

The validators are the runnable end-to-end examples; each exits non-zero on failure (so CI-able) and
writes a committed JSON under `validation/reports/`:

```sh
.venv/bin/python validation/validate_e2e_l4.py           # L4 decode gate (the headline ≤15% vs silicon)
.venv/bin/python validation/validate_topology_ab.py      # A/B/C topology swap (card reproduces the committed anchor)
.venv/bin/python validation/validate_sensitivity_l5.py   # ±20% robustness band
.venv/bin/python validation/validate_holdout.py          # genuine leave-8B-out
.venv/bin/python validation/validate_extrapolation_13b.py # 14B size/capacity/CIM-shape extrapolation (labelled)
.venv/bin/python validation/validate_contention.py       # shared-BW contention shape (simulated, #52)
.venv/bin/python validation/validate_m5_trace.py          # M5 trace vs the Phase-0.1 oracle
.venv/bin/python validation/validate_m7_energy.py         # M7 energy sanity + ±20% sensitivity
```

### Build the report / regenerate figures

```sh
.venv/bin/python docs/report/phase1-site/build.py --strict   # fails on any unresolved {{key}} or stale figure
.venv/bin/python tools/plotting/site_m1.py                    # (e.g.) regenerate one figure group from committed data
```

## Repository layout

```
CLAUDE.md  CONTEXT.md  OVERALL.md  README.md  LOG.md  requirements.phase0.txt
docs/              papers/ plans/ adr/ agents/ figures/ report/(phase1-site) + *-findings.md  voyager-sdk.md
simulator/
  ├── runtime/     the Phase-2 per-token simulator: config (SimConfig) · workload/dag (M5 op-DAG) ·
  │                scheduler (M6: AllCim / CimHetero + Phase-3 gpu/npu/cim/cpu_only · heteroinfer · tiled_hetero) ·
  │                tiling · solver (ADR-0009) · precision · resources/events (M3 engine) ·
  │                platform (per-op pricing) · runner · configs/(example JSON)
  ├── models/      M1/M2/M4/M7 unit timing+energy models + engine.py (frozen predict interface) + params/
  ├── specs/       swappable hardware json — memory / cpu / gpu / npu / sram + the cim_topo_* topologies
  └── engines/     external heavy-sim caches (ONNXim / ScaleSim / Ramulator2) behind engine=
characterization/  on-board measurement scripts (aetina/  metis_card/)
measurements/      silicon measurements + op inventory/profile (aetina/  metis_card/  op_inventory/  op_profile/)
traces/            per-token op×shape streams (per model, workload)  ·  fixture/ (the committed DAG source)
validation/        contracts/(m*.yaml acceptance criteria)  reports/(phase*)  validate_*.py
tools/             analysis/ (fits)  plotting/ (one script per figure)  report/  trace_export/  onnxim/  ramulator2/
tests/             pytest (249; Phase-3 hetero tests included)
```

## Honesty discipline

The simulator is calibrated against real silicon for *some* units/topologies and not others, so the
repo is strict about provenance: each value is tagged **measured** / **simulated** / **extrapolated** /
**counterfactual** (see CONTEXT.md § Glossary), and report numbers flow from committed JSON rather than
prose. **Results must never confirm their own assumptions** — no circular reasoning, no manufactured
cross-source agreement, no validation language without ground truth. See CLAUDE.md § *Honesty discipline*.

## Key docs

- **[OVERALL.md](OVERALL.md)** — project brief: goal, phases (0.1→2.5), modules M1–M8, validation
  layers L1–L6, workloads, open risks.
- **[CONTEXT.md](CONTEXT.md)** — domain glossary + a directory-level **repo index** (where everything
  lives; consult it before grepping).
- **[CLAUDE.md](CLAUDE.md)** — working guidelines (per-phase workflow, simplicity, honesty discipline).
- **[docs/adr/](docs/adr/)** — architecture decision records (ADR-0001…0007).
- **[docs/voyager-sdk.md](docs/voyager-sdk.md)** — how to measure Metis (Voyager SDK reference,
  tagged `[DOC]` / `[FORUM]` / `[MEASURED]` / `[GAP]`).
- **[docs/papers/](docs/papers/)** — curated literature + real-silicon notes (16 papers).

## External references

- Voyager SDK — <https://github.com/axelera-ai-hub/voyager-sdk>
- Axelera community (Metis M.2) — <https://community.axelera.ai/metis-m-2-3>
