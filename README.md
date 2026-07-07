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
| `scheduler.policy` | `all_cim` (silicon-gated) / `cim_hetero` (CIM-INT8 × GPU-FP16, simulated) | |
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
  │                scheduler (M6: AllCim / CimHetero) · precision · resources/events (M3 engine) ·
  │                platform (per-op pricing) · runner · configs/(example JSON)
  ├── models/      M1/M2/M4/M7 unit timing+energy models + engine.py (frozen predict interface) + params/
  ├── specs/       swappable hardware json — memory / cpu / gpu / npu / sram + the cim_topo_* topologies
  └── engines/     external heavy-sim caches (ONNXim / ScaleSim / Ramulator2) behind engine=
characterization/  on-board measurement scripts (aetina/  metis_card/)
measurements/      silicon measurements + op inventory/profile (aetina/  metis_card/  op_inventory/  op_profile/)
traces/            per-token op×shape streams (per model, workload)  ·  fixture/ (the committed DAG source)
validation/        contracts/(m*.yaml acceptance criteria)  reports/(phase*)  validate_*.py
tools/             analysis/ (fits)  plotting/ (one script per figure)  report/  trace_export/  onnxim/  ramulator2/
tests/             pytest (160)
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
