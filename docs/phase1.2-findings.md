# Phase 1.2 — modular "engine + swappable spec" component-layer findings

Phase 1.2 rebuilds every non-micro-benchmark unit as **one model engine + one swappable spec
file** behind a single frozen interface (`Engine(spec, engine='analytic').predict(wl) ->
{latency_us, bound, provenance}`). Swapping a model = swapping a spec; Phase 1.3 drops heavy C++
sims (Ramulator2 / ONNXim) in via `engine=` without touching the API. **All numbers below are
regenerable** from committed JSON via `tools/analysis/{fit,build,check}_*.py` and
`tools/plotting/*.py`; cross-check: `tools/analysis/check_phase1_2.py` (exit 0).

> **Honesty discipline (rigorously applied):** every output is tagged `calibrated` (fit to OUR
> silicon) / `simulated` / `assumption` / `borrowed`. **No fake gate** — where there is no silicon
> (NPU #13, GPU INT8) there is NO per-op numeric acceptance gate, only trend-shape / lower-bound
> acceptance, and the issue-#13 silicon gate is marked *superseded-not-satisfied* (an ADR-0006 gate
> revision), not achieved.

## Component summary

| Unit (ch) | Engine | Spec | Honesty | Acceptance / result |
|---|---|---|---|---|
| **CPU** (C) | instruction-count roofline `max(compute,memory)+overhead_op` | `cpu_rk3588` | **calibrated** (η_c+overhead to fp32 `cpu_ops.json`, 1 A76 core) | per-op residual **median 1.15%, p95 7.31%** |
| **NPU** (N) | analytic systolic-roofline (6 TOPS ceiling + borrowed 32×32 pad + order/shape + attn bmm) | `npu_rknpu2` | **simulator/engines/borrowed** (no silicon, #13) | trend-shape only: staircase knee@32, order/shape ≤6×, BW frac 59–66%/68 — all pass (SIMULATED) |
| **Memory** (M) | analytic all-spec eff-BW + PCIe floor | `mem_lpddr4/4x/5`, `cim_topo_alpha/card` | **mix**: LPDDR4x 24.2=calibrated anchor; LPDDR5 33.3=simulated; peaks=assumption; Alpha 911µs floor=measured | LPDDR5→33.3, LPDDR4x→24.2, alpha floor / card 0; monotone |
| **SRAM** (M) | CACTI tier + residency | `sram_metis_aipu` | **assumption** (CACTI BW/latency); **architecture-only** residency | 8B weights (≫32 MiB) → DRAM tier (never resident) |
| **GPU** (G) | analytic roofline SLOT (coexists with the 1.1 micro-benchmark model) | `gpu_mali_g610` | **simulated** (roofline lower bound, FP16-calibrated; **INT8 zero data**) | error vs 1.1 pts median 3.1% (lower-bound tail); no INT8 gate |
| **CIM-Card** | re-measure the same AIPU on the production card | `cim_topo_card` + `m1_cim.json` | **calibrated** (Alpha 13pts) + **Card-revalidated** | `CARD_REVALIDATED` — 13-pt cross-val median 4.8% / p95 9.7% (PR #25) — see below |

## Audit corrections baked in (the §audit list)

- **CIM**: `alloc_envelope_param_count` (6M) ≠ `native_max_kn` (4.19M); envelope ~14 MB = assumption (vs 1 GiB BAR).
- **CPU**: rope=heads·hd, softmax=heads·(kv+1); single A76 core single-thread = calibration basis (in the spec); A55 IPC=1; fp16 = emulated upper bound; **`eta_bw=0.6` is an ASSUMPTION** — no bandwidth-resolved op in the fp32 decode data + no CPU mem-BW micro-benchmark (the audit gap surfaced, not hidden).
- **Memory**: peaks (34.1/4224) = assumption (in-repo no data-rate source); sim eff 0.65 vs measured 0.71 explained (different memory, discounted not assumed equal).
- **GPU**: INT8 = zero data; 20.12 GFLOP/s = FP16; peak 512 = assumption (may underestimate 2-4×); `ksweep_saturation_M` = dead param, kept + annotated.
- **NPU**: dtypes only INT4/8/16/FP16; BW frac denominator = 68 (HeteroInfer Fig5), not RK3588's 34; no RKNPU2 power → energy not determinable.

## Decision A — Phase 1.1 calibration path preserved

The CPU/MEM constructors changed to `(spec, engine=)`, which would break the three Phase-1.1 callers.
All three were migrated and **re-verified**: `recompose_e2e.py` still gives the **8B decode hold-out
9.5%** (the gate uses op-profile bytes + vendor tok/s + a BW fit — model-independent; only the
*standalone transparency* CPU term shifted, 62→15 ms, which is explicitly "absorbed in BW, not
added"); `fit_m2.py` and `fit_m4_cpu.py` regenerate their frozen 1.1 reports **byte-identical**.

## CIM-Card revalidation — CARD_REVALIDATED (executed in PR #25)

The CIM compute kernel is **not frozen**: the same 800 MHz quad-core AIPU is alive on the production
card, so it can be re-measured and cross-checked vs the Alpha 13 points (no clock rescale). During the
1.2 analytic session this was **deferred** (SSH not yet authorized). It was subsequently **executed**
once board access was authorized (PR #25, `run_metis_cim_v16.py` via `axcompile`): the Card 13-point
cross-validation against the Alpha fit gives **median rel-diff 4.8%, p95 9.7%** (tolerance 10%/20%, no
clock rescale), so `validation/reports/phase1.2/cim_card_revalidate.json` now reports
`status: CARD_REVALIDATED`. CIM is therefore **Alpha-13-pts calibrated + Card-revalidated** (same
AIPU, 800 MHz). _(This section originally recorded `DEFERRED_FALLBACK`; corrected after PR #25 — the
JSON artifact is authoritative.)_

## What's ready for Phase 1.3

The `engine=` slot is live on `m2_memory.py` (→ `'ramulator2'`) and `m4_npu.py` (→ `'onnxim'`), same
constructor + frozen return contract. **Surfaced item:** LPDDR4/4x has no first-class Ramulator2 DRAM
preset, so 1.3 must supply/adapt a timing config to hit the 24.2 calibrated anchor (`m2_memory.json`
notes this; ADR-0002 reconcile lands in 1.3).
