# ADR-0008 — SoC profiles + heterogeneous unit pricing (`Platform.price` for every unit × op)

Status: Proposed (2026-10-23, branch `hetero-pricing`)

## Context

The 2026-10-23 research plan (cim-mobile-soc/RESEARCH_PLAN_2026-10-23.md) reuses this simulator's
M3 engine / shared-bandwidth knee / op-DAG and adds schedulers that may place ANY op tile on ANY
unit (CPU / GPU / NPU / SRAM-CIM) and compare two SoCs: **P1** a Snapdragon-8-Gen-3-like SoC
calibrated to HeteroInfer's (SOSP'25) published numbers plus an SRAM-CIM unit, and **P2** the
RK3588-measured units of this repo plus the Metis CIM. Through Phase 2.5 `Platform.price` only
knew cim/matmul, cpu/support-ops, gpu/attention-bmm (Mali) and mem; the NPU raised; there was no
GEMM on gpu/cpu, no CIM attention compute, no merge/sync cost, and no way to swap the whole SoC.
The L4 gate (AllCim 1B/3B/8B decode tok/s 14.4727 / 5.9659 / 2.7841, `validate_e2e_l4.py`) must
stay byte-identical.

## Decision

### 1. Pricing contract (`simulator/runtime/platform.py`)

`Platform.price(node) -> {latency_us, compute_provenance, source_model}` is decided ONLY from
`node.unit`, `node.category`, `node.wl` (M, K, N, kv, heads, dtype, nbytes, extra) and
`node.pricing_group` — never node ids; unknown `extra` fields are ignored.
`Platform.supports(unit, category)` is the oracle a solver enumerates (True iff `price` would not
raise); unsupported combos raise `NotImplementedError`, unknown units `ValueError`.

| unit | supported categories | models |
|---|---|---|
| cim | matmul, attention, convert, merge | M1 `CimTileModel` (topology params, `n_cores` from the profile); attention = `CimAttentionModel` (profiles) / 0 (legacy) |
| gpu | matmul, attention, norm, rope, ffn, softmax, residual, convert, merge | P1 `GpuAnchorRooflineModel`; P2/legacy `MaliGpuModel` (attn) + `GpuRooflineModel` (GEMM, shape trend only); elementwise = launch floor |
| npu | matmul, attention, norm, rope, ffn, softmax, residual, convert, merge | P1 `NpuStaircaseModel`; P2/legacy `NpuModel` (RKNPU2 analytic; onnxim/scalesim selectable); elementwise = launch floor |
| cpu | matmul, attention, norm, rope, ffn, softmax, residual, sampling, convert, merge | `CpuModel` (calibrated support ops, fp32 path) + `CpuGemmModel` (GEMM/attention roofline, new) |
| mem | embedding, kv_cache, convert, merge | compute 0, cost = bytes via the M3 pool |

* `merge` (a scheduler's split/merge + sync node): `latency_us = wl.extra["fixed_us"]` (default 0)
  on ANY unit, provenance `fixed-cost merge/sync (set by scheduler)`; bytes metered as usual.
* `wl.extra["fixed_us"]` is an ADDITIVE term for every other category (per-op launch/sync overhead
  a scheduler attaches); when absent the result is bit-identical to before.
* Attention composite (R2): the QK^T + S·V pair is priced ONCE on the rep (`pricing_group == id`),
  members return 0, an ungrouped bmm raises — on gpu, npu, cim and cpu alike. The composite is
  priced from the node's own `heads` / `kv` / `M` (a head- or kv-split sub-node prices its own
  shape). `wl.M` is the query-row count (1 = decode; a prefill scheduler may set M = Sq — the
  current `wl_from_row` leaves attention M = 1, an existing TTFT limitation).
* Elementwise ops on gpu/npu return only the per-kernel launch floor as COMPUTE: the data movement
  is the node's `bytes_streamed`, metered by the engine (`max(compute, memory)`), so pricing bytes
  again as compute would double count. On the CPU the calibrated m4_cpu term already includes the
  cache-resident memory term (R7, `cpu_cache` domain).

### 2. SoC profiles (`SimConfig.soc_profile`, `simulator/specs/soc_*.json`)

A profile declares per-unit `{spec, model, engine, default_dtype, elementwise_launch_us, gemm}`,
`cim {topology, n_cores, attention_model}`, `memory {single_stream_eff_BW_GBs, knee_GBs}` (mapped
onto `SharedBandwidth(eff_BW, knee)` — `resources.py` unchanged), `sync_us {fast, clfinish}`,
honesty and provenance. **Precedence:** the profile fixes the CIM topology (config `topology`
omitted → resolved from the profile; a conflicting value is rejected); the profile's memory block
overrides the topology-resolved eff_BW; `tunables.knee_GBs` (sensitivity knob) still overrides the
profile knee; `bw_efficiency` is rejected with a profile. `SimConfig` validates the name at
construction and at `run()`, and appends the profile's honesty line to `provenance` (never a
calibrated anchor). The runner must pass `soc_profile=cfg.soc_profile` to `Platform(...)` (one-line
change in `runner.py`, owned by the scheduler branch — not applied here).

| | P1 `sd8gen3_cim` | P2 `rk3588_cim` |
|---|---|---|
| GPU | `gpu_adreno750_sd8gen3`: 2-regime roofline fitted to HeteroInfer Tab.3 (**borrowed**) | `gpu_mali_g610`: measured attn_bmm fit + roofline_fit GEMM = **known-unoptimised kernel, shape trend only** |
| NPU | `npu_hexagon_sd8gen3`: 32-row staircase fitted to Tab.3, Fig.3 cross-check, Fig.4 order flag (**borrowed**) | `npu_rknpu2` analytic systolic (**simulated**, no silicon #13); `engine: scalesim` selectable but compute-only table, 8 probe shapes → analytic default |
| CPU | `cpu_rk3588` A76 fits as a **labelled proxy** + GEMM roofline (NEON peak × 0.5 efficiency, **assumption**) | same (measured on RK3588) |
| CIM | M1 Metis tile via `cim_topo_edge` (shared LPDDR, no floor), `n_cores` = scalable axis; attention = Alpha attn_floor fit | same |
| Memory | 43 GB/s single-stream, knee 60 (61.9 practical, 68 peak) — HeteroInfer §3.3/§5.4 (**borrowed**) | 29.97 = mem_lpddr5 33.3 × NoC 0.9 (= the edge topology, test-pinned), knee 33.03 = × the Card 4c/1c 1.102 (**simulated**) |
| sync_us | fast 30 (§4.3 "few us polling + tens of us submission"), clFinish 400 (§3.1) | same presets (borrowed, not RK3588-measured) |

**Legacy (soc_profile unset):** the Phase-2.5 unit set is built exactly as before (CimTileModel from
the topology params, `CpuModel(cpu_rk3588)`, `MaliGpuModel()`, CIM attention = 0) so every
pre-existing combo is byte-identical; only the NEW combos (gpu/npu GEMM + elementwise, cpu GEMM /
sampling, any npu) take the `rk3588_cim` unit parameters, so B's schedulers can run without a profile.

### 3. P1 fits (`tools/analysis/fit_hetero_anchors.py` → `anchor_fit` blocks; `validation/reports/hetero/anchor_fit.json`)

Minimax relative error over the six Table 3 anchors per unit (acceptance band ±15 % on every anchor).
Shapes: weight [N_out, K_in], activation [K_in, M]; GPU weight bytes = 0.5 B/param (W4A16).

**GPU** `t = floor + 0.5·K·N/bw` (M = 1); `t = floor + 2MKN / (C_max · M/(M+M_half))` (M > 1; bridged by
`min(M·GEMV, GEMM)` for 1 < M ≲ 7, an assumption — no anchors between M = 1 and 128).
Fitted: floor = 320.0 µs, bw = 31.52 GB/s, C_max = 1.130 TFLOPS, M_half = 104.1.

| [N_out, K_in] | M | anchor µs | pred µs | resid |
|---|---|---|---|---|
| 4096, 4096 | 1 | 511 | 586.1 | +14.7 % |
| 4096, 4096 | 128 | 7306 | 7210.9 | −1.3 % |
| 4096, 4096 | 256 | 10841 | 11010.8 | +1.6 % |
| 4096, 14336 | 1 | 1467 | 1251.4 | −14.7 % |
| 4096, 14336 | 256 | 35231 | 37737.6 | +7.1 % |
| 28672, 4096 | 1 | 1903 | 2182.7 | +14.7 % |

All within ±15 %, but the three decode anchors sit AT the band: their implied effective BW rises with
N_out (16 → 20 → 31 GB/s), which a constant-BW roofline cannot express (a single-regime roofline
`floor + max(flops/C, bytes/bw)` reaches 16.1 % and was rejected). The fitted 320 µs floor is a
GEMV-kernel cost (dequant + launch), NOT a per-launch floor — attention/elementwise kernels use
`kernel_launch_us = 20` (§4.3 "tens of µs", assumption), otherwise the paper's 14 tok/s Llama-8B
decode would be impossible.

**NPU** `t = floor + (K/4096)^p · [K·N/bw + ceil(M/32)·2·32·K·N/TOPS_eff] · order`.
Fitted: floor = 58.6 µs, bw = 42.76 GB/s (≈ the paper's per-unit 40–45 GB/s), TOPS_eff = 7.32
(paper: "up to 10 TFLOPS in actual"), p = 1.064 (NPU-3 contraction-dim penalty); order = 6 (Fig.4)
when `wl.extra["order"] == "activation_stationary"`.

| [N_out, K_in] | M | anchor µs | pred µs | resid |
|---|---|---|---|---|
| 4096, 4096 | 1 | 693 | 597.6 | −13.8 % |
| 4096, 4096 | 128 | 912 | 1037.7 | +13.8 % |
| 4096, 4096 | 256 | 1884 | 1624.3 | −13.8 % |
| 4096, 14336 | 1 | 6506 | 7211.8 | +10.8 % |
| 4096, 14336 | 256 | 23445 | 20836.9 | −11.1 % |
| 28672, 4096 | 1 | 3886 | 3831.7 | −1.4 % |

The [4096,4096] M = 1/128/256 triple (693/912/1884) is mutually inconsistent with any linear-in-steps
law (the M = 128 → 256 slope is 3× the M = 1 → 128 slope); minimax spreads it to ±13.8 %. Fig.3
cross-check (NOT fitted), [4096,14336] levels at steps 1..6: +4.5 / +5.3 / +1.9 / −6.8 / −6.3 / −5.9 %.

**CIM attention** (`params/m1_cim_attention.json`, measured Alpha `attn_floor`, single head, hd = 128,
M = 1): QK^T = 8.952 + 0.001706·kv µs, S·V = 8.999 + 0.000809·kv µs per bmm; heads serial
(assumption). Extrapolated for hd ≠ 128, kv > 1025, M > 1 (`max(floor fit, M1 prefill law)`).

## Consequences / honesty tiers

* P1 is **borrowed** end to end (published numbers, nothing measured by us); its attention,
  elementwise and bridging regimes are **extrapolated** from the GEMM anchors. P2's GPU GEMM is a
  measured but unoptimised kernel (shape trend); its NPU is **simulated** (no silicon); CPU GEMM
  efficiency 0.5 is an **assumption**; the knees are **simulated**.
* The L4 path is untouched (tests `test_soc_profile.py::test_legacy_pricing_identical_to_main_snapshot`
  and `::test_l4_tok_s_unchanged_without_profile`; `validate_e2e_l4.py` unchanged).
* Enabling the CIM attention floor through a profile changes an AllCim-style placement's attention
  cost from 0 to ~19 µs/head/layer (Alpha standalone-call basis); the Card's compiled graph passes L4
  with 0, so the floor likely OVER-prices in-graph attention — a profile parameter
  (`attention_model`), not a calibrated fact.

## Known gaps

1. No GPU/NPU energy (M7 has cim/dram/cpu only); `energy_J` ignores gpu/npu compute.
2. P1 GPU elementwise/attention floors and P2 launch floors are assumptions (no silicon).
3. NPU `scalesim` table is compute-only and sparse → seam vs the analytic memory roofline.
4. Prefill attention M = 1 in the current DAG (`wl_from_row`); CPU GEMM has no silicon; the CIM
   attention basis is Alpha standalone calls.
5. `runner.py` does not yet pass `soc_profile` to `Platform` (scheduler branch to wire).
6. Elementwise launch floors are charged PER ATEN NODE (the op_profile trace granularity: a
   decode layer has ~30 norm/rope/residual/softmax/swiglu nodes), not per fused kernel; the
   calibrated CPU path has the same granularity (its per-node overhead_op is what L4 sees). A
   scheduler that models fused kernels should attach the cost once via `fixed_us` / merge nodes.
7. `supports("gpu", "attention")` is True, but on the Mali (P2/legacy) path the NON-bmm attention
   subtype (scale/mask, no `hd` in `wl.extra`) still raises `NotImplementedError` (the pre-existing
   fail-loud guard, test-pinned); P1 prices it as an elementwise floor. Solvers should route
   scale/mask to the CPU on P2 or check the subtype.

## Addendum (2026-10-23): scalable axes `platform.cim_n_cores` / `platform.npu_scale`

* `platform.cim_n_cores` (int ≥ 1, default unset = the profile's `units.cim.n_cores`, legacy path = the
  M1 params default 4) overrides the CIM core count → `CimTileModel(n_cores=)`. Any value other than
  the profile's is provenance-flagged `extrapolated`. What it changes in the M1 model: the pass width
  `W = n_cores × 512` of the M=1 single-tile fit (a shape with both dims ≤ W is priced by the G_eff
  tiling fit, otherwise by the multi-tile residency-cliff model, which does NOT depend on `n_cores`),
  and the prefill tile area `K·N / W²` (so a prefill GEMM scales as 1/n_cores² — an optimistic
  extrapolation of the 2048×2048 tile fit).
* `platform.npu_scale` (float > 0, default 1.0) multiplies the NPU COMPUTE throughput only
  (`NpuModel.scale_compute`: `tops` for the RKNPU2 analytic model, `tops_eff` for the sd8gen3
  staircase); the weight-streaming bandwidth and dispatch floor are untouched, so a memory-bound
  decode GEMV barely moves while a prefill GEMM scales. Flagged `extrapolated` when ≠ 1.
* `runner.run` echoes the values actually used: `host`, `prefill_host`, `host_candidates_est_us`,
  `cim_n_cores`, `npu_scale`, `knee_GBs` (profile or `tunables.knee_GBs`), `soc_profile`.

## Revisions 2026-10-23 (independent review, fixes F1–F15; branch `hetero-fixes`)

Every change below is gated behind a SoC profile or the ADR-0009 schedulers; the legacy path
(`soc_profile` unset, AllCim / CimHetero) is byte-identical (`validate_e2e_l4.py` output unchanged,
`test_legacy_pricing_identical_to_main_snapshot`, `test_l4_tok_s_unchanged_without_profile`).
Regression tests: `tests/test_hetero_fixes.py` (one per fix), `tests/test_hetero_schedulers.py` (F9).

* **F1 — `price()` returns `fixed_us` as its own field.** `{latency_us: compute only, fixed_us, ...}`;
  a `merge` node is `latency_us 0, fixed_us = sync`. The engine (`run_dag` AND `run_serial`) and the
  solver charge it as a SERIAL pre-op delay: node = fixed + max(compute, memory) (unit busy, nothing
  streams during the delay). Before, the engine folded it into compute and hid it under max() while
  the solver added it outside (heteroinfer 1B decode: 2496 of 3050 charged µs visible); now charged ==
  visible. `Platform.compute_us` stays compute-only; `Platform.fixed_us(node)` added.
* **F2 — one memory model.** Every unit model is now a KERNEL-TIME model; the DRAM bytes of every unit
  are metered by the shared pool alone, with a per-unit single-stream cap `units.<u>.stream_eff` ×
  eff_BW (P1: 1.0 everywhere = HeteroInfer Fig.5 "each unit alone ~40–45"; P2 npu 0.67 = the old
  NpuModel 20 GB/s band / 29.97) and the knee across units (F10). Removed: the GPU anchor model's
  `0.5·K·N/bw` term and attention `bytes/bw` term, `NpuModel._gemm_us` memory ceiling and the staircase
  attention memory term, `CpuGemmModel` memory term. KEPT, relabelled: the NPU staircase `K·N/bw_GBs`
  term is the array weight-load pass (kernel side; the three [·,·] M=1/128/256 anchors need a
  steps-independent K·N term), so an NPU GEMV part never runs faster than its own array fill — the
  paper's "NPU-1 stage performance → GPU primary in decode". The P2 Mali `GpuRooflineModel` keeps its
  1.26 GB/s "BW" (a kernel-rate fit of the unoptimised kernel, not a DRAM term). The fit script now
  predicts an anchor as `max(kernel, K·N·wbpp/43 GB/s)`; every Table-3 anchor is kernel-bound under
  that rule, so the NPU parameters are unchanged (floor 58.6, 42.76, 7.32 TOPS, p 1.064).
* **F3 — GPU GEMV refit.** New form `gemv_floor + K·(N + n_half)/rate` (occupancy: N outputs = N
  work-items), GEMM regime keeps its own floor: gemv_floor 60.0 µs (bounded to the §4.3 "tens of µs"
  band), rate 73.4 Gparam/s, n_half 3505, gemm_floor 402 µs, C_max 1.174 TFLOPS, M_half 110.6. Max
  anchor residual 5.3 % (was 14.7 % at the band edge): 511→484, 7306→7225, 10841→10885, 1467→1544,
  35231→37092, 1903→1855. Small-shape plausibility: 2048×512 M=1 = 172 µs (was 337). Provenance flags
  `N_out < 4096` as EXTRAPOLATED (the occupancy term is anchored only at 4096 and 28672). Table 3
  through the REAL platform + solver (profile `units.npu.min_split_rows = 1024`, §4.4 constraint (2),
  value unstated by the paper): [4096,4096] M=1 → 1:1; [28672,4096] → GPU 0.75 (paper 3:1);
  [4096,14336] M=1 → GPU-only; [4096,4096] M=128 → NPU-only; [4096,14336] M=256 → NPU 0.625 (paper
  0.6). E2E sanity (not fitted): heteroinfer Llama-3.1-8B decode 6.67 tok/s vs the paper's 14.01 — the
  simulator still streams INT8 NPU/CIM weights and keeps the int8↔fp16 converts (ADR-0004); the GPU
  streams W4 (F13). On the 1B shapes (N ≤ 2048, below every anchor) the solver stays NPU-leaning
  (q/o proj npu 0.75 / gpu 0.25) — an extrapolation of the occupancy term, flagged.
* **F6 — `cim.attention_heads_mode ∈ {serial, batched}`** (`CimAttentionModel(heads_mode=)`): serial
  = (floor + b·kv) × heads (measured Alpha basis, default); batched = floor once per bmm + work ×
  heads (a compiled multi-head graph; no silicon). A sensitivity axis, not a calibrated fact.
* **F12 — physical `cim_n_cores` on the profile path** (`CimTileModel(scale_mode="physical")`, s =
  n/4): single-tile decode throughput s·G_eff(N/s, K), prefill tile throughput ∝ s (latency 1/n, was
  1/n²), multi-tile knee × s (resident capacity), resident slope / s, DRAM-spill floor kept as the
  array's measured ingest rate (n-independent kernel time; the pool meters the bytes). s = 1 ==
  legacy numerics; the legacy path keeps `scale_mode="legacy"`. **Conflicts by design with the knob
  addendum above** (`test_knobs.py::test_cim_n_cores_override_changes_tile_width_and_latency` now
  asserts 1/n and t8 < t4).
* **F13 — `units.<u>.weight_bytes_per_param`** (P1 gpu 0.5 W4A16, npu 1.0, cim 1.0, cpu 0.5; P2 all
  1.0; legacy 1.0 = op_profile INT8 bytes): a scheduler's `_finish` rescales every matmul node's weight
  bytes to its unit from the remembered INT8 basis `wl.extra["bytes_int8"]` (idempotent; tiling
  derives sub-node bytes from the basis), and the solver's `_dram_bytes` applies the same rule. The
  int8↔fp16 converts remain (ADR-0004). 8B heteroinfer decode 5.71 → 6.67 tok/s.
* **F14 — Mali non-bmm attention** (scale/mask) is priced as the elementwise launch floor on both GPU
  models (was `NotImplementedError`, which failed every rk3588_cim run with a GPU in the unit set).
  CimHetero routes scale/mask to the CPU, so the L4-era numerics never reached that branch (snapshot
  test unchanged).
* **CPU GEMM** (`CpuGemmModel`): kernel time only, `eff(M) = gemv_eff + (eff − gemv_eff)(M−1)/(M−1+m_half)`
  with P1 `gemv_efficiency 0.217` (a decode GEMV = 1.33 × its W4 pool time = HeteroInfer Fig.12
  llama.cpp/MNN decode ratio; assumption anchored on the paper's end-to-end), `m_half 32` (assumption).
  Consequence: with `cpu 0.5 B/param` a CPU GEMV (65 µs for 2048²) can beat the CIM's INT8 pool time
  (97 µs); it is excluded from the search unless `allow_cpu_compute=True`, as in the paper (§4).
* **F8** — the profile `sync_us` blocks are removed; `tiling.SYNC_US` is the one table
  (`Platform.sync_us` mirrors it). **Known gap 7 closed** (F14); gap 6 closed (F4, ADR-0009).

## Revisions 2026-10-23 round 2 (after matrix v1; branch `hetero`)

Fix for matrix-v1 anomaly 1 (`cim-mobile-soc/experiments/RESULTS_v1.md`): a CPU-only decode BEAT the
GPU-only decode (1B 30.9 vs 21.3 tok/s, 8B 6.55 vs 5.21) — the inverse of HeteroInfer Fig.12 (llama.cpp
on the CPU is ~1.33 × slower than MNN on the GPU end to end: InternLM-1.8B 19.38 / 14.20 s, Llama-7B
63.39 / 48.76 s) — because `CpuGemmModel` had no per-kernel floor and its `gemv_efficiency` 0.217 was
anchored per op ("a GEMV = 1.33 × its W4 pool time") while the GPU carries its fitted 60 µs GEMV floor
and 20 µs elementwise floors and the pool overlaps the CPU bytes. Plus two sensitivity hooks. Regression
tests: `tests/test_hetero_round2.py` (G1/G3). The legacy path (no profile: AllCim / CimHetero) is
byte-identical (floor 0, knob ignored; `validate_e2e_l4.py` unchanged, snapshot tests green).

* **R2-A — per-kernel CPU floor** `units.cpu.gemm.kernel_floor_us` (`CpuGemmModel(kernel_floor_us=)`,
  `Platform.cpu_kernel_floor_us`; profile path only, legacy 0): the multi-thread fork/join + dispatch
  cost of a 4-core NEON kernel — an ASSUMPTION (no CPU kernel-launch measurement), placed in the same
  "tens of µs" band as the GPU/NPU launch floors. Charged once per GEMM/GEMV kernel, twice per attention
  composite (QK^T + S·V), and once per FUSED elementwise chain head on the CPU (the ADR-0009 rev F4 rule;
  `tag_fused_chains` now tags CPU nodes; `sampling` = control plane, no floor) on top of the calibrated
  m4_cpu per-op cost (whose `overhead_op` is the single-thread numpy kernel's own fixed cost).
* **R2-B — END-TO-END refit** (`tools/analysis/fit_cpu_gemv.py` → the profile + `validation/reports/
  hetero/cpu_gemv_fit.json`): `(kernel_floor_us, gemv_efficiency)` chosen so that the SIMULATED
  `cpu_only` decode tok/s = `gpu_only` / 1.33 on BOTH llama-3.2-1b (InternLM-1.8B proxy) and
  llama-3.1-8b (Llama-7B proxy) at ctx 1024 / prefill 256 / decode 128 through the real Platform +
  scheduler + engine (the runner's kv sampling). Floor on the 10–30 µs grid, efficiency by bisection to
  the minimax point (both deviations monotone in it):

  | floor µs | gemv_eff | dev 1B | dev 8B |
  |---|---|---|---|
  | 10 | 0.1094 | +8.8 % | −8.8 % |
  | 20 | 0.1181 | +5.9 % | −5.9 % |
  | **30** | **0.1280** | **+3.1 %** | **−3.0 %** |

  Chosen: floor 30 µs (the top of the band: a larger floor would fit better, which says the model's
  per-kernel overhead share is at least this large), `gemv_efficiency 0.217 → 0.128`. Runner check on
  the written profile: 1B gpu_only 21.31 / cpu_only 16.52 = **1.290**, 8B 5.213 / 3.800 = **1.372**
  (target 1.33 ± 10 %; the paper's two ratios are 1.365 and 1.300 — our size trend runs the other way,
  within the band). The same values are mirrored into `soc_rk3588_cim.json` (same `cpu_rk3588` A76
  proxy). CPU weight bytes stay 0.5 B/param (llama.cpp Q4), the CPU single-stream cap 43 GB/s.
  **Honesty: the CPU GEMM model is SIMULATED — anchored only on Fig.12's end-to-end ratio, two
  published seconds; no CPU GEMM or kernel-launch silicon. Any claim that rests on the CPU as a compute
  unit must be validated on silicon.** The `cpu_gemv_efficiency` config knob keeps overriding the
  fitted efficiency (floor unchanged); `runner` echoes `cpu_kernel_floor_us`.
* **R2-C — `platform.cim_weight_bytes_per_param`** (`Platform(cim_weight_bytes_per_param=)`, needs a
  profile; number in (0, 4]; provenance-flagged `simulated` when it differs from the profile's 1.0):
  streamed CIM weight bytes per parameter. 1.0 = the INT8 crossbar weights streamed as stored (the
  Metis basis); 0.5 = W4 storage with on-load unpack into the INT8 array — a DESIGN OPTION with no
  silicon, the knob that decides H2 (matrix v1: the CIM's INT8 bytes vs the GPU/CPU's W4). Bytes only;
  the CIM compute law is untouched. Runner echoes `cim_weight_bytes_per_param` (and, as before,
  `cim_n_cores`).
* **R2-D — `price_table.py --cim-n-cores N`** prices the table at another CIM core count (named in the
  output and the file suffix `_cim<N>c`; EXTRAPOLATED).

## Revisions 2026-10-26 (CIM beyond-capacity on the profile path; branch `hetero`)

User decision: the Metis Card's beyond-capacity behaviour — the ~70 GOP/s DRAM "spill floor"
(`multitile_floor_gops`, the Phase-1.5 residency cliff) — is a Metis AIPU **implementation artifact**, not
a property of SRAM-CIM, so the SoC-profile CIM must not inherit it. F12 above had kept the floor as the
"array's measured ingest rate" on the profile path; this revision replaces it there. The legacy path
(`soc_profile` unset, `scale_mode="legacy"`, the Card topology / L4 gate) keeps the cliff byte-identically
(`validate_e2e_l4.py` unchanged, snapshot tests green). Regression tests: `tests/test_cim_residency.py`;
`test_hetero_fixes.py::test_f12…` now asserts "physical == legacy at s = 1 **within capacity**".

* **R3 — re-programming instead of a spill floor** (`CimTileModel(scale_mode="physical", program_BW_GBs=,
  residency_policy=, weight_bytes_per_param=)`; `Platform.cim_program_BW_GBs` / `.cim_residency_policy`).
  Resident regime (K·N ≤ knee × s, knee ∝ n_cores as in F12): unchanged (single-tile G_eff tiling law /
  affine resident law / prefill tile law). Beyond capacity the weights that do not fit must be
  re-programmed into the array from the SHARED DRAM pool for every use:
  `latency = max(compute, program_us)` with `compute` = the resident law for the WHOLE GEMM (decode:
  the Card-calibrated affine resident rate `a + b/s·K·N` extended past the knee — continuous at the
  knee, no 3.5× collapse — or the G_eff law for a single-tile shape; prefill: the tile law, F12) and
  `program_us = bytes_not_resident / program_BW_GBs`, `bytes_not_resident = max(0, K·N − knee × s) ×
  weight_bytes_per_param`. Prefill: the same rule, the program term once per prefill forward (per GEMM
  node). The DRAM bytes themselves are metered ONCE by the engine's shared pool (F2): the program term is
  the on-chip write time overlapping the stream, hence `max`, never a sum, and the engine then takes
  `max(latency, pool time)` as for every unit. `is_extrapolated()` is True beyond capacity (no silicon);
  the price provenance says `BEYOND RESIDENT CAPACITY … ASSUMPTION, no silicon`.
* **`units.cim.program_BW_GBs`** (array write bandwidth; P1 43.0, P2 29.97 = the unit's single-stream
  DRAM cap `single_stream_eff_BW_GBs × cim.stream_eff`, which is also the default when absent): at the
  cap, re-programming is DRAM-stream-bound — an ASSUMPTION (no array-write measurement); a lower value
  models a slow array-write path (sensitivity axis).
* **`units.cim.residency_policy ∈ {pin_first, none}`** (default `pin_first`): `pin_first` assumes the
  first knee × s elements stay pinned across tokens (decode) so only the remainder is re-programmed per
  use; `none` re-programs the whole beyond-capacity set per use (nothing pinned). Within capacity
  nothing is re-programmed under either policy. ASSUMPTION, sensitivity axis.
* **`cim_weight_bytes_per_param`** (R2-C) is unchanged (profile 1.0, knob 0.5) but now also scales the
  re-programmed bytes (W4 storage halves the program term); within capacity it stays bytes-only.
* Physical-mode `spills()` is capacity arithmetic for ANY shape (K·N > knee × s), no single-tile
  exclusion (legacy keeps the cliff model's multi-tile-only domain): at 8 cores (W = 4096) a 4096²
  single tile is 0.46 M elements past the 16.3 M knee → an 11 µs program term, below its compute.
* Price-table consequence (sd8gen3_cim, 4 cores, CIM compute µs, before → after): decode 4096×1024 39.4
  → 39.4 (resident); 4096×4096 480.0 → 200.4 (= 8.62 MB / 43 GB/s; its pool time 390 µs now dominates);
  4096×14336 1680.1 → 1175.8 (pool 1366); 4096×128256 (lm_head) 15031 → 12027 (pool 12220); prefill
  M=256 4096×4096 259.7 → 259.7 (tile law dominates). At 1 core (knee 2.04 M): decode 4096×1024 120.0
  → 112.1, 4096×4096 480.0 → 403.0, 4096×14336 1680.1 → 1372.8 (the affine resident law × 4 dominates
  the program term); prefill M=256 4096×4096 1038.8 → 1038.8. The 8B `cim_only` decode therefore
  becomes pool-bound on its big GEMVs instead of floor-bound (`validation/reports/hetero/price_table_*.json`).
