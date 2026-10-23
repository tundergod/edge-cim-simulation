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
