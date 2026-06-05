# Phase 1 — component modeling & validation findings

Phase 1 turns the committed Phase 0.2/0.3 measurements into **closed-form latency
equations** (replacing lookup tables) + per-component validation (M1–M7). Software-only,
calibrated against the two Metis boards. **All numbers below are regenerable** from
committed JSON via `tools/analysis/fit_*.py`, `tools/analysis/recompose_e2e.py`,
`validation/validate_*.py`, and `tools/plotting/phase1_figs.py`.

> **Calibration-scope declaration (important): Phase 1 is a *decode-calibrated* model.**
> The whole prefill path is analytic and **unvalidated** (see §Prefill). The only hard gate
> is the 8B decode hold-out (≤25%). Per-op equation gates are ADR-0006 (median ≤10%, p95 ≤20%).

## Gate summary

| Component | Equation | Gate | Result | Pass |
|---|---|---|---|---|
| M1 CIM tile | `dev_lat = Σ N-tiles 2·M·K·n/G_eff(n,K)` (quad-core 512×512, **GOP/s**) | 2D G_eff(N,K) native fit median≤10%, p95≤20% | median **2.7%**, p95 **14.9%** | ✅ |
| M2 PCIe/DMA | `floor + bytes/BW` (911µs, 3.9 GB/s fixed) | sanity + boundary recorded | positive, monotonic | ✅ |
| M2 LPDDR5 | analytic eff-BW (Ramulator2 → Phase 2) | eff ∈[0.4·peak,peak], brackets 24 | 24.2 / 51.2 GB/s (47%) | ✅ |
| M4 GPU (Mali) | `attn_bmm = 27.74 + 0.442·kv` µs (offload) | median≤10%, p95≤20% | median **0.6%**, p95 **1.1%** | ✅ |
| M4 CPU (A76) | softmax `a + b·kv`; others constants | median≤10%, p95≤20% | median **0.3%**, p95 **1.8%** | ✅ |
| M4 NPU | — | blocked on #13 | placeholder | ⏸ |
| M5 trace | deterministic (oracle) | 0 orphans, semantic covered | 4 models, 0 orphans | ✅ |
| M7 energy | spec (15 TOPS/W, pJ/bit, A76 W) | sanity + ±20% no-flip | memory-dominated, robust | ✅ |
| M3 / M6 | — | contract only (Phase 2) | tunables + gaps recorded | ✅ |
| **Recompose** | `tok_s = BW_eff/weight_bytes` (fit 1b/3b→8b) | 8B hold-out ≤25% | **9.5%** (2.44 vs 2.70) | ✅ |

## Op-category coverage (9 + conversion)

| op category | model / unit | status |
|---|---|---|
| matmul | M1 CIM (+M4 GPU ref) | ✅ decode; prefill M≥512 unvalidated |
| attention bmm | M4 Mali (offload); C4 CIM-penalty | ⚠ decode OK; prefill scaling 1 pt |
| softmax | M4 CPU (linear-in-kv) | ⚠ decode OK; prefill S×S not covered |
| norm / ffn / rope / residual | M4 CPU (constants) | ⚠ decode OK; prefill ×S analytic, unvalidated |
| kv_cache | M2 analytic `kv_bytes/BW_eff` | analytic, unvalidated (12.6–33.5% of LongBench decode bytes, 8B 22.2%) |
| embedding | host gather | decode≈0 (folded into overhead); prefill ~192MB analytic |
| **conversion (quant/dequant)** | scheduler-inserted (M6) | ❌ **headline gap** — ADR-0004 Phase-0.2 calibration never done; M6 tunable + measurement gap |

## Per-component detail

### M1 — CIM tile
**Architecture (ISSCC 2024, papers/metis-silicon/metis-aipu-isscc2024.md):** Metis is **quad-core**,
each core a **512×512** INT8 D-IMC crossbar. The simulator's minimum unit is one core (512 wide)
and **`n_cores` is a free parameter** (=4 for Metis → effective output width 2048). Throughput is
INT8 **GOP/s**, not FLOP/s. Latency tiles the output N into passes of width ≤ n_cores·512 and **sums
per tile (rising** — a partial last tile adds its own size, not a full tile). The 2D throughput
`G_eff(N,K) = Gmax·N/(N+Na)·K/(K+Kb)` (Gmax=333.7, Na=577, Kb=574 GOP/s) is fit on **13 native
single-tile points** (K·N ≤ 4.19M, the largest natively measurable) → **median 2.7%, p95 14.9%,
max 17.6%**.
- **Honest native vs generated:** only **5 of 16** proj_decode shapes are native measurements
  (1B q_o; 1B/3B/8B/Qwen kv, all N≤2048); the other **11 are model tile-sum output** (K·N > the
  ~6 M-param envelope, no measurement) — **NOT** presented as 0%-error.
- **K-effect is FITTED** (retracts the prior "unfittable, single point" claim): wider K raises
  throughput (N=512: K2048→112.6 vs K3584→147.4; N=1024: K2048→170 vs K4096→227); the 2D fit
  captures it, and the high-K corner (8B kv K4096×N1024) is over-predicted **+21.5%**, shown not hidden.
- **Multi-tile = UNVALIDATED:** the one native multi-tile point (N=4096,K=1024 = 37.1µs) is
  over-predicted **+36%** (continued-rise tiling is pessimistic); everything K·N > 4.19M is
  extrapolation (board offline → cannot re-measure; issues #2/#11/#17). lm_head N≈128k/152k and
  prefill M≥512 stay analytic, no measurement.
- **Device envelope** = the PCIe-IOMMU window (default ~14 MB, forum #1330; Alpha has no real
  on-card DRAM), **NOT** the 32 MB L2 / 52 MiB on-chip SRAM. Compute ceiling (~52 TOPS/core) not
  modeled — decode never approaches it (issue #16).

### M2 — memory / PCIe
PCIe `transfer_us = 911µs + bytes/3.9GB/s` (fixed params; no per-shape sweep collected). **Floor
applies to discrete host↔device transfers only** (KV-reload, activation handoff, conversion
traffic); decode weight-streaming uses the BW term (no per-call floor) in the production
prediction. On the Alpha board itself every decode-GEMV paid the floor (topology artifact, not
extrapolated). LPDDR5 effective BW 24.2 GB/s (= measured decode wall), 47% of the 51.2 GB/s
JEDEC LPDDR5-6400 peak; Ramulator2 deferred to Phase 2 (ADR-0002 swappable). kv_cache append =
analytic `kv_bytes/BW_eff`, **unvalidated**. **No SRAM L1/L2 residency model** (Alpha l2/ddr
ratio ≈1.00–1.01, no on-card DRAM) — recorded so Phase 2 doesn't build it.

### M4 — GPU (Mali), CPU (A76), NPU
- **GPU:** `attn_bmm_us(kv) = 27.74 + 0.442·kv` (single-head decode QK^T+S·V, f16) — the
  attention-offload reference (median 0.6%). GEMM absolute throughput is a **lower bound**
  (unoptimised kernel; ksweep saturates ~20 GFLOP/s by M=128) — only the shape-trend is fit.
- **CPU:** softmax `a+b·kv` per (model,dtype) (median 0.3%); rmsnorm/rope/residual/swiglu/argmax
  are per-(model,dtype) constants (no within-op sweep). fp16 is numpy-emulated → **upper bound**.
  Measured latencies, not analytic FLOPs (issue #10). Prefill applies ×S (analytic, unvalidated).
- **NPU:** placeholder, **blocked on issue #13** (rknpu2_matmul.json not collected). The
  attention-offload thesis stands on the GPU comparison alone.

### M5 — workload / trace (validated on known inputs)
Deterministic; no fitted params. Reuses the Phase 0.1 inventory oracle: all 4 models
`all_semantic_covered=true`, and **0 orphan ops** across ~3300 profile rows each (every
op_profile op was traced from HF; counts from inventory, no hand ×layers).

### M7 — energy (spec-based, ADR-0005)
No power telemetry → spec estimation. 8B decode per-token = CIM **1.0 mJ** / DRAM **240 mJ** /
CPU **15 mJ** → **memory-dominated** (the CIM-centric thesis: cheap CIM compute, memory is the
wall). Robust to ±20% on every coefficient (16 corners, 0 flips). Energy is estimated, not measured.

### M3 / M6 — contract only (behavioral validation = Phase 2)
- **M6:** tunables include `precision_boundary_conversion_op_cost` flagged as the **ADR-0004
  measurement gap** (conversion-op cost was supposed to be calibrated in Phase 0.2 but never was;
  the headline mixed-precision contribution has no cost basis yet — Phase 2 must measure it).
- **M3:** tunables include the `bandwidth_contention_knee` (ADR-0001 obligation: reproduce the
  ~60 GB/s saturation knee). Acceptance criteria cite the ADR-0006 Phase-2 system thresholds
  (e2e ≤15%, contention knee ≤15%).

## End-to-end recompose (L1→L4 capstone)
Decode backbone `tok_s = BW_eff / weight_bytes` with weight_bytes summed from op_profile per-sig
decode-matmul bytes (matches the closed form to 0.1%); `BW_eff` fit on 1B+3B (**18.33 GB/s**),
predicting held-out **8B = 2.44 vs measured 2.70 tok/s (9.5% error)** ✅. Implied per-model BW
16.16 / 20.51 / 20.27 GB/s (the documented size trend). The 911µs Alpha floor is excluded
(production = on-card DRAM). Non-streaming terms (CPU support, GPU-offload attention, kv_cache)
are reported standalone — they are already absorbed into `BW_eff` at the fit point, so adding
them would double-count (Phase-2 fidelity item). CIM-attention penalty (C4) **31–46 ms/token** is
reported separately as the offload justification (≈2 orders slower than GPU-native, 96–370× over kv 129–1025, P7).

### Prefill (ungated, unvalidated)
Vendor `ttft_s_median` (8B = 3.79 s) implies **~4.1 TOPS** effective prefill GEMM throughput, but
the decode-GEMV throughput (204 GOP/s = 0.2 TOPS) would give an absurd 75 s — i.e. prefill GEMM
runs ~20× faster than decode GEMV and is **unmeasured** (proj M≥512 device-fail; prefill attention
S-scaling 1 pt; softmax S×S). Prefill path unvalidated → Phase-2 gap. (`prefill_ms_median` is a
degenerate vendor field ~0.007 s across all sizes — unused.)

## Gaps & deferrals
- **NPU (M4):** issue #13 — not collected (aetina offline).
- **conversion-op cost (M6):** ADR-0004 Phase-0.2 calibration never done — headline mixed-precision gap.
- **prefill path:** decode-calibrated only; whole prefill path analytic/unvalidated.
- **kv_cache / embedding:** analytic, unvalidated (Phase 0.3 didn't isolate / micro-bench them).
- **Ramulator2:** deferred to Phase 2 (analytic LPDDR5 ships in Phase 1).
- **Phase-2 watch-items:** kv_append vs BW double-count; attention heads×layers rollup.
