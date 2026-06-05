# ADR-0006 — Validation contract, L4 bridging, memory extrapolation

Status: Accepted (2026-06-03)

## Context
Three coupled questions: what acceptance thresholds make the simulator "correct"; how the L4 anchor (production Metis Card, on-card DRAM) validates a simulated host-MMIO topology; and how upward memory extrapolation is justified.

## Decision

### Validation contract (thresholds PROVISIONAL — revisit after the first end-to-end sim)
- **Phase 1 (component):** per-op fit error **median ≤ 10%, p95 ≤ 20%** (report median+p95+max distribution); roofline knee drift ≤ 15%; sanity (no NaN/Inf, monotonic with op size, positive latency).
- **Phase 2 (system):** end-to-end **latency / tok-s ≤ 15%** vs silicon; **contention knee ≤ 15%** (the load-bearing memory-wall check); energy via uncertainty + ±20% sensitivity; SOTA reproduction (ADR-0003).
- **Gate 6c (hard rule):** a component failing its contract may **not** enter integration; integration failures are fixed at the responsible component — **never** masked by integration-layer parameter tuning.

### L4 bridging (on-card DRAM ≠ host-MMIO)
L4 validates **CIM compute + the general memory-wall shape** (decode ∝ weight bytes; effective decode bandwidth ≈24 GB/s — measured on the 1B artifact, weight-streaming-dominated, already folding in KV/activation traffic), **not** the specific PCIe topology. (8B is *more* bandwidth-bound — 4c/1c speedup 1.31×→1.12× from 1B→8B — so this constant is not guaranteed size-invariant, which is exactly what the hold-out checks.) **Validate-then-swap:** run the simulator in an **on-card-DRAM config** to reproduce L4, then **swap** the (swappable, ADR-0002) memory topology to **host-MMIO** for the target prediction, reporting an **A/B topology sensitivity** experiment that makes the data-movement delta explicit.

### Memory extrapolation (conservative)
Extrapolate the **validated mechanistic model** (compute equations + Ramulator reconfigured to the larger DRAM), **not** a fitted line. Validate the extrapolation mechanism by **hold-out** (fit on 1B/3B, predict the measured 8B). Flag any out-of-range prediction explicitly + sensitivity. **Claim up to ~13B / 32GB as a bounded extrapolation with sensitivity** (the hold-out validates the 1B/3B→8B mechanism; 13B/32GB is one further doubling beyond the largest measured point — defensible via the mechanistic model + sensitivity, *not* "precise"; 32GB is one step beyond the real 16GB M.2 Max SKU). 64GB-class is a final ablation, deferred.

### Revision (2026-06-06, Phase 1.2) — no-silicon units carry no numeric gate
The per-op median/p95 numeric gate above presumes silicon to fit against. For a unit with **no silicon**, that gate is **superseded-not-satisfied** (not achieved, not waived) and is replaced by a **trend-shape / lower-bound acceptance**, tagged `simulated`. First case: **M4-NPU (RKNPU2)** — issue #13 micro-benchmark was never collected (board offline), so the Phase 1.2 analytic systolic-roofline ships with the #13 median/p95 silicon gate marked *superseded-not-satisfied* and acceptance = trend-shape agreement with the borrowed HeteroInfer references (staircase knee at the borrowed 32×32, order/shape ≤6×, BW frac 59–66% of 68). This is the "no fake gate" rule: where there is no silicon, there is no numeric acceptance gate. (GPU INT8 = zero data is the same situation: no INT8 gate.) See `validation/contracts/m4_npu.yaml`.

## Consequences
First-sim results recalibrate the thresholds. Need both an on-card-DRAM and a host-MMIO platform config. Hold-out validation consumes the 1B/3B/8B measured points.
