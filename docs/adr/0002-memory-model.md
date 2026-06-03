# ADR-0002 — Memory model: Ramulator2, representative-iteration, swappable

Status: Accepted (2026-06-03)

## Context
Memory access is the crux (decode memory wall). The subfield standard backend is Ramulator2 / DRAMsim3 (AttAcc, SpecPIM, Duplex, CENT, PAPI). Our simulated SoC has CIM accessing host LPDDR over PCIe (MMIO unified) while NPU/GPU/CPU access the same LPDDR natively. Full per-cacheline simulation of full generations is infeasible: 8B INT8 decode ≈ 1.25e8 **64-B-cacheline** requests/token (assumes a 64 B burst/cacheline; reconfirm against the actual LPDDR burst when configuring Ramulator); over a representative ~340-token decode (≈ GSM8K mean) ≈ 4e10 requests ≈ hours per config.

## Decision
**Ramulator2** as the DRAM/LPDDR backend at **cacheline granularity** (full DRAM fidelity). Make it feasible by simulating **representative iterations** — one prefill + decode tokens at several KV lengths (e.g. 128/512/1024; 2048 only where a silicon artifact supports it — Llama precompiled context is ≤1024 and >1024 silently yields 0 tokens per voyager-sdk.md, phi3 reaches 2048) — and computing full-generation latency by interpolating the (smooth, steady-state) per-token latency across those points; verified once against a measured silicon per-token curve **at the KV lengths that have an L4 anchor**. Add a **PCIe serial model** in front of CIM (Aetina link ≈3.9 GB/s usable, Gen3 ×4) and a **calibrated interconnect-efficiency layer** — Ramulator models the DRAM device, not the SoC NoC arbitration, so the achievable-vs-peak gap (HeteroInfer's 68→~60 GB/s is illustrative) is **calibrated from Aetina concurrent micro-benchmarks**, not imported. Hide all of this behind a **swappable `MemoryModel` interface**; the analytical contention model (the "(i)" option) is the fast-DSE fallback, not the primary.

Representative-iteration preserves full memory fidelity (every simulated token is full cacheline-level); a literal full-run would force coarser (tile) granularity and *lower* fidelity.

## Consequences
M2 budgets Ramulator2 LPDDR5/PIM config + Python co-sim (OVERALL.md risk #6). The per-token-smoothness assumption is validated once against silicon. The swappable interface is also what enables the L4 validate-then-swap bridging (ADR-0006).
