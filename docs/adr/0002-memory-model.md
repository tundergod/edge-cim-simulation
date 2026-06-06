# ADR-0002 — Memory model: Ramulator2, representative-iteration, swappable

Status: Accepted (2026-06-03)

## Context
Memory access is the crux (decode memory wall). The subfield standard backend is Ramulator2 / DRAMsim3 (AttAcc, SpecPIM, Duplex, CENT, PAPI). Our simulated SoC has CIM accessing host LPDDR over PCIe (MMIO unified) while NPU/GPU/CPU access the same LPDDR natively. Full per-cacheline simulation of full generations is infeasible: 8B INT8 decode ≈ 1.25e8 **64-B-cacheline** requests/token (assumes a 64 B burst/cacheline; reconfirm against the actual LPDDR burst when configuring Ramulator); over a representative ~340-token decode (≈ GSM8K mean) ≈ 4e10 requests ≈ hours per config.

## Decision
**Ramulator2** as the DRAM/LPDDR backend at **cacheline granularity** (full DRAM fidelity). Make it feasible by simulating **representative iterations** — one prefill + decode tokens at several KV lengths (e.g. 128/512/1024; 2048 only where a silicon artifact supports it — Llama precompiled context is ≤1024 and >1024 silently yields 0 tokens per voyager-sdk.md, phi3 reaches 2048) — and computing full-generation latency by interpolating the (smooth, steady-state) per-token latency across those points; verified once against a measured silicon per-token curve **at the KV lengths that have an L4 anchor**. Add a **PCIe serial model** in front of CIM (Aetina link ≈3.9 GB/s usable, Gen3 ×4) and a **calibrated interconnect-efficiency layer** — Ramulator models the DRAM device, not the SoC NoC arbitration, so the achievable-vs-peak gap (HeteroInfer's 68→~60 GB/s is illustrative) is **calibrated from Aetina concurrent micro-benchmarks**, not imported. Hide all of this behind a **swappable `MemoryModel` interface**; the analytical contention model (the "(i)" option) is the fast-DSE fallback, not the primary.

Representative-iteration preserves full memory fidelity (every simulated token is full cacheline-level); a literal full-run would force coarser (tile) granularity and *lower* fidelity.

### Revision (2026-06-06, Phase 1.3) — staging of the Ramulator2 backend
The Ramulator2 backend is staged across phases (it was loosely "Phase 2" in earlier wording):
- **Phase 1.2** ships the **analytic** LPDDR4/4x/5 effective-BW model as the primary `MemoryModel` (the "(i)" fast-DSE option), calibrated to the LPDDR4x 24.2 GB/s decode anchor.
- **Phase 1.3** drops the **Ramulator2 LPDDR5** backend in behind the SAME swappable interface as `MemoryModel(spec, engine='ramulator2')` — a drop-in for `engine='analytic'` — to **cross-check the single-stream** per-token BW/latency at representative KV lengths (128/512/1024). *(This session: the Ramulator2 C++ build was not authorized; the `engine='ramulator2'` branch is wired and falls back to analytic with a documented note until the build lands — see `docs/phase1.3-findings.md`.)*
- **Phase 2** uses Ramulator2 for its **signature value: multi-unit contention** (CIM+NPU+GPU+CPU sharing LPDDR) and the token-by-token whole-machine run.

LPDDR4/4x have **no first-class Ramulator2 preset** — **CONFIRMED 2026-06-06** by building Ramulator2: `src/dram/impl/` ships DDR3/4/5, GDDR6, HBM/2/3, and **LPDDR5 only** (no LPDDR4/4x). So the analytic LPDDR4/4x specs cannot be cross-checked by a stock Ramulator2 preset — a 1.3/2 config item (port an LPDDR4 timing set, or restrict the Ramulator2 cross-check to LPDDR5).

## Consequences
M2 budgets Ramulator2 LPDDR5/PIM config + Python co-sim (OVERALL.md risk #6). The per-token-smoothness assumption is validated once against silicon. The swappable interface is also what enables the L4 validate-then-swap bridging (ADR-0006).
