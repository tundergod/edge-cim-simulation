# Phase 1.3 — heavy-fidelity engines (ONNXim / Ramulator2) findings

Phase 1.3 drops two heavy C++ sims into the **same** Phase-1.2 interface as drop-in `engine=`
backends: **ONNXim** (NPU) and **Ramulator2** (memory, LPDDR5). Same constructor
`Engine(spec, engine=).predict(wl)`, same frozen return `{latency_us, bound, provenance}`. Both are
`simulated, NOT silicon-validated` (ONNXim ≠ issue #13). This branch is **stacked on `phase-1.2`**
(it imports the 1.2 spec/engine layer; preflight gate passes).

> **Positioning (unchanged from the plan):** the heavy sims' *signature* value — Ramulator2's
> multi-unit contention, the token-by-token whole machine — is **Phase 2**. Phase 1.3's job is
> (i) cross-check the 1.2 analytic single-stream trend, and (ii) **interface-readiness** so Phase 2
> plugs in instantly. Immediate payoff is low by design; this is "build it ready".

## What this session delivered (non-blocked)

- **Interface-ready `engine=` drop-in adapters** — `MemoryModel(spec, engine='ramulator2')` and
  `NpuModel(spec, engine='onnxim')`, same constructor + frozen contract, verified by
  `tools/analysis/check_phase1_3.py` (drop-in interchange + faithful fallback + honesty tags).
- **Build runbooks + cache contract** — `tools/ramulator2/README.md`, `tools/onnxim/README.md`:
  the exact clone/build steps + the JSON cache format each adapter reads
  (`simulated/ramulator2/lpddr5_eff.json`, `simulated/onnxim/rknpu2_sim_matmul.json`).
- **ADR-0002 reconcile** — the "Ramulator2 → Phase 2" wording is corrected everywhere to
  "single-stream LPDDR5 cross-check = Phase 1.3 (`engine='ramulator2'`); multi-unit contention =
  Phase 2": `docs/adr/0002-memory-model.md` (revision), `OVERALL.md` (risk #6),
  `docs/phase1.1-findings.md`, `docs/report/phase1.1/chapters/A2-m2-memory.md`,
  `tools/analysis/fit_m2.py` (+ regenerated `m2_lpddr5.json`).

## ⛔ Deferred (C++ build not authorized this session)

The harness did not authorize **cloning/building external code** (Ramulator2 / ONNXim), the same
class of guardrail that blocked the CIM-Card SSH — and the user was offline to grant it. So, per the
plan's documented C++-build fallback (**single-point failure → documented fallback, does NOT affect
the merged 1.2**):

- The heavy engines **fall back to the Phase-1.2 analytic result** with an honest provenance note
  (`engine='ramulator2'|'onnxim' requested but C++ build deferred -> ANALYTIC fallback`, risk #6/#7).
- **Not produced** (need the built sims): the heavy-sim per-shape data
  (`simulated/{ramulator2,onnxim}/*.json`), the ONNXim-vs-analytic and Ramulator2-vs-analytic
  **delta reports** (`validation/reports/phase1.3/*.json`), figures `N3` / `M2-ramulator2`, and
  their report chapters. The driver scripts (`mem_ramulator2.py`, `npu_onnxim_trace.py`,
  `build_*`) are specified in the READMEs but not written, since they are untestable without the
  built binaries.
- **Open item (couldn't check without building):** whether Ramulator2 ships an LPDDR4/4x preset —
  ADR-0002 flags it `assumption`, "confirm against `src/dram/impl/` when building".

## To unblock

Grant external-build authorization (add a Bash permission rule for the clone/build commands), then
follow `tools/ramulator2/README.md` + `tools/onnxim/README.md`. The adapters **automatically** use
the cache once it exists — no engine code change. CIM trust and the 1.2 analytic layer are unaffected.
