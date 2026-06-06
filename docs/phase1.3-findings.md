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

## Heavy-sim builds (authorized 2026-06-06) — Ramulator2 BUILT, ONNXim toolchain-blocked

- **Ramulator2 — LIVE (v2.1).** The main-branch LPDDR5 `Failed to send refresh!` abort is a known bug
  fixed on the **v2.1 branch** (maintainer, issues #58/#60/#89). Built v2.1 (Python-bindings-only;
  pinned commit `278f1ef`; one Apple-clang-17 `template`-keyword patch in `base/param.h`; `cmake
  -DPython_EXECUTABLE=.venv`). `tools/analysis/mem_ramulator2.py` drives v2.1's own
  `latency_throughput` harness in-process and reads `total_throughput_MBps` (no hand-rolled tCK).
  LPDDR5_6400 saturated streaming runs clean (**peak 98.6% of theoretical = saturation proof**), and
  `engine='ramulator2'` is now LIVE (`check_phase1_3.py` hits the heavy path).
  - **Result (the cross-check):** Ramulator2 (DRAM-**device**) single-stream efficiency **0.92**
    (47.1 GB/s) vs the analytic **system-level 0.65** (33.3 GB/s). NOT a contradiction — the gap is
    the controller/NoC/queueing overhead the analytic captures (silicon-calibrated to the 24.2 GB/s
    decode wall) and Ramulator2's device model omits. This **validates ADR-0002**: the DRAM device is
    not the single-stream bottleneck, and system efficiency is rightly calibrated from silicon, not
    imported from Ramulator2. Analytic 33.3 stays **primary**; Ramulator2's signature multi-unit
    contention value is Phase 2. Report: `validation/reports/phase1.3/m2_ramulator2.json`, figure
    `M2-ramulator2`, chapter `M-memory-ramulator2.md`.
  - **ADR-0002 open item RESOLVED:** Ramulator2 ships **LPDDR5 only**, no LPDDR4/4x (confirmed by
    building) — so the LPDDR4x anchor can't be Ramulator2-cross-checked without porting a preset.
- **ONNXim — cloned, build not completed (toolchain friction).** Pins `conan == 1.57.0` (conan 1.x,
  incompatible with the conan 2.x that installs on Python 3.13); upstream recommends its Dockerfile.
  Lower-value (sim-vs-sim against the already-`simulated` analytic NPU). `engine='onnxim'` falls back
  to analytic (risk #7). Follow-up steps in `tools/onnxim/README.md`.
- **Not produced** (need a clean heavy-sim run): `simulated/{ramulator2,onnxim}/*.json`, the
  delta reports `validation/reports/phase1.3/*.json`, figures `N3`/`M2-ramulator2`, chapters. The
  `engine=` adapters **auto-use** these caches the moment they exist — no engine code change.

## To finish

Resolve the Ramulator2 LPDDR5 refresh-config item (then `mem_ramulator2.py` writes
`simulated/ramulator2/lpddr5_eff.json` and `engine='ramulator2'` goes live); build ONNXim via Docker
or conan 1.57.0. CIM trust and the 1.2 analytic layer are unaffected throughout.
