# Change LOG

Repo-level changelog for cross-cutting corrections (units, architecture framing,
consistency sweeps) that span more than one phase. Per-phase work lives in its
`phase-<id>` branch + PR; this file records the few changes that retroactively touch
already-merged prior-phase deliverables, so the history of *why* they changed is in one place.

Reverse-chronological (newest first).

---

## 2026-06-13 — declutter: `plans/` and `papers/` moved under `docs/`

`plans/` → `docs/plans/` and `papers/` → `docs/papers/`, completing the half-finished
`218c1a6` intent (which created `docs/plans/` for new plans but never moved the existing
ones, splitting plans across two dirs). Root-level + code references updated to `docs/…`
(CLAUDE.md per-phase workflow, CONTEXT.md repo index, README, OVERALL, four `tools/**` docstrings);
relative markdown links inside `docs/` already resolve correctly post-move and were left as-is;
historical handoffs/old LOG entries left verbatim.

---

## 2026-06-06 — Phase 1.2: shared engine interface + spec layer (touches 1.1 callers)

Phase 1.2 introduces `simulator/models/engine.py` (`UnitEngine(spec, engine='analytic')` +
`Workload` + frozen `predict()` keys `{latency_us, bound, provenance}`) and `simulator/specs/`
(9 swappable specs + loader). The CPU and memory engines were rewritten from their Phase-1.1
constructors (`CpuModel(params)`, `MemoryModel(pcie, lpddr5)`) to the spec-based `(spec, engine=)`
form — which **retroactively touched three already-committed Phase-1.1 callers** (decision A):

- `tools/analysis/recompose_e2e.py` — migrated to the spec engines (`cpu_rk3588`, `mem_lpddr4x`).
  The 8B decode hold-out **gate is unchanged (9.5%)** — it uses op-profile bytes + vendor tok/s +
  a BW fit, not the model classes. The only output change: `recompose.json` `cpu_support_us`
  (62023→15007 µs), a *standalone transparency* term explicitly "absorbed in BW, NOT added".
- `tools/analysis/fit_m2.py`, `tools/analysis/fit_m4_cpu.py` — migrated; both **regenerate their
  frozen Phase-1.1 reports/params byte-identical** (verified by empty git-diff).

GPU is additive (new `m4_gpu_roofline.py`; the 1.1 `MaliGpuModel` is untouched). The 1.1 measured
params (`m2_pcie`, `m2_lpddr5`, `m4_cpu.json`) are superseded by the spec layer but kept as the 1.1
record. Ramulator2/ONNXim reconcile (the "Phase 2" → "Phase 1.3" wording) is deferred to the 1.3 PR.

---

## 2026-06-05 — Establish Phase 1.x structure: rename `phase1` → `phase1.1`

Phase 1 (component modeling & validation) is split into two sub-phases by **data source**
(mirrors the existing Phase 0.1–0.4 convention):

- **Phase 1.1** (done) — components calibrated against our own Metis silicon measurements
  (CIM, GPU, CPU-softmax, DRAM-LPDDR4x, M5-trace, M7-energy) + the e2e recompose hold-out.
- **Phase 1.2** (next) — components lacking our own silicon measurement, modeled via
  simulator/datasheet/literature: **NPU** (ONNXim + HeteroInfer, #13), **DRAM SoC hierarchy**
  (LPDDR5 + L1/L2 residency), **full CPU compute model**. Labeled `simulated, not
  silicon-validated` (weaker gate). See OVERALL.md § 階段總覽.

The already-merged Phase 1 work IS Phase 1.1, so its artifacts were renamed (git mv, history
preserved):

- `docs/report/phase1/`        → `docs/report/phase1.1/` (incl. `phase1-report.pdf` → `phase1.1-report.pdf`)
- `docs/figures/phase1/`       → `docs/figures/phase1.1/`
- `validation/reports/phase1/` → `validation/reports/phase1.1/`
- `docs/phase1-findings.md`    → `docs/phase1.1-findings.md`
- `plans/phase-1.md`           → `plans/phase-1.1.md`

Updated all in-file path references (8 fit/validate scripts, the report builder + its
embed regex, chapter figure links, the plan) and relabeled report prose `Phase 1 → Phase 1.1`.
Reclassified three deferrals from `Phase 2 → Phase 1.2` in the report (NPU build, L1/L2 SRAM
residency, full CPU model) — these are now 1.2, not integration. HTML + PDF regenerated.

**Not renamed (intentional):** the merged git branch `phase-1` (history); tool *filenames*
`phase1_figs.py` / `build_phase1_report.py` (kept to avoid churn — they build the Phase-1
umbrella reports; only their internal paths changed). Older LOG entries below still name
pre-rename paths (accurate at the time).

## 2026-06-05 — Split `docs/report/` by phase

`docs/report/` mixed a flat Phase-0 report (`report.html`/`report.pdf`) with the already-
nested `phase1/`. Split by phase to mirror `validation/reports/phase1/` and the `phase1/`
report layout:

- `docs/report/report.html`  → `docs/report/phase0/index.html`
- `docs/report/report.pdf`   → `docs/report/phase0/phase0-report.pdf`

Now symmetric: `docs/report/phase0/index.html` + `docs/report/phase1/index.html`. The
Phase-0 report is hand-written (no generator); the older LOG entry below still names its
pre-move path `docs/report/report.html` (accurate at the time).

## 2026-06-05 — Post-Phase-1 consistency audit: repo-wide unit + architecture sync

After merging Phase 1 (PR #15), a consistency audit found that the Phase-1 corrections
(discovered via issues #17/#18 and the AIPU ISSCC-2024 ingest) had **not** been propagated
back into the already-merged Phase 0.3 deliverables, leaving the repo internally inconsistent.
Synced them.

**1. CIM throughput unit: `GFLOP/s` → `GOP/s` (INT8).**
Metis CIM is an INT8 integer engine, so its throughput is giga-*ops*/s, not giga-*float*-ops/s
(issue #18). Phase 1 fixed this in its own report; this sweep fixed the remaining spots:

- `tools/analysis/recompose_e2e.py` + its two generated JSONs (`validation/reports/phase1/recompose.json`,
  `measurements/metis_card/twopillar_prediction_fitted.json`) — the "204 GFLOP/s" note string.
- `tools/plotting/phase03_figs.py` — `fig5_cim_throughput` docstring + y-axis label (figure regenerated).
- `docs/phase0.3-findings.md` — headline + A1d.3 / A1d.6 / tiled-op rows.
- `docs/report/report.html` (standalone Phase 0 report) — §5.1 + Table 4 rows.
- `plans/phase-0.3.md`, `plans/phase-1.md` — CIM-context label refs.

**2. CIM crossbar framing: "2048×2048 array" → "4 cores × 512×512, effective 2048 output width".**
The Metis AIPU is a **quad-core** D-IMC engine (512×512 INT8 per core), not one 2048×2048
array; the N≈2048 tiling boundary is the 4-core combined output width
(see `papers/metis-silicon/metis-aipu-isscc2024.md`). Reframed in:

- `docs/phase0.3-findings.md` (device-envelope row), `docs/report/report.html` (§5 + Table 4).

**Deliberately NOT changed (correct as-is):**

- **GPU / NPU `GFLOP/s`** — Mali-G610 and RKNPU2 run FP16, so float-ops/s is correct
  (`fit_m4_gpu.py`, `phase1_figs.py` P4 ksweep, `phase0.3-findings.md:51` GEMM-kernel line,
  `run_rknpu2.py`, the GPU/NPU lines of both plans).
- **`papers/`** — external curated source material; left verbatim (editing sources would
  falsify them). The one external "46.2 GFLOP/s" desktop-investigation note stays as published.
- **`dev_gflops` JSON field key** in measurement data — an internal schema identifier, not a
  displayed unit; renaming it would break readers. Only human-facing labels were changed.

Landed via PR (branch `fix/cim-unit-recompose`).
