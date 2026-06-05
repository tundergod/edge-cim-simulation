# Change LOG

Repo-level changelog for cross-cutting corrections (units, architecture framing,
consistency sweeps) that span more than one phase. Per-phase work lives in its
`phase-<id>` branch + PR; this file records the few changes that retroactively touch
already-merged prior-phase deliverables, so the history of *why* they changed is in one place.

Reverse-chronological (newest first).

---

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
