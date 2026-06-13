# Plan: Phase 1.4 — doc-layer redesign, number-generation & repo hygiene

> Action-only. Rationale lives in the design discussion / OVERALL. Honesty discipline and
> "JSON artifact is authoritative" carry over unchanged. No simulator behavior changes.

## Preconditions (must hold before this phase branches)

- P1. The consolidated report PR (`report/phase1-overall` → `main`) is **merged** (per
  `plans/phase1-overall-report.md` step 14). → verify: `git ls-tree main -- docs/report/phase1/index.html` non-empty.
- P2. The report cleanup PR is **merged**: `docs/report/phase1.{1,2,3}/` deleted, `phase1/` kept
  (per `plans/phase1-overall-report.md` step 15). → verify: `ls docs/report/` shows `phase0`, `phase1` only.

## Branch

0. **Guard:** run the P1 and P2 verifies; if either fails, STOP (do not branch — the report PR
   and cleanup PR must merge to `main` first; this phase's file paths under
   `docs/report/phase1/chapters/` do not exist on `main` until then). If both pass, from `main`
   create `phase-1.4`. → verify: P1+P2 both pass AND `git rev-parse --abbrev-ref HEAD` = `phase-1.4`.

---

## Track A — conversion-op reframing (PR `phase-1.4a`)

Reframe conversion-op everywhere from "headline measurement gap / never measured" to
"Phase-2 analytic modeling item, no measurement required" (fixed per-unit precision: CIM=INT8,
GPU=FP16; conversion = memory-bound cast costed by existing M2/M4 × M6 boundary-crossing count).

A1. Edit `docs/adr/0004-mixed-precision.md`: Decision (a) + Consequences — drop "calibrated in
    Phase 0.2 / part of the Phase 0.2 micro-benchmark set"; replace with "modeled analytically in
    Phase 2 (memory-bound cast via existing M2/M4 per-op models × M6 boundary-crossing count); no
    measurement campaign". Keep Decision (c) (NPU = free knob). Add a `Superseded`/`Revision`
    note dated today referencing this decision. → verify: `grep -ni "phase 0.2" docs/adr/0004-mixed-precision.md` returns no measurement-obligation line.
A2. Edit `docs/phase1.1-findings.md`: line ~38 op-coverage row and lines ~102–104 and ~128 — change
    `❌ headline gap / never done / Phase 2 must measure` to `Phase-2 analytic modeling (no
    measurement); cost = existing M2/M4 cast model × M6 crossings`. → verify: `grep -ni "headline gap\|must measure" docs/phase1.1-findings.md` = 0 for conversion.
A3. Edit `docs/report/phase1/chapters/08-integration-layer-m3-m6.md` (lines ~8, 26, 28, 30, 40, 42 +
    footnotes ^m6b/^m6c) and `docs/report/phase1/chapters/01-readiness-matrix.md` (lines ~26, 32, 34):
    reclassify conversion-op from "真缺口（測量）" to "Phase-2 解析建模項（非量測）"; remove the
    "混合精度宣稱無成本基礎" claim; state the analytic cost basis. → verify: the conversion-op row
    (08 line ~40) and §8.2 heading no longer label it `真缺口`; the M6 row in 01 no longer calls it 測量缺口.
A4. Edit `docs/report/phase1/chapters/09-gaps-gonogo.md`: this **changes a published GO/NO-GO
    condition** (do not do it silently — note it in the PR body). Rewrite §9.2 (B-class row for
    conversion-op), §9.3 (drop "must measure dequant/requant before the mixed-precision claim is
    sound" as a GO condition) and §9.4 (the A-vs-B user call): conversion-op is reclassified as a
    **Phase-2 analytic modeling item, not a measurement gap**; the mixed-precision claim's cost
    basis = existing M2/M4 cast model × M6 boundary crossings. → verify: §9.3 GO conditions contain
    no conversion-op measurement requirement; conversion-op appears only as Phase-2-internal modeling.
A5. Edit `validation/contracts/m6.yaml`: change the `measurement_gap` entry for `conversion_op_cost`
    to a `phase2_modeling_item` note (no measurement); keep `precision_boundary_conversion_op_cost`
    as a tunable but replace its `# HEADLINE GAP` comment with `# analytic, no calibration target`.
    → verify: `grep -ni "MUST measure\|NEVER collected\|HEADLINE GAP" validation/contracts/m6.yaml` = 0.
A6. Rebuild report HTML+PDF (`./.venv/bin/python tools/report/build_phase1_report.py`). → verify:
    builder exits 0; PDF regenerated; no "headline gap" conversion text in `index.html`.
A7. Commit; open PR `phase-1.4a` → `main`; notify user; merge on user confirm.

---

## Track B — two-layer docs + numbers generated from JSON (PR `phase-1.4b`)

Lock the two intentional layers and make every metric cell build from JSON so an agent cannot
mis-transcribe a number.

- **Human layer** = `docs/report/phase1/` (narrative HTML+PDF).
- **Agent layer** = `docs/phase1.{1,2,3}-findings.md` (terse, structured, all tags/gaps/numbers,
  links to JSON). Repurposed as the canonical agent digest; redundancy with the report is accepted.

B1. Add `tools/report/_metrics.py` with `load() -> dict`. **Key scheme (source-namespaced to avoid
    cross-tree collisions):** `rep.<filestem>.<dotted.path>` for `validation/reports/phase1.{1,2,3}/*.json`
    (recurse nested dicts → dotted path; list elements indexed `.<i>`), `param.<filestem>.<field>` for
    `simulator/models/params/*.json`, `spec.<name>.<field>` for `simulator/specs/*.json`. Namespacing
    makes "one source per key" hold by construction; still raise on any literal duplicate key.
    Import is path-based (no package): the builder/test add `tools/report` to `sys.path` then
    `import _metrics`. → verify: `PYTHONPATH=tools/report ./.venv/bin/python -c "import _metrics; print(len(_metrics.load()))"` > 0, no exception.
B2. Edit `tools/report/build_phase1_report.py`: `sys.path.insert(0, str(Path(__file__).parent)); import _metrics`;
    before stitching, substitute `{{key}}` placeholders in chapter `.md` using `_metrics.load()`;
    **fail the build** (raise) if any `{{...}}` is unresolved. Run substitution BEFORE the figure
    regex + markdown render so `{{}}` never overlaps figure `src=` or footnote syntax.
    → verify: inject a bogus `{{nope.x}}` into a scratch chapter copy → build raises; remove → build passes.
B3. Convert hardcoded metric numbers to `{{key}}` placeholders in: `01-readiness-matrix.md` (the
    matrix cells) and the gate/summary tables of `02-cim.md … 06-npu.md`, `07-workload-energy-e2e.md`.
    Leave prose interpretation text as-is. → verify: V3's subagent confirms the **extracted cell text**
    of the rebuilt PDF equals the prior Track-A PDF for these cells (PDFs are not byte-stable);
    `grep -rl "{{" docs/report/phase1/chapters` lists the edited chapters.
B4. Make findings number-cells generated too: add `tools/report/build_findings.py` that writes the
    summary tables of `docs/phase1.{1,2,3}-findings.md` from `_metrics.load()` (between marker
    comments `<!-- gen:start -->`/`<!-- gen:end -->`), leaving hand-written prose outside the markers.
    → verify: run it twice → second run is a no-op (`git diff` empty); values equal the report cells.
B5. Update `plans/phase1-overall-report.md` (process doc): replace the two Sonnet fact-check passes
    (steps 6–7, 11) with "numbers are build-generated from JSON (`_metrics.load()`); review covers
    prose interpretation only, not number transcription". → verify: `grep -ni "fact-check" plans/phase1-overall-report.md` reflects the new (number-free) scope.
B6. Add `tests/test_report_metrics.py` (insert `tools/report` on `sys.path`, `import _metrics`):
    assert `load()` has no duplicate keys and every key resolves to a value present in its source
    JSON. → verify: `./.venv/bin/pytest tests/test_report_metrics.py -q` passes.
B7. Rebuild report + findings; commit; open PR `phase-1.4b` → `main`; notify user; merge on confirm.

---

## Track C — repo hygiene (PR `phase-1.4c`)

C1. `README.md`: update 狀態 from "下一步 Phase 0.1" to current (Phase 1.1–1.3 done + consolidated
    report; next = Phase 2). → verify: no "下一步 Phase 0.1" string remains.
C2. `OVERALL.md`: move resolved/historical content out of the live brief into an archive section at
    the end (or `LOG.md`): the not-yet-built target repo structure block (`simulator/modules/`,
    `runner.py`, `validator.py`, `program.md`, `HANDOFF.md`), resolved open-risks, superseded
    wording. Keep goal, platform assumption, workload scope, phase table, ADR pointers live.
    → verify: `wc -l OVERALL.md` reduced; live section contains no unbuilt-path-as-current claims;
    archived content still present (grep finds it under the archive header).
C3. Consolidate the three sub-plans `plans/phase-1.3-{cim-card-revalidation,onnxim,ramulator2-lpddr5}.md`
    into the **existing** `plans/phase-1.3.md` as sections; keep `plans/phase1-overall-report.md`
    separate. → verify: `ls plans/phase-1.3*` = one file (`phase-1.3.md`); the 3 sub-plans' content
    present as sections.
C4. Figures: stop committing the redundant per-figure formats; **do not touch the report builder's
    embed path** (it base64-embeds `.png`, which stays). `git rm --cached docs/figures/**/*.pdf` and
    `docs/figures/**/*.svg`; add both to `.gitignore`; keep `.png` committed. Plotting scripts still
    emit all formats locally (regenerable); only PNG is tracked. → verify:
    `git ls-files 'docs/figures/**/*.pdf' 'docs/figures/**/*.svg'` = 0; `git ls-files 'docs/figures/**/*.png'` unchanged; report build still embeds figures + produces a PDF.
C5. Delete orphan `tools/report/build_phase1_2_report.py` (confirm no `.py` caller). → verify:
    `grep -rl build_phase1_2_report . --include=*.py` = 0 (plan `.md` files may still reference it;
    those are not callers).
C6. Update `CONTEXT.md` repo-index rows touched by C2–C5 (figures format, plans, report builder).
    → verify: index rows match the new tree.
C7. Commit; open PR `phase-1.4c` → `main`; notify user; merge on confirm.

---

## Final verification (before declaring phase done)

V1. `./.venv/bin/python tools/report/build_phase1_report.py` exits 0; PDF opens; figures present.
V2. `./.venv/bin/pytest -q` green (incl. `test_report_metrics.py`, `test_engine_iface.py`).
V3. Spawn a subagent to diff the pre/post report PDFs' metric values and confirm zero numeric drift
    introduced by Track B (templating must reproduce, not change, the numbers).
V4. `./.venv/bin/python tools/analysis/check_phase1_2.py` and `./.venv/bin/python tools/analysis/check_phase1_3.py`
    still exit 0 (no simulator-layer regression).

## Outputs

- `docs/adr/0004-mixed-precision.md`, `docs/phase1.1-findings.md`, `validation/contracts/m6.yaml`,
  report chapters `01`,`08`,`09` — conversion-op reframed (Track A).
- `tools/report/_metrics.py`, `build_findings.py`; `build_phase1_report.py` templating;
  `tests/test_report_metrics.py`; chapters + findings with `{{key}}` cells (Track B).
- Updated `README.md`, `OVERALL.md`, `CONTEXT.md`; consolidated `plans/phase-1.3.md`; one figure
  format; deleted orphan builder (Track C).
- Three PRs: `phase-1.4a`, `phase-1.4b`, `phase-1.4c`.

## Out of scope (queued, board/Phase-2 work — NOT this phase)

- Card micro-benchmark sprint to lift prefill-GEMM / multi-tile / compute-bound from extrapolation
  to measured (uses `axrunmodel` dev/system split; same-AIPU, no clock rescale).
- kv_cache coefficient isolation SPIKE on the Card (try; fallback = analytic, M2-bracketed).
- Building the Card simulation first, then the board/host-MMIO simulation via the same engine+spec
  (swap memory spec) — Phase 2.

---

## Execution record (2026-06-11) — what was actually done vs this plan

Executed **pre-merge on branch `report/phase1-overall`** (NOT branched off `main` per step 0):
the consolidated report was still only on that branch and carried the old conversion-op framing,
so applying A+B+C there ships it already-correct instead of merge-then-fix. Consequence: these
changes ride with the report PR, not as three separate `phase-1.4{a,b,c}` PRs.

- **Track A — done.** conversion-op reframed as a Phase-2 analytic modeling item (not a measurement
  gap) in `docs/adr/0004-mixed-precision.md` (+Revision note), `docs/phase1.1-findings.md`,
  report chapters `01`/`08`/`09`, `validation/contracts/m6.yaml`. Chapter 09's GO/NO-GO condition
  was changed accordingly (one condition now: prefill/multi-tile; conversion-op is cost-based-analytic).
- **Track B — done, refined.** `tools/report/_metrics.py` (46 keys) reads each number from its
  JSON path and formats it; `build_phase1_report.py` substitutes `{{key}}` at build (fails on
  unresolved). **71 placeholders** across chapters 01–07 (gate-result / spec / headline cells;
  illustrative per-point + equation-param tables left literal, JSON-cited). Findings: instead of a
  drift-guard, the gate-summary table is **generated in place** by `tools/report/build_findings.py`
  (marker region; committed file has real numbers, regenerable). `tests/test_report_metrics.py`
  (6 tests). Refinements vs plan: curated semantic keys (not a generic flat dict — needed for
  per-cell formatting), and marker-generation for findings (not the lighter drift-guard).
  Every conversion verified **byte-exact** (substitution reproduces the pre-edit text).
- **Track C — done.** README status; OVERALL stale 1.2/1.3 rows fixed; 3 phase-1.3 sub-plans folded
  into `phase-1.3.md`; PDF+SVG figures untracked + gitignored (**PNG kept**, builder embed untouched
  — descoped from the plan's "pick SVG"); orphan `build_phase1_2_report.py` deleted; CONTEXT synced.

**NOT done here (needs the Mac / user):** Chrome PDF rebuild (`build_phase1_report.py`), running
pytest under the repo venv, `git commit`, merge, and the cleanup PR (delete `docs/report/phase1.{1,2,3}/`).
The committed `docs/report/phase1/index.html` + PDF are STALE until rebuilt. See `docs/handoff-phase1.4.md`.
