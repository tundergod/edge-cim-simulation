# Plan: Phase 2.4 — gate/test hardening (post-2.3 audit fixes)

Branch `phase-2.4` off `main` (15efd35, PR#64 merged). Execute D before E (D changes `sensitivity.json`,
which becomes the byte-identical baseline for E-18). No committed number changes; AllCim L4 stays
0.1073/0.0649/0.0311.

## A. Wire exit codes on gates that currently exit 0 regardless of pass/fail

1. `validation/validate_m5_trace.py` — `main()` returns `0 if all_ok else 1`; call `sys.exit(main())`
   → verify: step-A5 test asserts `main() != 0` when a check is forced false; normal run exits 0.
2. `validation/validate_m7_energy.py` — `main()` returns non-zero when any `bool_checks` entry is false;
   `sys.exit(main())` → verify: step-A5 test asserts `main() != 0` under a flipped coefficient.
3. `validation/validate_contention.py` — `main()` returns `0 if (rising_then_flat and off_linear) else 1`
   (replaces unconditional `return 0`); keep the "SIMULATED, no numeric gate" prose (shape-only)
   → verify: step-A5 test asserts `main() != 0` when saturation is broken.
4. Grep for tests asserting these three exit 0 today; none should break (they already exit 0 on the
   normal path) → verify: `pytest tests/` unchanged count before code edits below.
5. New `tests/test_gate_exit_codes.py` — for each of the three `main()`s, monkeypatch one check false and
   assert `main() != 0` → verify: `pytest tests/test_gate_exit_codes.py` green.

## B. Carry-forward PR#64 nits

6. `validation/contracts/m7.yaml:10` — rename `lpddr5_pj_per_bit` → `dram_pj_per_bit`
   → verify: `grep -rn lpddr5_pj_per_bit .` returns nothing outside `.venv`.
7. `simulator/runtime/workload.py` `build_token_dag` — after the `struct` loop, raise ValueError if
   `pending_qk is not None` (global attention-bmm parity guard; the flat DAG carries no layer boundaries,
   so this is global not per-layer) → verify: new `tests/test_workload.py` case monkeypatches the
   structure source (`_load_structure`/`_STRUCT_CACHE`) to an odd attention-bmm count and asserts it raises.

## C. Tests for existing fail-loud branches

8. New `tests/test_m2_memory.py` — convert the two `assert`s in `MemoryModel.predict` (~lines 86, 94) to
   explicit `raise ValueError` (grep for any test expecting `AssertionError` from these first); cover:
   `op='pcie'` without a topology raises; `stream` without `eff_BW_GBs` raises; all 7 `_bw_tag()` branches
   (ramulator2 / -na / -deferred / edge / LPDDR4x / LPDDR5 / default) return the expected tag
   → verify: `pytest tests/test_m2_memory.py` green.
9. `tests/test_scheduler.py` — a DAG node with an unmapped category makes BOTH `AllCimScheduler` and
   `CimHeteroScheduler` raise ValueError (scheduler.py:61 / :91) → verify: pytest green.
10. `tests/test_runner_e2e.py` — `platform.engine={"cim":"ramulator2"}` through `run()` raises
    NotImplementedError (runner.py:64-68) → verify: pytest green.
11. New `tests/test_m4_cpu.py` — cover `op_us()` default (non-`model`-extra) path; the dtype ValueError
    (m4_cpu.py:135); the L3 cache-tier branch of `_tier_bw` → verify: pytest green.
12. Anchor–spec desync guards: `validate_sensitivity_l5.py` add `assert ANCHOR == load_spec("mem_lpddr4x")
    ["eff_BW_GBs"]`; `validate_contention.py` add `from simulator.specs.loader import load_spec` and
    `assert EFF_BW == load_spec("mem_lpddr4x")["eff_BW_GBs"]` (its constant is `EFF_BW`, not `ANCHOR`)
    → verify: both validators exit 0.
13. New `tests/test_unit_smoke.py` — instantiate `SramTier` (m1_cim_spm), `GpuRooflineModel`
    (m4_gpu_roofline), `NpuModel` (m4_npu) and call one representative method each, asserting a finite
    positive latency → verify: pytest green.

## D. Honesty-framing corrections — docs/report only, no number changes (run before E)

14. `docs/adr/0006-validation-bridging-extrapolation.md` (~line 16) — reword: 24.2 GB/s is the effective
    decode bandwidth extracted from the decode size-sweep (PARTLY IN-SAMPLE; inverse slope of
    decode-time∝weight-bytes, r²=0.997), NOT an independent DRAM streaming benchmark; keep the still-valid
    24.2 (weight-bytes basis) vs fit_BW≈18.33 (no-intercept basis) distinction; remove only the
    independence implication → verify: text present; `grep` finds no remaining "measured memory rate"
    independence claim; the `[measured]` provenance tag is untouched.
15. `tools/report/_metrics.py` (~line 249) — refine the `"the hard silicon gate"` comment to
    "silicon-gated ≤15%; partly in-sample on the 24.2 backbone (out-of-sample content = CIM correction)"
    (refine, do not erase the silicon-gated fact); reword the sensitivity `conclusion` VALUE in
    `validate_sensitivity_l5.py` to "internal-consistency band" and regenerate `sensitivity.json`; do NOT
    rename `conclusion_robust` (read by `_metrics.py:271` → built HTML) → verify: `docs/report/phase1-site/
    build.py --strict` green (14 pages, `conclusion_robust` still resolves); the two comment/value edits
    are Python-comment / JSON-value only (built HTML unchanged).
16. `simulator/runtime/events.py` — add a `# WATCH:` comment at BOTH DRAM-metering sites (~line 62 in
    `run_serial`, ~line 135 in `run_dag`) pointing to the double-count decision; open a GitHub issue
    capturing audit finding P1-2 (24.2 fit on weight bytes vs KV/attn/embed metered on the same pool; no
    model change here) → verify: issue created; both comments reference it.

## E. Zero-risk de-duplication (after D; baseline = post-D snapshot)

17. Site-figure style: create `tools/plotting/_site_style.py` (or an `apply_site_style()` fn) holding the
    byte-identical site rcParams block + color constants (`INK`/`GRID`/`HERO`/…) + `load()` + `_grid()`
    from the 8 `tools/plotting/site_*.py`; import from there. LEAVE `_style.py`'s existing module-level
    rcParams (font.size 7, Arial — the Phase 0.2/0.3 style) UNTOUCHED → verify: regenerate the site
    figures; committed PNGs visually unchanged; `build.py --strict` green.
18. DROPPED (executed decision, karpathy simplicity): factoring the `ROOT`/`OUT`/`MODELS` boilerplate into
    `validation/_common.py` was rejected — each validator must bootstrap `sys.path` with its own `ROOT`
    line BEFORE it could import that shared module, so the dedup adds import indirection worse than the
    2–3 trivial duplicated lines it removes. The `_cfg()` wrappers were already out (signatures differ).
    Net: no `validation/_common.py`; the validators keep their inline bootstrap.

## Final gate (Phase 2.4)

- `.venv/bin/pytest tests/` all green (existing + new).
- All 8 validators exit 0; the 3 newly-wired gates (steps 1-3) return non-zero under a forced-false probe.
- `docs/report/phase1-site/build.py --strict` green (14 pages).
- `validation/reports/phase2/e2e_l4.json` unchanged (0.1073/0.0649/0.0311).
- `phase2/*.json` byte-identical to the post-D snapshot (only `sensitivity.json` changes vs main, at
  step 15; step 18 dropped so no validator refactor touches the JSONs).

Outputs: edited `validate_m5_trace.py`, `validate_m7_energy.py`, `validate_contention.py`,
`validate_sensitivity_l5.py`, `m7.yaml`, `workload.py`, `m2_memory.py`, ADR-0006, `_metrics.py`,
`events.py`; new `tests/test_gate_exit_codes.py`, `test_m2_memory.py`, `test_m4_cpu.py`,
`test_unit_smoke.py` + additions to `test_scheduler.py`/`test_runner_e2e.py`/`test_workload.py`; new
`tools/plotting/_site_style.py` + refactored `site_*.py`; `.gitignore` (.coverage); GitHub issue #67.

## Out of scope
- Compute-engine generality (wire `cim_compute_params`, non-Metis provenance guard) → Phase 2.5.
- Removing dead knobs (`concurrency_overlap_factor`, `cfg.engine.npu`, `units.npu`).
- Re-deriving the 10%/20%/15% tolerances from measured variance.
- Resolving (not documenting) the double-count.
