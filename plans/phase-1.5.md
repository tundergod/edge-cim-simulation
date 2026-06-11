# Plan: Phase 1.5 — CIM compute 補量（prefill + multi-tile + KV-cache SPIKE）

Branch `phase-1.5-cim-compute-supplement` off `main`. Full plan + context + validation strategy: `/Users/tundergod/.claude/plans/1-decode-cim-splendid-yeti.md` (opus-approved). Board run is driven end-to-end over `ssh metiscard` (needs a Bash permission rule for remote-shell writes).

## Measurement axes (grids)
- **A** prefill M-sweep densify (canonical 2048×2048 `gate_up` tile): fit-densify M∈{16,32,48,64,96,128,192,224,256}; no-bridge band M∈{2,4,8,16,32}; wall-pin M∈{248,252,254,255,256,257,258,260,264,272,288,320} (tolerate compile-limit error).
- **B** more tile shapes @ M∈{64,128,256}: down_proj K=14336/N=4096, lm_head K=4096/N=128256, small-K K=2048/N=14336, large-K K=14336/N=14336, control gate_up K=4096/N=14336.
- **C** M-axis chunked tiling (M>256): M_eff∈{512,1024,2048} = ⌈M/256⌉ back-to-back M=256 runs; record total/sum/additive_residual_pct/per_chunk_overhead_us; label `m_tiled_chunked`.
- **D.1** envelope probe (M=1, native `measure()`, vary one axis): (K,N)∈{(2048,2304),(2048,2560),(2048,3072),(2048,4096),(2304,2048),(2560,2048),(3072,2048),(4096,2048),(2560,2560),(3072,3072),(2048,8192),(8192,2048)}; record `compiles_native`+gating axis.
- **D.2** native multi-tile (whatever D.1 admits): decode M=1 (K,N)∈{(1024,4096),(2048,3072),(3072,2048)}; prefill M∈{64,128} K=2048/N=3072. Compare native dev_lat vs tile-sum. Fallback `NATIVE_MULTITILE_UNCOMPILABLE` keeps Alpha +36%.
- **E** KV-cache SPIKE: memory-bound proxy (thin conv K=1 / elementwise) streaming kv_bytes at 8B kv∈{128,512,1024,2048}; derive eff_BW; `compute_negligible` control. Isolable→calibrate kv coeff; else `ANALYTIC_RETAINED`.

## Steps (action → verify)
1. Branch off main → verify: branch current.
2. Write this plan → verify: file exists.
3. Extend `characterization/metis_card/run_metis_cim_v16.py`: add families `prefill_msweep`(A), `prefill_shapes`(B), `mtile`+`measure_m_chunks()`(C), `envelope_probe`(D.1), `multitile`(D.2), `kv_proxy`(E); wall-pin + probe tasks fail-tolerant; update docstring/wall comments (M_MAX/SAFE_KN now probed) → verify: `python -c "import ast;ast.parse(open(...).read())"` ok; helper imports; grids present.
4. Run on Card: rsync → `--spike` → scoped `--only <group>` → full → rsync results back → verify: `cim_card_revalidate_raw.json` fresh, all 6 families present.
5. Inspect `envelope_probe` FIRST; record probed wall in this plan's notes; keep uncompilable shapes flagged → verify: each probe has `compiles_native`; wall noted.
6. `fit_m1_cim.py` (preserves prefill keys) → verify: exit 0, PASS, keys intact.
7. D-wiring: ingest Card native-multitile → `validation/reports/phase1.5/cim_multitile.json` (`card_native_multitile` block or `NATIVE_MULTITILE_UNCOMPILABLE`) + held-out meas-vs-pred → verify: block present.
8. `fit_cim_prefill.py` (hard gate max_rel_err≤0.05 stays) + `m_axis_tiling` block (C) + held-out split (A); break affinity→narrow `prefill_M_fit`, document curvature → verify: exit 0; `prefill_M_fit`/`prefill_M_max` reflect probed grid/wall; `m_axis_tiling` + `holdout_meas_vs_pred` present.
9. Axis E: isolable→extend `fit_m2.py` to ADD kv coeff field + `validation/reports/phase1.5/kv_append_spike.json`; else status `ANALYTIC_RETAINED`+reason. Must not perturb `BW_eff` → verify: spike report has decision; `recompose.*` unchanged.
10. `validate_cim_card.py` → verify: exit 0; `CARD_REVALIDATED` (n≥8); new shapes present.
11. `_metrics.py`: existing cim.* auto-update; add `cim.prefill_M_max`, `cim.m_tile_residual_pct`, `cim.native_multitile_overpred_pct` (Alpha fallback), `kv.*` if landed → verify: prints all keys non-empty.
12. Update hardcodes in `tests/test_report_metrics.py` to new values (recompose.* must NOT move) → verify: `pytest tests/test_report_metrics.py` green.
13. `build_findings.py` → verify: in-sync.
14. Update readiness labels: `01-readiness-matrix.md` (CIM rows) + `02-cim.md` (prefill M>256 / multi-tile +36% / readiness table) + `03-memory.md` (kv, if E landed) → replace static "M>256 extrapolated" with probed wall + Axis-C result; "+36%" with Card-native (or honest uncompilable) → verify: no orphan `{{...}}`.
15. `build_phase1_report.py` (Chrome), last → verify: PDF fresh, no unresolved placeholder.
16. `check_phase1_2.py` + `check_phase1_3.py` exit 0; pytest green → verify: all exit 0.
17. Subagent code-review → address → `gh pr create` phase-1.5→main → notify user → merge on explicit confirm → verify: PR open, checks green.

Outputs: `characterization/metis_card/run_metis_cim_v16.py`; refreshed `measurements/metis_card/cim_card_revalidate_raw.json`; `simulator/models/params/m1_cim.json`; `validation/reports/phase1.1/m1.json`, `phase1.2/{cim_prefill_fit,cim_card_revalidate}.json`, new `phase1.5/{cim_multitile,kv_append_spike}.json`; `tools/report/_metrics.py`, `tests/test_report_metrics.py`, `docs/phase1.1-findings.md`, `docs/report/phase1/chapters/{01,02,03}-*.md`, `phase1-report.pdf`; `plans/phase-1.5.md`.
```
```
## Execution record (campaign run 2026-06-12, 85 tasks, 0 errors)

**Two prior assumptions overturned by the probes:**
- **M_MAX=256 wall was ~2x too LOW (not absent).** prefill_msweep compiled the canonical tile cleanly up to **M=508**; **M=511/512 fail** (no_model_json, consistent through 4096). Real SRAM wall ~M=510, ~2x the assumed 256. prefill calibrated to M<=508 (M>508 extrapolated).
- **K=2048 is NOT a K (contraction) limit.** K-staircase (N=512, sweep K, M=1) compiles natively to >=16384 (no tiling; crossbar accumulates K internally), throughput rising K=2048->106, K=12288->255 GOP/s, then the SAME K*N cliff at K=16384/N=512 (8.4M -> 72 GOP/s). 2048 is the OUTPUT width W; only N>2048 is tiled.
- **SAFE_KN=2048×2048 native wall is FALSE.** envelope_probe natively compiled multi-tile GEMMs up to **K·N=16.78M**.

**Axis A — prefill M-amortization:** dense sweep M∈{2..320} → affine `tile_lat = 40.27 + 0.0991·M` µs, residual median 0.66% / max 3.56%. Bridges to the M=1 decode anchor (41.83µs) within 3.5% — the old "no-bridge 2.5× gap" was a sparse-fit artifact; M-amortization is continuous M=1→320.

**Axis D — native multi-tile RESIDENCY CLIFF (the headline).** M=1 native throughput rises smoothly to ~264 GOP/s up to a knee at **K·N ≈ 8.0M** (resident, weights in SRAM), then **collapses ~3.5× to a ~69.7 GOP/s floor** (DRAM spill). Knee sharply bracketed: resident max 7.93M ↔ spill min 8.39M. New 2-regime model — resident `lat=14.58+5.84·(K·N/1e6)` µs, spill `lat=2·K·N/(69.7·1e9)` — **held-out median ~3% (unique-K·N split, incl. K/N staircases)**; vs all native pts it beats the old tile-sum **31%/100% → 2.8%/12.5%**. The old model was +31% over (sub-knee) and −65% under (supra-knee, the cliff it lacks). Matters for Phase 2: real decode FFN/lm_head GEMVs (K·N≥16M) live in the spill regime.

**Axis C — M-tiling:** chunked serving is additive (total = n×chunk); per-chunk host/DMA overhead ~1989µs for the 14-tile gate_up GEMM. NB premise weakened — native M compiles to 508, so chunking is only for M>508 (the real wall ~510).

**Axis E — KV SPIKE = PROXY_INCONCLUSIVE (honest negative).** The K=1 memory-bound proxy eff_BW RISES with transfer size and never converges (9.6/17.0/26.7/35.9/44.4 GB/s for M=64..1024; M>=2048 fail). Every compilable point's working set (output N·M ≤ 2.1M elems) is BELOW the ~8M SRAM knee → the proxy is SRAM-staging-bound, never reaches the DRAM regime, so it CANNOT isolate the kv_append DRAM BW. The earlier 'M=256 ≈ M2' was coincidental. kv_append stays analytic on M2's measured DRAM streaming BW (24.2); the only DRAM-bound on-card datapoint is the cliff spill floor (~34.8 GB/s, same order). DRAM-BW independent validation → Phase 2.
