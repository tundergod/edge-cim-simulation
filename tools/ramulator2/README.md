# Ramulator2 heavy backend (Phase 1.3, `engine='ramulator2'`)

The `MemoryModel(spec, engine='ramulator2')` branch (in `simulator/models/m2_memory.py`) is
**interface-ready**: it reads a cached Ramulator2 LPDDR5 effective-bandwidth result and swaps it in
for the analytic eff_BW, behind the same constructor + frozen `predict()` contract. **Until the
cache exists it falls back to the analytic value** with an honest provenance note (`risk-#6`).

> **Status (2026-06-06): BUILT + DDR4-verified.** `./build.sh` builds Ramulator2 on macOS (Apple
> clang 17, cmake 4.3) after two toolchain patches (captured in `build.sh`): (1) `param.h:91` needs
> the `template` keyword for a dependent template name; (2) cmake 4.x needs
> `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` for the bundled `yaml-cpp`. The binary runs a DDR4 stream e2e
> clean (`ddr4_smoke.yaml`: 20000 req, 248476 cycles). **LPDDR5 single-stream BW is a bounded
> follow-up** — `lpddr5.yaml` aborts with `Failed to send refresh!` under saturation (a
> ramulator2/LPDDR5 refresh-config interaction; DDR4 runs identically, so the build is fine). Until
> resolved, `engine='ramulator2'` falls back to analytic (the plan's position).
>
> **ADR-0002 open item RESOLVED:** `src/dram/impl/` ships DDR3/4/5, GDDR6, HBM/2/3, and **LPDDR5
> only** — no LPDDR4/4x preset (confirmed by building).

## Build

```bash
cd tools/ramulator2 && ./build.sh    # clone + 2 patches + cmake + make (upstream/ gitignored)
```

## Produce the cache the adapter reads

Run a **representative-iteration** sweep (ADR-0002: decode at KV 128/512/1024, per-token steady
state) at the `mem_lpddr5` config (6400 MT/s, the 33.3 GB/s analytic anchor to cross-check), then
write the per-token effective BW to the cache file the adapter loads:

```
simulated/ramulator2/lpddr5_eff.json   = {"eff_BW_GBs": <float>, "per_kv": {"128": ..., "512": ..., "1024": ...},
                                          "honesty": "simulated (Ramulator2 LPDDR5), NOT silicon"}
```

A `tools/analysis/mem_ramulator2.py` driver (subprocess + per-shape cache) writes that file; then
`tools/analysis/build_mem_ramulator2.py` computes the per-shape **Ramulator2-vs-1.2-analytic delta**
→ `validation/reports/phase1.3/m2_ramulator2.json` + figure `M2-ramulator2`. The single-stream
delta is the Phase-1.3 value; the **signature multi-unit contention is Phase 2** (ADR-0002).
