# Ramulator2 heavy backend (Phase 1.3, `engine='ramulator2'`)

The `MemoryModel(spec, engine='ramulator2')` branch (in `simulator/models/m2_memory.py`) is
**interface-ready**: it reads a cached Ramulator2 LPDDR5 effective-bandwidth result and swaps it in
for the analytic eff_BW, behind the same constructor + frozen `predict()` contract. **Until the
cache exists it falls back to the analytic value** with an honest provenance note (`risk-#6`).

> **Status: build deferred.** This session could not build Ramulator2 — the harness did not authorize
> cloning/building external code (user offline). The adapter + this runbook are ready; building is
> one authorization away.

## Build (after authorizing external builds)

```bash
cd tools/ramulator2
git clone --depth 1 https://github.com/CMU-SAFARI/ramulator2 upstream   # gitignored
cd upstream && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4                          # C++20 (Apple clang 17 OK)
# confirm LPDDR4/4x presence (ADR-0002 open item — Ramulator2 ships LPDDR5):
ls ../src/dram/impl/   # expect LPDDR5; LPDDR4/4x = assumption until seen here
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
