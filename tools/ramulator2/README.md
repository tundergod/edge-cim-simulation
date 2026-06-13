# Ramulator2 heavy backend (Phase 1.3, `engine='ramulator2'`)

`MemoryModel(spec, engine='ramulator2')` (in `simulator/models/m2_memory.py`) reads the cached
Ramulator2 LPDDR5 effective-bandwidth result (`simulator/engines/ramulator2/lpddr5_eff.json`) and swaps it
in for the analytic eff_BW, behind the same constructor + frozen `predict()` contract. If the cache
is absent it falls back to analytic with an honest note.

> **Status (2026-06-06): LIVE.** Built on Ramulator **2.1** (the main-branch LPDDR5 `Failed to send
> refresh!` bug is fixed in v2.1 — issues #58/#60/#89). LPDDR5_6400 saturated streaming runs clean
> (**98.6% of peak** — saturation proof) and `engine='ramulator2'` now uses the Ramulator2 BW.
> **Result:** Ramulator2 (DRAM-device) single-stream efficiency **0.92** (47.1 GB/s) vs the analytic
> **system-level 0.65** (33.3 GB/s) — not a contradiction: the gap is the controller/NoC/queueing
> overhead the analytic captures (silicon-calibrated) and the device model omits. This **validates
> ADR-0002** (system efficiency calibrated from silicon, not imported). Analytic 33.3 stays primary.
>
> **ADR-0002 open item RESOLVED:** Ramulator2 ships **LPDDR5 only** — no LPDDR4/4x preset.

## Build (Ramulator 2.1 — Python-bindings-only, no CLI/YAML)

```bash
cd tools/ramulator2 && ./build.sh    # pins v2.1 SHA 278f1ef; cmake -DPython_EXECUTABLE=.venv + make
```

`build.sh` checks out the pinned v2.1 commit, applies one toolchain patch (Apple-clang-17
`template`-keyword in `base/param.h`, grep-guarded), and builds the `ramulator` Python extension
under `python/ramulator/` (upstream/ gitignored). Needs CMake≥3.14 + Python≥3.10 (with dev headers).

## Produce / refresh the cache

```bash
.venv/bin/python tools/analysis/mem_ramulator2.py        # -> simulator/engines/ramulator2/lpddr5_eff.json
.venv/bin/python tools/analysis/build_mem_ramulator2.py  # -> validation/reports/phase1.3/m2_ramulator2.json
.venv/bin/python tools/plotting/mem_ramulator2_fig.py    # -> docs/figures/phase1.3/M2-ramulator2.*
```

`mem_ramulator2.py` drives v2.1's own `latency_throughput` harness **in-process** (`run_simulation` /
`resolve_spec` / `checks.py` — no CLI, no hand-rolled tCK; v2.1 computes `total_throughput_MBps`
itself). It measures peak (refresh off) + achievable (AllBank) saturated streaming and writes the
efficiency-scaled eff_BW. Cache schema: `{eff_BW_GBs, efficiency, peak_efficiency_saturation,
channel_width, bytes_per_req, rate, peak_GBs_single_channel, refresh_mode, v2_1_commit, honesty}`.
The single-stream delta is the Phase-1.3 value; the **signature multi-unit contention is Phase 2**
(ADR-0002).
