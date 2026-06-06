# ONNXim heavy backend (Phase 1.3, `engine='onnxim'`)

`NpuModel(spec, engine='onnxim')` (in `simulator/models/m4_npu.py`) reads the cached ONNXim
(generic-systolic, RKNPU2-approx) per-shape latency table (`simulated/onnxim/rknpu2_sim_matmul.json`)
and uses it in place of the analytic systolic-roofline, behind the same constructor + frozen
`predict()` contract. If the cache is absent it falls back to analytic. ONNXim is a heavier
**simulator**, NOT RKNPU2 silicon (≠ issue #13, which stays *superseded-not-satisfied*).

> **Status (2026-06-06): LIVE.** ONNXim is Ubuntu-20.04/gcc-10/conan-1.57 only (won't build on macOS;
> conan 1.57 won't install on Py3.13). Built on **metiscard** (x86 Ubuntu + Docker) via its own
> `ubuntu:20.04` Dockerfile (pinned commit `a1e86296`, image `onnxim`), configured RKNPU2-approx
> (`rknpu2_approx.json`: 3×32×32, INT8, 6.14 TOPS, ramulator2-DDR4 25 GB/s). **Result:** ONNXim
> (cycle-level) vs the analytic roofline — the channel **staircase trend agrees** (monotone, ∝N), but
> ONNXim sits a *consistent* **~4×** above the analytic (median |delta| 318%): ONNXim models
> systolic-fill/NoC/DRAM-scheduling overhead the roofline abstracts. Both simulated, neither silicon →
> the trend agreement is the cross-check; analytic stays primary; ONNXim ≠ #13. See chapter
> `N-npu-onnxim.md`, report `validation/reports/phase1.3/m4_npu_onnxim.json`, figure `N3`.

## Build (Docker on a Linux host — `metiscard`)

```bash
ssh metiscard 'cd ~/edge-cim-simulation && git clone https://github.com/PSAL-POSTECH/ONNXim onnxim \
  && cd onnxim && git checkout a1e86296 && git submodule update --recursive --init \
  && docker build . -t onnxim'        # ubuntu:20.04 base installs gcc-10, conan 1.57, cmake 3.22, torch
```

CLI is `./build/bin/Simulator --config <hw> --models_list <list>` (NOT `--model`; the README is
stale). Run INSIDE the image tree (`cd $ONNXIM_HOME=/workspace/ONNXim`) — do NOT bind-mount over
`/workspace` (it shadows the in-image build). Two gotchas baked into `rknpu2_approx.json`: the
8×8-template `spad_size:64` is too small for a 32×32 array (div-by-zero in `Mapping.cc` tiling →
scaled to 2048/512); and ONNXim **SIGFPE-crashes on GEMMs with N≤64** (degenerate tiling) — the
sweep uses N≥128. `ramulator2` DRAM works (no issue-#32 crash); `dram_type:simple` is the fallback.

## Produce / refresh the cache

```bash
.venv/bin/python tools/analysis/npu_onnxim_trace.py        # one docker run on metiscard -> simulated/onnxim/rknpu2_sim_matmul.json
.venv/bin/python tools/analysis/build_m4_npu_onnxim.py     # -> validation/reports/phase1.3/m4_npu_onnxim.json (per-shape delta; asserts each is an ONNXim hit)
.venv/bin/python tools/plotting/npu_n3_fig.py              # -> docs/figures/phase1.3/N3.*
```

The canonical (M,K,N) list lives in `npu_onnxim_trace.py` and is the single source of truth (the
delta report reads back exactly those shapes — N4). Output is under `simulated/` (NOT
`measurements/`) so simulated data is never mixed with silicon.
