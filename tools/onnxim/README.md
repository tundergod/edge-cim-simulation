# ONNXim heavy backend (Phase 1.3, `engine='onnxim'`)

The `NpuModel(spec, engine='onnxim')` branch (in `simulator/models/m4_npu.py`) is **interface-ready**:
it reads a cached ONNXim (generic-systolic, RKNPU2-approx) per-shape latency table and uses it in
place of the analytic systolic-roofline, behind the same constructor + frozen `predict()` contract.
**Until the cache exists it falls back to the analytic value** with an honest provenance note
(`risk-#7`). ONNXim is a heavier **simulator**, NOT RKNPU2 silicon (≠ issue #13, which stays
*superseded-not-satisfied*).

> **Status (2026-06-06): cloned; build NOT completed (toolchain friction).** ONNXim pins
> **`conan == 1.57.0`** (conan 1.x), which is incompatible with the modern `conan 2.x` that installs
> on this Python 3.13; the upstream's recommended path is its **Dockerfile**. Resolving the conan-1.x
> / Docker build is a bounded follow-up — and ONNXim is the **lower-value** heavy sim (it is
> *simulator-vs-simulator* against the already-`simulated` analytic NPU, neither being silicon;
> contrast Ramulator2-vs-analytic, where Ramulator2 is a validated DRAM model). Until built,
> `engine='onnxim'` falls back to analytic (risk #7).

## Build (bounded follow-up — needs conan 1.57.0 or Docker)

```bash
cd tools/onnxim
git clone --depth 1 https://github.com/PSAL-POSTECH/ONNXim upstream      # gitignored (done)
# Option A (Docker, upstream-recommended):
cd upstream && docker build -t onnxim .
# Option B (conan 1.x in an isolated env — NOT conan 2.x):
python3 -m venv /tmp/onnxim-conan && /tmp/onnxim-conan/bin/pip install 'conan==1.57.0'
#   then follow upstream README (conan install + cmake >= 3.22.1)
```

Configure as **RKNPU2-approx** by reading `simulator/specs/npu_rknpu2.json` (systolic dim borrowed
Hexagon 32×32, 6 TOPS INT8). Build failure → OVERALL risk #7 fallback (analytic-only / lookup-
override) + report user.

## Produce the cache the adapter reads

Export ONNX for the NPU shapes in `measurements/op_inventory/` (ADR-0007: export is secondary,
fallback = build the ONNXim input from the traced graph), run ONNXim, and write per-shape latency:

```
simulated/onnxim/rknpu2_sim_matmul.json = {"rows": [{"shape": [M, K, N], "latency_us": <float>}, ...],
                                           "honesty": "simulated (generic-systolic, RKNPU2-approx), NOT silicon"}
```

A `tools/analysis/npu_onnxim_trace.py` driver writes that file; then `tools/analysis/
build_m4_npu_onnxim.py` computes the per-shape **ONNXim-vs-1.2-analytic delta** + HeteroInfer trend
→ `validation/reports/phase1.3/m4_npu_onnxim.json` + figure `N3`. Output lives under `simulated/`
(NOT `measurements/`) so simulated data is never mixed with silicon.
