"""M4 — RKNPU2 (RK3588 NPU) timing model — ANALYTIC systolic-roofline (Phase 1.2, D2).

NO RKNPU2 silicon (board offline, issue #13) -> EVERY number here is `simulated`/`borrowed`,
NOT calibrated. This is an analytic systolic-array roofline whose *shape* is borrowed from the
HeteroInfer characterization (SOSP'25, papers/methodology-and-simulators/) and whose ceilings come
from the RKNPU2 datasheet (6 TOPS INT8). It exists so the heterogeneous simulator has a swappable
NPU slot; it is replaced, not validated, by silicon (#13) or ONNXim (Phase 1.3).

Model (GEMM M x K x N, INT8):
  - alignment padding: each dim is padded UP to the systolic dimension (borrowed Hexagon 32x32,
    HeteroInfer §3.2 Fig3) -> padded MACs = ceil(M/sd)*sd * ceil(K/sd)*sd * ceil(N/sd)*sd. This
    is the Fig3 STAIRCASE: latency steps every time a dim crosses a multiple of 32 (small/odd
    shapes waste a whole pad row/col). The knee sits at the 32x32 boundary.
  - compute ceiling: 6 TOPS INT8 datasheet peak -> compute_us = 2*padded_MACs/(tops*1e12)*1e6,
    scaled by an order/shape FACTOR (HeteroInfer §3.2 Fig4): reversing the Matmul dim order, or a
    wide-activation/narrow-weight shape, breaks the weight-stall paradigm and costs up to 6x. We
    model that as a multiplier in [1, 6] driven by the N:K aspect (capped at the borrowed 6x).
  - memory ceiling: bytes / eff_BW, with eff_BW from the spec's HeteroInfer Fig5 band (40-45 of a
    68 GB/s peak = 59-66%); the active weights/activations must stream in.
  - dispatch floor: a small fixed per-op cost so tiny ops are not free.
  bound = argmax(compute, memory, floor).

Native attention (op='attn_bmm'): activation x activation, no static weight to stall on -> pure
compute-bound at the padded TOPS ceiling (no order/shape penalty, no weight stream). Per (kv,heads,
layers): QK^T (M=heads*1, K=hd, N=kv) + S.V (M=heads*1, K=kv, N=hd), padded to 32.

dtypes: INT4/8/16 + FP16 only (datasheet). No RKNPU2 power telemetry -> energy NOT determinable.
"""
import json
import math
from pathlib import Path

from simulator.models.engine import UnitEngine, Workload

# borrowed HeteroInfer constants (SOSP'25 §3.2/§3.3) — all `borrowed`, none measured here.
_ORDER_SHAPE_MAX = 6.0      # Fig4: up to 6x order/shape penalty
_DISPATCH_FLOOR_US = 2.0    # simulated fixed per-op dispatch (no silicon -> nominal, assumption)
_DTYPE_BYTES = {"int4": 0.5, "int8": 1.0, "int16": 2.0, "fp16": 2.0}
_ONNXIM = Path(__file__).resolve().parents[2] / "simulator/engines/onnxim/rknpu2_sim_matmul.json"
_SCALESIM = Path(__file__).resolve().parents[2] / "simulator/engines/scalesim/rknpu2_sim_matmul.json"


def _onnxim_table():
    """Phase 1.3 heavy engine: cached ONNXim (generic-systolic RKNPU2-approx) per-shape latency
    table (produced by tools/onnxim/build.sh + tools/analysis/npu_onnxim_trace.py). Returns {}
    when absent (the C++ build is deferred) -> documented analytic fallback."""
    if not _ONNXIM.exists():
        return {}
    return {tuple(r["shape"]): r["latency_us"] for r in json.loads(_ONNXIM.read_text())["rows"]}


def _scalesim_table():
    """Phase 1.6 third engine: cached SCALE-Sim v2 (generic 32x32-WS systolic, RKNPU2-approx,
    3-core /cores aggregation) per-shape latency table (tools/scalesim/run_rknpu2_scalesim.py).
    Native systolic behaviour emergent, NOT tuned; SIMULATED, NOT silicon (#13). {} when absent
    -> documented analytic fallback (incl. the giant shapes skipped as cycle-sim-intractable)."""
    if not _SCALESIM.exists():
        return {}
    return {tuple(r["shape"]): r["latency_us"] for r in json.loads(_SCALESIM.read_text())["rows"]}


class NpuModel(UnitEngine):
    """Analytic RKNPU2 systolic-roofline. All outputs `simulated`/`borrowed` (no silicon, #13)."""

    def __init__(self, spec, engine="analytic"):
        super().__init__(spec, engine)
        sd = spec["systolic_dim"]
        self.sd = int(sd[0])                       # borrowed 32x32 -> single alignment quantum
        self.tops = float(spec["tops_int8"])       # 6 TOPS INT8 datasheet ceiling
        bw = spec["bw_GBs"]
        # eff BW band (Fig5 59-66% of 68 peak); use the low end as the roofline ceiling (pessimistic).
        self.bw_eff_low = float(bw["eff_low"])     # ~20.1 GB/s
        self.bw_eff_high = float(bw["eff_high"])   # ~22.4 GB/s
        self.dtypes = set(spec["dtypes"])          # restricted to INT4/8/16/FP16 (datasheet)
        # Phase 1.3 heavy backend: engine='onnxim' replaces the analytic latency with ONNXim's
        # per-shape sim (same constructor + frozen contract). Empty table -> analytic fallback.
        self.onnxim = _onnxim_table() if engine == "onnxim" else {}
        # Phase 1.6 third engine: engine='scalesim' (SCALE-Sim v2, same drop-in contract).
        self.scalesim = _scalesim_table() if engine == "scalesim" else {}

    def _pad(self, x):
        """Pad a dim UP to the borrowed systolic quantum (Fig3 staircase source)."""
        return self.sd * max(1, math.ceil(x / self.sd))

    def _order_shape_factor(self, K, N):
        """Fig4 order/shape penalty in [1, 6]: a wide activation relative to the weight (large N:K)
        breaks the weight-stall paradigm. Linear in log2(N/K), saturated at the borrowed 6x."""
        if K <= 0 or N <= 0:
            return 1.0
        ratio = N / K
        if ratio <= 1.0:
            return 1.0
        f = 1.0 + (_ORDER_SHAPE_MAX - 1.0) * min(1.0, math.log2(ratio) / math.log2(8.0))
        return min(_ORDER_SHAPE_MAX, f)

    def _gemm_us(self, M, K, N, dtype):
        """Padded-MAC systolic roofline for one GEMM. Returns (latency_us, bound)."""
        Mp, Kp, Np = self._pad(M), self._pad(K), self._pad(N)
        padded_macs = Mp * Kp * Np
        osf = self._order_shape_factor(K, N)
        compute_us = 2.0 * padded_macs / (self.tops * 1e12) * 1e6 * osf
        # weights stream in at the eff-BW ceiling (low end of the Fig5 band = pessimistic roofline).
        wbytes = K * N * _DTYPE_BYTES.get(dtype, 1.0)
        memory_us = wbytes / (self.bw_eff_low * 1e9) * 1e6
        return self._argmax_bound(compute_us, memory_us)

    def _attn_us(self, kv, heads, layers, hd, dtype):
        """Native attention QK^T + S.V, compute-bound (act x act, no weight stall). Heads are the
        BATCH dimension -- NOT padded into the systolic M slot (padding GQA heads=8 up to sd=32 would
        4x-over-count the attention MACs); only the matmul dims hd/kv are padded to the 32x32 quantum,
        then multiplied by the raw head count. The dispatch floor applies PER LAYER, not once for the
        whole L-layer rollup."""
        # per head per layer: QK^T (hd x kv) + S.V (kv x hd), padded; x heads (batch).
        per_head = self._pad(hd) * self._pad(kv) + self._pad(kv) * self._pad(hd)
        per_layer_us = 2.0 * heads * per_head / (self.tops * 1e12) * 1e6
        lat = layers * max(per_layer_us, _DISPATCH_FLOOR_US)
        return lat, ("compute" if per_layer_us >= _DISPATCH_FLOOR_US else "floor")

    def _argmax_bound(self, compute_us, memory_us):
        floor = _DISPATCH_FLOOR_US
        lat = max(compute_us, memory_us, floor)
        bound = "compute" if lat == compute_us else ("memory" if lat == memory_us else "floor")
        return lat, bound

    def predict(self, wl: Workload) -> dict:
        """Frozen dict {latency_us, bound, provenance}. ALL outputs simulated/borrowed (no silicon)."""
        if wl.op in ("attn_bmm", "attention"):
            hd = wl.extra.get("hd", wl.K or 128)
            lat, bound = self._attn_us(wl.kv, wl.heads, wl.layers, hd, wl.dtype)
            prov = ("simulated: analytic systolic attn (act x act, padded to borrowed %dx%d, "
                    "compute-bound; HeteroInfer Fig3); NO silicon (#13)" % (self.sd, self.sd))
        else:
            lat, bound = self._gemm_us(wl.M, wl.K, wl.N, wl.dtype)
            prov = ("simulated: analytic systolic roofline %d TOPS INT8 ceiling + borrowed %dx%d "
                    "padding (Fig3 staircase) + order/shape factor <=%gx (Fig4); memory roofline at "
                    "bw_eff_low=%.1f GB/s (Fig5 59%% band); borrowed, NO silicon (#13)"
                    % (int(self.tops), self.sd, self.sd, _ORDER_SHAPE_MAX, self.bw_eff_low))
        if self.engine == "onnxim":   # Phase 1.3 heavy backend (drop-in; ONNXim != #13 silicon)
            hit = self.onnxim.get((wl.M, wl.K, wl.N))
            if hit is not None:
                lat, bound = float(hit), "compute"
                prov = "simulated (ONNXim generic-systolic RKNPU2-approx, NOT silicon; Phase 1.3)"
            else:
                prov += ("; engine='onnxim' requested but the ONNXim C++ build is deferred -> "
                         "ANALYTIC fallback (risk-#7 documented, report user)")
        elif self.engine == "scalesim":   # Phase 1.6 third engine (drop-in; SCALE-Sim != #13 silicon)
            hit = self.scalesim.get((wl.M, wl.K, wl.N))
            if hit is not None:
                lat, bound = float(hit), "compute"
                prov = ("simulated (SCALE-Sim v2 32x32-WS systolic, 3-core /cores aggregation; native "
                        "behaviour emergent NOT tuned; RKNPU2-approx, NOT silicon; Phase 1.6)")
            else:
                prov += ("; engine='scalesim' requested but this shape was skipped as cycle-sim-"
                         "intractable -> ANALYTIC fallback (documented)")
        return {"latency_us": float(lat), "bound": bound, "provenance": prov}
