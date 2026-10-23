"""M4 — RKNPU2 (RK3588 NPU) timing model — ANALYTIC systolic-roofline (Phase 1.2, D2).

NO RKNPU2 silicon (board offline, issue #13) -> EVERY number here is `simulated`/`borrowed`,
NOT calibrated. This is an analytic systolic-array roofline whose *shape* is borrowed from the
HeteroInfer characterization (SOSP'25, docs/papers/methodology-and-simulators/) and whose ceilings come
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
  - dispatch floor: a small fixed per-op cost so tiny ops are not free.
  bound = argmax(compute, floor). KERNEL TIME ONLY (ADR-0008 revision F2): the weight/activation
  DRAM traffic is metered by the shared pool in the runtime (the Fig5 per-unit 40-45 GB/s band is
  the profile's `stream_eff`), no longer a memory ceiling inside this model.

Native attention (op='attn_bmm'): activation x activation, no static weight to stall on -> pure
compute-bound at the padded TOPS ceiling (no order/shape penalty, no weight stream). Per (kv,heads,
layers): QK^T (K=hd, N=kv) + S.V (K=kv, N=hd), padded to 32, x heads (batch) x ceil(M/32) query-row
steps (F5: the 32-row staircase applies to the query rows in prefill).

dtypes: INT4/8/16 + FP16 only (datasheet). No RKNPU2 power telemetry -> energy NOT determinable.
"""
import json
import math
from pathlib import Path

from simulator.models.engine import UnitEngine, Workload

# borrowed HeteroInfer constants (SOSP'25 §3.2/§3.3) — all `borrowed`, none measured here.
_ORDER_SHAPE_MAX = 6.0      # Fig4: up to 6x order/shape penalty
_DISPATCH_FLOOR_US = 2.0    # simulated fixed per-op dispatch (no silicon -> nominal, assumption)
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
        self.bw_eff_low = float(bw["eff_low"])     # ~20.1 GB/s  (informational: the band now enters the
        self.bw_eff_high = float(bw["eff_high"])   # ~22.4 GB/s   runtime as the profile's npu.stream_eff, rev F2)
        self.dtypes = set(spec["dtypes"])          # restricted to INT4/8/16/FP16 (datasheet)
        # Phase 1.3 heavy backend: engine='onnxim' replaces the analytic latency with ONNXim's
        # per-shape sim (same constructor + frozen contract). Empty table -> analytic fallback.
        self.onnxim = _onnxim_table() if engine == "onnxim" else {}
        # Phase 1.6 third engine: engine='scalesim' (SCALE-Sim v2, same drop-in contract).
        self.scalesim = _scalesim_table() if engine == "scalesim" else {}

    def scale_compute(self, s):
        """Scalable axis (ADR-0008): multiply the NPU COMPUTE throughput (TOPS) by `s` > 0. The
        weight-streaming bandwidth and the dispatch floor are untouched (a decode GEMV stays
        memory-bound). EXTRAPOLATED: no silicon at any s != 1."""
        if s <= 0:
            raise ValueError(f"npu_scale must be > 0, got {s!r}")
        self.tops *= float(s)
        return self

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
        """Padded-MAC systolic kernel time for one GEMM (compute vs dispatch floor; DRAM bytes are
        pool-metered by the runtime). Returns (latency_us, bound)."""
        Mp, Kp, Np = self._pad(M), self._pad(K), self._pad(N)
        padded_macs = Mp * Kp * Np
        osf = self._order_shape_factor(K, N)
        compute_us = 2.0 * padded_macs / (self.tops * 1e12) * 1e6 * osf
        return self._argmax_bound(compute_us)

    def _attn_us(self, kv, heads, layers, hd, dtype, M=1):
        """Native attention QK^T + S.V, compute-bound (act x act, no weight stall). Heads are the
        BATCH dimension -- NOT padded into the systolic M slot (padding GQA heads=8 up to sd=32 would
        4x-over-count the attention MACs); only the matmul dims hd/kv are padded to the 32x32 quantum,
        then multiplied by the raw head count and by ceil(M/32) query-row tiles (F5). The dispatch
        floor applies PER LAYER, not once for the whole L-layer rollup."""
        # per head per layer per 32-row query tile: QK^T (hd x kv) + S.V (kv x hd), padded; x heads.
        steps = max(1, math.ceil(max(1, M) / self.sd))
        per_head = self._pad(hd) * self._pad(kv) + self._pad(kv) * self._pad(hd)
        per_layer_us = 2.0 * heads * steps * per_head / (self.tops * 1e12) * 1e6
        lat = layers * max(per_layer_us, _DISPATCH_FLOOR_US)
        return lat, ("compute" if per_layer_us >= _DISPATCH_FLOOR_US else "floor")

    def _argmax_bound(self, compute_us):
        floor = _DISPATCH_FLOOR_US
        lat = max(compute_us, floor)
        return lat, ("compute" if lat == compute_us else "floor")

    def predict(self, wl: Workload) -> dict:
        """Frozen dict {latency_us, bound, provenance}. ALL outputs simulated/borrowed (no silicon)."""
        if wl.op in ("attn_bmm", "attention"):
            hd = wl.extra.get("hd", wl.K or 128)
            lat, bound = self._attn_us(wl.kv, wl.heads, wl.layers, hd, wl.dtype, M=wl.M)
            prov = ("simulated: analytic systolic attn (act x act, padded to borrowed %dx%d, x ceil(M/%d) "
                    "query tiles, compute-bound; HeteroInfer Fig3); NO silicon (#13)" % (self.sd, self.sd, self.sd))
        else:
            lat, bound = self._gemm_us(wl.M, wl.K, wl.N, wl.dtype)
            prov = ("simulated: analytic systolic kernel time %d TOPS INT8 ceiling + borrowed %dx%d "
                    "padding (Fig3 staircase) + order/shape factor <=%gx (Fig4); DRAM bytes metered by the "
                    "shared pool (per-unit stream_eff = the Fig5 band); borrowed, NO silicon (#13)"
                    % (int(self.tops), self.sd, self.sd, _ORDER_SHAPE_MAX))
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


# ---------------------------------------------------------------------------------------
# HeteroInfer-ANCHORED staircase variant (SoC profile sd8gen3_cim, ADR-0008). Coexists with the
# RKNPU2 analytic/onnxim/scalesim NpuModel above (P2 keeps that path untouched).
# ---------------------------------------------------------------------------------------
_ORDERS = ("weight_stationary", "activation_stationary")


class NpuStaircaseModel(NpuModel):
    """Hexagon-like (8 Gen 3) systolic NPU, FITTED to HeteroInfer's published anchors (SOSP'25
    Table 3 + Fig.3; docs: heteroinfer-design-reference.md §2.2/§3b). `borrowed` throughout —
    none of it is our silicon. Spec = npu_hexagon_sd8gen3 (needs the `anchor_fit` block written by
    tools/analysis/fit_hetero_anchors.py; residuals in validation/reports/hetero/anchor_fit.json).

    Model (weight [N_out=N, K_in=K], activation rows M; KERNEL TIME us, ADR-0008 rev F2):
        steps  = ceil(M / 32)                              # Fig.3 STAIRCASE: one 32-row tile per step
        g(K)   = max(1, (K / k_ref) ** k_penalty_exp)      # NPU-3: a large contraction dim hurts (~linear)
        t      = floor_us + g(K) * [ K*N*wbpp / load_GBs  +  steps * 2*32*K*N / tops_eff ] * order
      array WEIGHT-LOAD pass (the stationary operand is written into the 32x32 weight-stall arrays once
      per op at `load_GBs` ~43 GB/s — the fit recovers the paper's per-unit 40-45 GB/s, i.e. the array
      fill is paced by what one unit can pull; it is a KERNEL-side term, the DRAM bytes themselves are
      metered by the shared pool, so a split part never gets faster than its own array fill) + per-step
      streamed-activation term at the EFFECTIVE TOPS (7.3 vs 34 peak; the paper says "up to 10 TFLOPS
      in actual"). `order` = 1 for the paper's optimal weight-stationary form, `order_penalty_max` (6x,
      Fig.4) when wl.extra["order"] == "activation_stationary" — the flag a scheduler toggles to
      explore the reversed operand order. The anchors are refitted under max(kernel time, W bytes /
      43 GB/s) (fit_hetero_anchors.py): every Table-3 anchor is kernel-bound, so the parameters equal
      the pre-F2 fit (documented in ADR-0008 revisions).
    Attention (act x act, no weight stall): floor + heads*steps*2*32*pad(hd)*pad(kv)*2 / tops_eff
      (compute only; bytes pool-metered) — EXTRAPOLATED (no attention anchor in the paper).
    Elementwise support ops are priced by the Platform at the unit's launch floor (bytes metered by M3).
    """

    def __init__(self, spec, engine="analytic"):
        super().__init__(spec, engine)
        if engine != "analytic":
            raise ValueError(f"NpuStaircaseModel is analytic-only (the onnxim/scalesim tables are "
                             f"RKNPU2-approx, not sd8gen3); got engine={engine!r}")
        fit = spec["anchor_fit"]
        self.floor_us = float(fit["floor_us"])
        self.bw_w = float(fit["bw_GBs"])
        self.tops_eff = float(fit["tops_eff"])
        self.k_exp = float(fit["k_penalty_exp"])
        self.k_ref = float(fit["k_ref"])
        self.wbpp = float(fit["weight_bytes_per_param"])
        self.order_pen = float(spec.get("order_penalty_max", _ORDER_SHAPE_MAX))

    def scale_compute(self, s):
        super().scale_compute(s)
        self.tops_eff *= float(s)          # the fitted effective TOPS is this model's compute term
        return self

    def _k_penalty(self, K):
        return max(1.0, (K / self.k_ref) ** self.k_exp)

    def _order_factor(self, order):
        if order not in _ORDERS:
            raise ValueError(f"NPU wl.extra['order'] must be one of {_ORDERS}, got {order!r}")
        return self.order_pen if order == "activation_stationary" else 1.0

    def _gemm_us(self, M, K, N, dtype, order="weight_stationary"):
        steps = max(1, math.ceil(M / self.sd))
        g = self._k_penalty(K)
        load_us = K * N * self.wbpp / (self.bw_w * 1e3)          # array weight-load pass (kernel side)
        stream_us = steps * 2.0 * self.sd * K * N / (self.tops_eff * 1e12) * 1e6
        lat = self.floor_us + g * (load_us + stream_us) * self._order_factor(order)
        return lat, ("compute" if stream_us >= load_us else "floor")

    def _attn_us(self, M, kv, heads, hd, nbytes_bmm):
        steps = max(1, math.ceil(M / self.sd))
        per_head = self._pad(hd) * self._pad(kv) * 2            # QK^T + S.V padded tiles
        comp_us = heads * steps * 2.0 * self.sd * per_head / (self.tops_eff * 1e12) * 1e6
        return self.floor_us + comp_us, "compute"

    def predict(self, wl: Workload) -> dict:
        if wl.op in ("attn_bmm", "attention"):
            hd = wl.extra.get("hd", wl.K or 128)
            lat, bound = self._attn_us(max(1, wl.M), max(1, wl.kv), max(1, wl.heads), hd, wl.nbytes)
            prov = ("extrapolated: NPU attention act x act on the Table-3-anchored staircase model "
                    "(no attention anchor in HeteroInfer; padded to %dx%d, %s-bound; borrowed sd8gen3)"
                    % (self.sd, self.sd, bound))
        else:
            if wl.K <= 0 or wl.N <= 0:
                raise ValueError(f"NpuStaircaseModel GEMM needs K,N > 0 (got K={wl.K}, N={wl.N})")
            order = wl.extra.get("order", "weight_stationary")
            lat, bound = self._gemm_us(max(1, wl.M), wl.K, wl.N, wl.dtype, order)
            prov = ("borrowed: NPU 32-row staircase (steps=%d) fitted to HeteroInfer Table 3/Fig 3 "
                    "(floor=%.0f us, array weight-load %.1f GB/s, %.2f TOPS eff, K-penalty (K/%d)^%.2f, order=%s%s; "
                    "max anchor residual %.1f%%; %s-bound kernel time, DRAM bytes pool-metered; no silicon of ours)"
                    % (max(1, math.ceil(max(1, wl.M) / self.sd)), self.floor_us, self.bw_w, self.tops_eff,
                       int(self.k_ref), self.k_exp, order,
                       (" x%g Fig.4 penalty" % self.order_pen) if order == "activation_stationary" else "",
                       100 * float(self.spec["anchor_fit"].get("max_abs_rel_residual", 0.0)), bound))
        return {"latency_us": float(lat), "bound": bound, "provenance": prov}
