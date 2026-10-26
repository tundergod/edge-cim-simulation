"""Test doubles for the ADR-0009 pricing contract (worker A's Platform): `price(node)`,
`compute_us(node)`, `supports(unit, category)`, `bw`. NOT a model of any silicon.

AnalyticFakePlatform  — simple MAC-rate costs per unit (NPU pads M to 32; optionally
                        static-graph only: non-standard M on the NPU is unpriceable).
Table3FakePlatform    — the HeteroInfer Table 3 anchor latencies (8 Gen 3, [LIT]) for the
                        GPU/NPU, scaled LINEARLY in N for sub-shapes; any other (K, M) is a
                        KeyError (the solver must only ask for shapes the test intends).
Both return `wl.extra["fixed_us"]` as the separate `fixed_us` field of price() (the ADR-0009
rev F1 contract: a serial pre-op delay, latency_us = compute only) and price `merge` at 0 compute.
"""
from __future__ import annotations

from simulator.runtime.resources import SharedBandwidth
from simulator.runtime import tiling

_SUPPORT = ("norm", "rope", "ffn", "softmax", "residual")
_COMPUTE = ("matmul", "attention") + _SUPPORT + ("sampling",)
STD = (1, 128, 256, 512, 1024)


def _pad32(m):
    return -(-m // 32) * 32


class AnalyticFakePlatform:
    """rates = {unit: MAC/us}; support ops cost nbytes / byte_rate[unit] (+ launch_us)."""

    def __init__(self, rates=None, byte_rates=None, *, bw_GBs=24.2, knee_GBs=48.0,
                 static_npu=False, launch_us=0.0, npu_fixed_us=0.0):
        self.rates = rates or {"gpu": 1.0e6, "npu": 4.0e6, "cim": 8.0e6, "cpu": 0.1e6}
        self.byte_rates = byte_rates or {"gpu": 2e4, "npu": 1e4, "cim": 1e4, "cpu": 4e3}
        self.bw = SharedBandwidth(bw_GBs, knee_GBs=knee_GBs)
        self.static_npu = static_npu
        self.launch_us = launch_us
        self.npu_fixed_us = npu_fixed_us

    def supports(self, unit, cat):
        if cat in ("merge", "convert"):
            return True
        if unit == "mem":
            return cat in ("embedding", "kv_cache")
        if unit == "npu":
            return cat == "matmul"
        if unit == "cim":
            return cat in ("matmul", "attention")
        return cat in _COMPUTE                       # gpu, cpu

    def compute_us(self, node):
        return self.price(node)["latency_us"]

    def price(self, node):
        w, u, cat = node.wl, node.unit, node.category
        fixed = float(w.extra.get("fixed_us", 0.0))
        if cat in ("merge", "convert") or u == "mem":
            lat = 0.0
        elif cat == "matmul":
            M = w.M
            if u == "npu":
                if self.static_npu and M not in STD:
                    lat = float("inf")              # no pre-built NPU graph for this length
                else:
                    lat = _pad32(M) * w.K * w.N / self.rates[u] + self.npu_fixed_us
            else:
                lat = M * w.K * w.N / self.rates[u]
        elif cat == "attention" and "hd" in w.extra:
            if node.pricing_group is not None and node.pricing_group != node.id:
                lat = 0.0                            # grouped member: priced on the rep
            else:
                lat = 2 * w.heads * tiling._attn_M(node) * w.extra["hd"] * w.kv / self.rates[u]
        else:
            lat = self.launch_us + w.nbytes / self.byte_rates.get(u, 1e4)
        return {"latency_us": lat, "fixed_us": fixed, "compute_provenance": f"fake analytic ({u}/{cat})",
                "source_model": "fake"}


# HeteroInfer Table 3 (8 Gen 3; us) keyed (unit, K, M) -> (N_anchor, latency). [LIT]
TABLE3 = {
    ("gpu", 4096, 1): (4096, 511.0),   ("npu", 4096, 1): (4096, 693.0),
    ("gpu", 4096, 1, 28672): (28672, 1903.0), ("npu", 4096, 1, 28672): (28672, 3886.0),
    ("gpu", 14336, 1): (4096, 1467.0), ("npu", 14336, 1): (4096, 6506.0),
    ("gpu", 4096, 128): (4096, 7306.0), ("npu", 4096, 128): (4096, 912.0),
    ("gpu", 4096, 256): (4096, 10841.0), ("npu", 4096, 256): (4096, 1884.0),
    ("gpu", 14336, 256): (4096, 35231.0), ("npu", 14336, 256): (4096, 23445.0),
}


class Table3FakePlatform:
    """GPU/NPU matmul latency = Table 3 anchor for (K, M) scaled linearly in N; the
    [28672,4096] row is a separate anchor (same K, M) selected by N_full (the solver's
    sub-shapes are tagged with the original N through `n_full`). Everything else is 0.
    Memory is made irrelevant (huge bandwidth) so decisions are pure Table-3 compute."""

    def __init__(self, n_full=4096):
        self.bw = SharedBandwidth(1e6, knee_GBs=1e6)
        self.n_full = n_full

    def supports(self, unit, cat):
        if cat in ("merge", "convert"):
            return True
        if unit in ("gpu", "npu"):
            return cat == "matmul" or (unit == "gpu" and cat in _COMPUTE)
        return False

    def compute_us(self, node):
        return self.price(node)["latency_us"]

    def price(self, node):
        w, u, cat = node.wl, node.unit, node.category
        fixed = float(w.extra.get("fixed_us", 0.0))
        if cat != "matmul":
            return {"latency_us": 0.0, "fixed_us": fixed, "compute_provenance": "fake", "source_model": "fake"}
        key = (u, w.K, w.M) if self.n_full == 4096 else (u, w.K, w.M, self.n_full)
        n_anchor, lat = TABLE3[key]
        return {"latency_us": lat * w.N / n_anchor, "fixed_us": fixed,
                "compute_provenance": "HeteroInfer Tab.3 anchor, linear in N [LIT]", "source_model": "fake_tab3"}
