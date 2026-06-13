"""M3 resources — the contended shared memory bandwidth + per-unit compute slots (Phase 2.1).

ADR-0001: compute units run concurrently; **shared memory bandwidth is the one
contended resource**. The contention model is a saturating fair-share: each of `k`
concurrent memory streams demands the single-stream `eff_BW_GBs`, but the aggregate
is capped at `knee_GBs * interconnect_efficiency`. So the aggregate rises with added
streams until it saturates at the knee (the memory-wall shape), and each stream's
effective rate falls as `min(eff_BW, knee/k)`.

The knee is a SIMULATED swept assumption (no concurrent-unit silicon — Aetina offline,
issue #52), anchored to Ramulator2 multi-stream + the Card 4c/1c trend. NOT a reproduced
silicon value (see validation/contracts/m3.yaml).
"""
from __future__ import annotations


class SharedBandwidth:
    """Saturating fair-share memory channel. `eff_BW_GBs` = single-stream effective
    bandwidth (e.g. the measured 24.2 GB/s on-card LPDDR4x anchor); `knee_GBs` = the
    aggregate ceiling concurrent streams saturate at (SIMULATED); `interconnect_efficiency`
    de-rates the achievable aggregate."""

    def __init__(self, eff_BW_GBs, knee_GBs=None, interconnect_efficiency=1.0):
        self.eff_BW = float(eff_BW_GBs)
        # default knee = single-stream eff_BW (no headroom for concurrency) unless given
        self.knee = float(knee_GBs) if knee_GBs else float(eff_BW_GBs)
        self.icn = float(interconnect_efficiency)

    def aggregate_GBs(self, k, *, contention=True):
        """Total achievable bandwidth with k concurrent memory streams (GB/s)."""
        if k <= 0:
            return 0.0
        if not contention:
            return k * self.eff_BW                       # ablation: no knee, sums linearly
        return min(k * self.eff_BW, self.knee * self.icn)

    def per_stream_GBs(self, k, *, contention=True):
        """Bandwidth available to EACH of k concurrent streams (GB/s)."""
        if k <= 0:
            return self.eff_BW
        return self.aggregate_GBs(k, contention=contention) / k

    def stream_us(self, nbytes, k, *, contention=True):
        """Time to stream `nbytes` bytes as one of k concurrent streams (microseconds)."""
        if nbytes <= 0:
            return 0.0
        rate = self.per_stream_GBs(k, contention=contention)   # GB/s
        return nbytes / (rate * 1e9) * 1e6


class ComputeUnit:
    """One concurrent compute slot (cim/gpu/npu/cpu). Ops on the same unit serialize
    (busy_until); different units run concurrently. `engine` is the Phase-1 UnitEngine /
    model used to price an op's compute latency."""

    def __init__(self, name, engine=None):
        self.name = name
        self.engine = engine
        self.busy_until = 0.0
