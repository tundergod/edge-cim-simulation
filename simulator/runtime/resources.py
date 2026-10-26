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

PER-UNIT STREAM CAPS (ADR-0009 revision F10, SoC profiles only): with `unit_caps`
({unit: GB/s}, default = `eff_BW_GBs` x the profile's per-unit `stream_eff`) the streams of ONE
unit can never exceed that unit's single-stream cap in aggregate (HeteroInfer §3.3: a unit
alone pulls ~43 GB/s however many kernels it has in flight), while streams on DIFFERENT units
share the knee by water-filling (GPU+NPU -> ~60). Without `unit_caps` (the legacy Phase-2.1
path) the old equal fair-share over all streams is used, byte-identically.
"""
from __future__ import annotations


class SharedBandwidth:
    """Saturating fair-share memory channel. `eff_BW_GBs` = single-stream effective
    bandwidth (e.g. the measured 24.2 GB/s on-card LPDDR4x anchor); `knee_GBs` = the
    aggregate ceiling concurrent streams saturate at (SIMULATED); `interconnect_efficiency`
    de-rates the achievable aggregate."""

    def __init__(self, eff_BW_GBs, knee_GBs=None, interconnect_efficiency=1.0, unit_caps=None):
        self.eff_BW = float(eff_BW_GBs)
        # per-unit single-stream caps (GB/s); None = legacy uniform fair-share (no per-unit cap)
        self.unit_caps = None if unit_caps is None else {u: float(c) for u, c in unit_caps.items()}
        if self.unit_caps is not None and any(c <= 0 for c in self.unit_caps.values()):
            raise ValueError(f"SharedBandwidth: every unit cap must be > 0 (got {unit_caps!r})")
        # default knee = single-stream eff_BW (no headroom for concurrency) unless given.
        # Distinguish None (use default) from 0/negative (invalid) -> fail loud, never mask.
        self.knee = float(eff_BW_GBs) if knee_GBs is None else float(knee_GBs)
        self.icn = float(interconnect_efficiency)
        if self.eff_BW < 0:
            raise ValueError(f"SharedBandwidth: eff_BW_GBs must be >= 0 (got {eff_BW_GBs!r})")
        if self.knee <= 0:
            raise ValueError(
                f"SharedBandwidth: knee_GBs must resolve to > 0 "
                f"(got {self.knee!r}; knee_GBs={knee_GBs!r}, eff_BW={self.eff_BW!r})")
        if self.icn <= 0:
            raise ValueError(
                f"SharedBandwidth: interconnect_efficiency must be > 0 (got {interconnect_efficiency!r})")

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

    def unit_cap(self, unit):
        """Single-stream cap of `unit` (GB/s): its `unit_caps` entry, else eff_BW."""
        if self.unit_caps is None:
            return self.eff_BW
        return self.unit_caps.get(unit, self.eff_BW)

    def share(self, streams, *, contention=True):
        """Per-stream rate (GB/s) for `streams` = {unit: n_active_streams}. Legacy (no unit_caps):
        every stream gets aggregate_GBs(k)/k. With unit_caps: a unit's streams split
        min(n_u * eff_BW, cap_u) equally, and the units' demands are water-filled under the knee
        (each unit gets min(demand_u, fair share of the remaining knee), excess redistributed).
        `contention=False` (ablation) removes both the knee and the per-unit cap."""
        k = sum(streams.values())
        if k <= 0:
            return {}
        if self.unit_caps is None or not contention:
            r = self.per_stream_GBs(k, contention=contention)
            return {u: r for u in streams}
        demand = {u: min(n * self.eff_BW, self.unit_cap(u)) for u, n in streams.items() if n > 0}
        grant = _waterfill(demand, self.knee * self.icn)
        return {u: grant[u] / streams[u] for u in demand}

    def stream_us(self, nbytes, k, *, contention=True):
        """Time to stream `nbytes` bytes as one of k concurrent streams (microseconds)."""
        if nbytes <= 0:
            return 0.0
        rate = self.per_stream_GBs(k, contention=contention)   # GB/s
        return nbytes / (rate * 1e9) * 1e6


def _waterfill(demand, total):
    """Max-min fair allocation of `total` over `demand` = {key: wanted}: everyone gets
    min(wanted, share), the unused remainder is re-shared among the still-unsatisfied."""
    grant, left, remaining = {}, dict(demand), float(total)
    while left:
        share = remaining / len(left)
        done = [u for u, d in left.items() if d <= share + 1e-12]
        if not done:
            for u in left:
                grant[u] = share
            break
        for u in done:
            grant[u] = left.pop(u)
            remaining -= grant[u]
    return grant


class ComputeUnit:
    """One concurrent compute slot (cim/gpu/npu/cpu). Ops on the same unit serialize
    (busy_until); different units run concurrently. `engine` is the Phase-1 UnitEngine /
    model used to price an op's compute latency."""

    def __init__(self, name, engine=None):
        self.name = name
        self.engine = engine
        self.busy_until = 0.0
