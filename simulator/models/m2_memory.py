"""M2 — memory hierarchy: PCIe/DMA (2a), host LPDDR5 backend (2b), kv_cache append (2c).

PCIe (2a): per-call host<->device transfer = fixed_overhead + bytes/BW. The 911 us
per-call DMA floor (measurements/aetina pcie_floor_A1d5) and 3.9 GB/s Gen3x4 BW are
FIXED params (no per-shape PCIe sweep was collected). Applicability boundary: the floor
is paid by DISCRETE host<->device transfers (KV-reload, activation handoff, conversion-op
traffic); decode weight-streaming uses the bandwidth term only (no per-call floor) in the
recompose/production prediction. On the Alpha board itself every decode-GEMV call did pay
the floor (a topology artifact, not extrapolated to the production card).

LPDDR5 (2b): analytic effective-bandwidth model (Ramulator2 integration deferred to
Phase 2, ADR-0002 swappable). Effective BW parameterised from JEDEC LPDDR5 peak + the
measured ~24 GB/s decode wall.

kv_cache (2c): KV-append is a pure-bandwidth op (Phase 0.3 did not isolate it) ->
analytic kv_bytes/BW_eff, UNVALIDATED.
"""
import json
from pathlib import Path

_PCIE = Path(__file__).parent / "params" / "m2_pcie.json"
_LPDDR5 = Path(__file__).parent / "params" / "m2_lpddr5.json"


class MemoryModel:
    def __init__(self, pcie=None, lpddr5=None):
        p = pcie if pcie is not None else json.loads(_PCIE.read_text())
        l = lpddr5 if lpddr5 is not None else json.loads(_LPDDR5.read_text())
        self.floor_us = p["fixed_overhead_us_median"]
        self.pcie_BW_GBs = p["pcie_BW_GBs"]
        self.lpddr5_eff_BW_GBs = l["effective_BW_GBs"]
        self.lpddr5_peak_GBs = l["jedec_peak_GBs"]

    def pcie_transfer_us(self, nbytes, discrete=True):
        """Host<->device transfer latency (us). discrete=True pays the per-call floor."""
        bw_us = nbytes / (self.pcie_BW_GBs * 1e9) * 1e6
        return (self.floor_us if discrete else 0.0) + bw_us

    def lpddr5_stream_us(self, nbytes):
        """Host LPDDR5 streaming latency (us) at the effective decode bandwidth."""
        return nbytes / (self.lpddr5_eff_BW_GBs * 1e9) * 1e6

    def kv_append_us(self, kv_bytes):
        """KV-cache append (bandwidth op). Analytic, UNVALIDATED (Phase 0.3 gap)."""
        return kv_bytes / (self.lpddr5_eff_BW_GBs * 1e9) * 1e6
