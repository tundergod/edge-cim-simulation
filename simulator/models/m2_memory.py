"""M2 — memory hierarchy engine (Phase 1.2): one analytic engine, swappable spec.

`MemoryModel(spec, engine='analytic')` (a `UnitEngine`) covers TWO spec kinds with the
SAME predict() contract — the spec decides which physics applies:

  * a MEMORY spec (mem_lpddr4/4x/5; has `eff_BW_GBs`) -> streaming / kv-append timed at
    that spec's effective bandwidth. bound = 'memory'. The production-card LPDDR4x
    eff_BW 24.2 GB/s (mem_lpddr4x) is the MEASURED decode-wall anchor (calibrated);
    LPDDR5 33.3 is SIMULATED (sim eff 0.65 < the 0.71 measured on 4x — different memory,
    discounted not assumed equal); peaks are ASSUMPTION.

  * a CIM TOPOLOGY spec (cim_topo_alpha/card; has `topology`) -> a discrete host<->device
    PCIe transfer = `per_call_floor_us + nbytes/pcie_BW`. Alpha pays the MEASURED 911 us
    per-call DMA floor (bound='floor'); the Card pays 0 (on-card streaming, bound='memory').
    The floor is a TOPOLOGY artifact of the pre-production Alpha board, NOT extrapolated to
    the production card (see A2.2).

predict(Workload) returns the frozen {latency_us, bound, provenance}. The convenience
methods (`kv_append_us`, `lpddr5_stream_us`, `pcie_transfer_us`) are thin wrappers over the
spec model so the Phase-1.1 recompose/fit_m2 callers migrate with a minimal diff.

Heavy backend (Ramulator2, `engine='ramulator2'`) = Phase 1.3, same constructor + contract.
kv-append stays UNVALIDATED (Phase 0.3 did not isolate it; pure-BW form is correct, board
offline -> re-calibrate the coefficient when recovered).
"""
from simulator.models.engine import UnitEngine, Workload, check_return


class MemoryModel(UnitEngine):
    def __init__(self, spec, engine="analytic"):
        super().__init__(spec, engine)
        self.is_topo = "topology" in spec
        if self.is_topo:
            self.pcie_BW_GBs = spec["pcie_BW_GBs"] if "pcie_BW_GBs" in spec else None
            self.floor_us = spec.get("per_call_floor_us", 0.0)
            # the Card streams from on-card DRAM at its measured eff_BW; Alpha has none
            self.eff_BW_GBs = spec.get("dram_eff_BW_GBs")
        else:
            self.eff_BW_GBs = spec["eff_BW_GBs"]   # mem_lpddr4/4x/5 effective bandwidth

    # ---- engine contract -------------------------------------------------
    def predict(self, wl: Workload) -> dict:
        """Time one memory op. `op='pcie'` needs a topology spec; stream/kv-append need a
        memory spec (or the Card's on-card DRAM). bytes come from wl.nbytes (or kv)."""
        if wl.op == "pcie":
            assert self.is_topo and self.pcie_BW_GBs, "op='pcie' needs a CIM topology spec with pcie_BW_GBs"
            bw_us = wl.nbytes / (self.pcie_BW_GBs * 1e9) * 1e6
            lat = self.floor_us + bw_us
            bound = "floor" if self.floor_us > 0 else "memory"
            prov = (f"analytic PCIe transfer = per_call_floor + bytes/BW; floor={self.floor_us}us "
                    f"[{'measured' if self.floor_us else 'architecture (card, no floor)'}], "
                    f"BW={self.pcie_BW_GBs}GB/s [measured]")
        else:  # stream / kv_append: pure-bandwidth at the spec's effective BW
            assert self.eff_BW_GBs, "stream/kv-append needs a memory spec (eff_BW_GBs) or on-card DRAM"
            nbytes = wl.nbytes if wl.nbytes else wl.kv
            lat = nbytes / (self.eff_BW_GBs * 1e9) * 1e6
            bound = "memory"
            tag = self._bw_tag()
            kv_note = " (kv-append UNVALIDATED: Phase 0.3 gap, pure-BW form)" if wl.op == "kv_append" else ""
            prov = f"analytic stream = bytes/eff_BW({self.eff_BW_GBs}GB/s) [{tag}]{kv_note}"
        return check_return({"latency_us": lat, "bound": bound, "provenance": prov})

    def _bw_tag(self):
        """Honesty tag for the bound memory's effective BW."""
        mt = self.spec.get("memory_type") or self.spec.get("dram_type")
        if mt == "LPDDR4x":
            return "calibrated (production-card LPDDR4x decode anchor 24.2 GB/s)"
        if mt == "LPDDR5":
            return "simulated (eff 0.65 < measured 0.71, different memory)"
        return "assumption (eff derived from measured-4x efficiency)"

    # ---- convenience wrappers (keep recompose/fit_m2 migration minimal) ---
    def pcie_transfer_us(self, nbytes, discrete=True):
        """Host<->device PCIe transfer latency (us). discrete=True pays the per-call floor.

        `discrete=False` drops the floor (decode weight-streaming on a card with on-card
        DRAM uses the BW term only; see A2.2)."""
        bw_us = nbytes / (self.pcie_BW_GBs * 1e9) * 1e6
        return (self.floor_us if discrete else 0.0) + bw_us

    def lpddr5_stream_us(self, nbytes):
        """Host DRAM streaming latency (us) at the bound spec's effective bandwidth."""
        return self.predict(Workload(op="stream", nbytes=int(nbytes)))["latency_us"]

    def kv_append_us(self, kv_bytes):
        """KV-cache append (pure-bandwidth op). Analytic, UNVALIDATED (Phase 0.3 gap)."""
        return self.predict(Workload(op="kv_append", nbytes=int(kv_bytes)))["latency_us"]
