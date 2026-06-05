"""M1-SPM — Metis AIPU on-chip SRAM tier (Phase 1.2, CACTI tier, NOT Ramulator2).

`SramTier(spec)` reads sram_metis_aipu: a quad-core scratchpad hierarchy (L1 4 MiB/core +
L2 32 MiB shared + D-IMC 1 MiB/core = 52 MiB) with a CACTI/ASSUMPTION (latency, BW) pair
(no published Metis SRAM BW/latency). `residency='architecture-only'`: an 8B INT8 weight
set (~8 GB) >> 32 MiB L2, so weights are NEVER resident — predict() RESOLVES such a
workload to the DRAM tier (the decode memory wall is host LPDDR, not SRAM). SRAM holds
activation / KV tiles only.

predict(Workload) returns the frozen {latency_us, bound='memory', provenance}; the
provenance names which tier the working set resolved to (SRAM vs spilled-to-DRAM). This
makes the SRAM residency a load-bearing what-if variable (per the reversed M2 decision)
WITHOUT claiming weights ever fit.
"""
from simulator.models.engine import UnitEngine, Workload, check_return

_MiB = 1024 * 1024
DRAM_EFF_BW_GBs = 24.2   # production-card LPDDR4x decode anchor (mem_lpddr4x), the spill target


class SramTier(UnitEngine):
    def __init__(self, spec, engine="analytic"):
        super().__init__(spec, engine)
        self.l2_MiB = spec["l2_MiB_shared"]            # 32 MiB shared = the residency capacity
        self.bw_GBs = spec["bw_GBs"]                   # CACTI assumption
        self.latency_ns = spec["latency_ns"]           # CACTI assumption
        self.residency = spec["residency"]             # 'architecture-only'
        self.dram_eff_BW_GBs = DRAM_EFF_BW_GBs

    def resident(self, nbytes):
        """True iff the working set fits the L2 residency capacity (32 MiB)."""
        return nbytes <= self.l2_MiB * _MiB

    def predict(self, wl: Workload) -> dict:
        """Time a working set against the SRAM tier. If it fits L2 -> SRAM BW + access
        latency; else it RESOLVES to the DRAM tier (never resident) -> DRAM eff_BW. An
        8B weight set (>>32 MiB) always takes the DRAM branch."""
        nbytes = wl.nbytes if wl.nbytes else wl.kv
        if self.resident(nbytes):
            lat = self.latency_ns / 1e3 + nbytes / (self.bw_GBs * 1e9) * 1e6
            prov = (f"analytic SRAM tier (resident, <= L2 {self.l2_MiB}MiB): "
                    f"access {self.latency_ns}ns + bytes/BW({self.bw_GBs}GB/s) "
                    "[CACTI assumption; residency architecture-only]")
        else:
            lat = nbytes / (self.dram_eff_BW_GBs * 1e9) * 1e6
            prov = (f"working set > L2 {self.l2_MiB}MiB -> resolves to DRAM tier (never "
                    f"resident): bytes/eff_BW({self.dram_eff_BW_GBs}GB/s) "
                    "[residency architecture-only; DRAM BW = calibrated LPDDR4x anchor]")
        return check_return({"latency_us": lat, "bound": "memory", "provenance": prov})
