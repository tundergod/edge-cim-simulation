"""M7 — energy model (spec-based, ADR-0005). params/m7_energy.json.

No power telemetry on either board -> spec-based per-component estimation:
  CIM  = vendor 15 TOPS/W (INT8); energy_J = ops / (TOPS/W * 1e12), ops = 2*MAC
  DRAM = JEDEC pJ/bit * bits          PCIe = spec pJ/bit * bits
  CPU  = ARM A76 datasheet W * cores * active time
Energy is ESTIMATED, not measured. Report with +/-20% coefficient sensitivity; the
qualitative conclusion (which unit dominates) must be robust to that band.
"""
import json
from pathlib import Path

_PARAMS = Path(__file__).parent / "params" / "m7_energy.json"


class EnergyModel:
    def __init__(self, params=None, scale=None):
        p = params if params is not None else json.loads(_PARAMS.read_text())
        s = scale or {}
        self.cim_tops_w = p["cim_tops_w"] * s.get("cim", 1.0)
        self.lpddr5_pj_bit = p["lpddr5_pj_per_bit"] * s.get("dram", 1.0)
        self.pcie_pj_bit = p["pcie_pj_per_bit"] * s.get("pcie", 1.0)
        self.a76_core_w = p["a76_core_w"] * s.get("cpu", 1.0)
        self.cpu_cores = p["cpu_cores"]

    def cim_J(self, flops):
        return flops / (self.cim_tops_w * 1e12)

    def dram_J(self, nbytes):
        return nbytes * 8 * self.lpddr5_pj_bit * 1e-12

    def pcie_J(self, nbytes):
        return nbytes * 8 * self.pcie_pj_bit * 1e-12

    def cpu_J(self, active_us):
        return self.a76_core_w * self.cpu_cores * active_us * 1e-6
