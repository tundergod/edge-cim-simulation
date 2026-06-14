"""Phase 2.3 step 1 — topology wiring (card / alpha / edge) + the topology x memory_spec
consistency contract.

Single source of the decode memory wall = the topology spec, resolved through the existing
simulator/models/m2_memory.py::MemoryModel (do NOT build a second resolver):
  cim_topo_card  -> on-card LPDDR4x dram_eff_BW_GBs 24.2 (the MEASURED L4 anchor), floor 0
  cim_topo_alpha -> PCIe pcie_BW_GBs 3.9 (no on-card DRAM, all traffic over host PCIe;
                    COUNTERFACTUAL: Alpha is LLM-incapable, -1301), per-call floor 911.1 us
  cim_topo_edge  -> mem_lpddr5 eff_BW 33.3 x noc_efficiency 0.9 = 29.97 (SIMULATED), floor 0

memory_spec is a CONSISTENCY TAG asserted against the topology's own keys (single physics
source = the topology spec), NOT a second bandwidth feed: an explicit memory_spec that
disagrees with the topology fails loud at BOTH the SimConfig boundary and the runner.run()
boundary. The per-call floor is added to TTFT only (decode tok/s stays bandwidth-bound).

    .venv/bin/pytest tests/test_topology.py
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.runtime.config import SimConfig          # noqa: E402
from simulator.runtime.platform import Platform         # noqa: E402
from simulator.runtime.runner import run                # noqa: E402


# ---- eff_BW + per-call floor resolution (single source = topology spec via MemoryModel) ----

def test_card_eff_bw_is_the_literal_24_2_byte_identical():
    # the L4 anchor must stay the literal 24.2 (byte-identity guard) and pay no per-call floor
    plat = Platform("llama-3.2-1b", topology="cim_topo_card", memory_spec="mem_lpddr4x")
    assert plat.bw.eff_BW == 24.2
    assert plat.per_call_floor_us == 0.0


def test_alpha_eff_bw_is_pcie_3_9_with_911_floor():
    # no on-card DRAM -> the wall is the host PCIe BW; pays the measured 911.1 us per-call floor
    plat = Platform("llama-3.2-1b", topology="cim_topo_alpha")
    assert plat.bw.eff_BW == 3.9
    assert plat.per_call_floor_us == pytest.approx(911.1)


def test_edge_eff_bw_is_lpddr5_times_noc():
    # integrated edge: LPDDR5 eff 33.3 x noc_efficiency 0.9, no per-call floor
    plat = Platform("llama-3.2-1b", topology="cim_topo_edge", memory_spec="mem_lpddr5")
    assert plat.bw.eff_BW == pytest.approx(33.3 * 0.9)
    assert plat.per_call_floor_us == 0.0


# ---- topology x memory_spec consistency contract (fail-loud, both boundaries) ----

def _cfg(topology, memory_spec="__omit__", model="llama-3.2-1b", **plat):
    p = {"topology": topology, **plat}
    if memory_spec != "__omit__":
        p["memory_spec"] = memory_spec
    sched = "all_cim"
    units = {"cim": True, "gpu": True, "npu": False, "cpu": True}
    return {"workload": {"model": model}, "platform": {**p, "units": units},
            "scheduler": {"policy": sched}}


def test_consistent_combos_accepted():
    SimConfig.from_dict(_cfg("cim_topo_card"))                 # card + omitted -> mem_lpddr4x
    SimConfig.from_dict(_cfg("cim_topo_card", "mem_lpddr4x"))  # card + explicit match
    SimConfig.from_dict(_cfg("cim_topo_edge"))                 # edge + omitted -> mem_lpddr5
    SimConfig.from_dict(_cfg("cim_topo_edge", "mem_lpddr5"))   # edge + explicit match
    SimConfig.from_dict(_cfg("cim_topo_alpha"))                # alpha + omitted (no DRAM) OK


@pytest.mark.parametrize("topology,memory_spec", [
    ("cim_topo_card", "mem_lpddr5"),     # card has LPDDR4x, not LPDDR5
    ("cim_topo_edge", "mem_lpddr4x"),    # edge uses the SoC LPDDR5, not the card's 4x
    ("cim_topo_alpha", "mem_lpddr4x"),   # alpha has no on-card DRAM -> any LPDDR spec is misleading
    ("cim_topo_alpha", "mem_lpddr5"),
])
def test_mismatch_raises_at_config_boundary(topology, memory_spec):
    with pytest.raises(ValueError):
        SimConfig.from_dict(_cfg(topology, memory_spec))


def test_mismatch_raises_at_runner_boundary():
    # construct a valid card cfg, then mutate to an inconsistent memory_spec -> run() re-validates
    cfg = SimConfig.from_dict(_cfg("cim_topo_card"))
    cfg.memory_spec = "mem_lpddr5"
    with pytest.raises(ValueError):
        run(cfg)


def test_unknown_topology_raises_both_boundaries():
    with pytest.raises((ValueError, NotImplementedError)):
        SimConfig.from_dict(_cfg("cim_topo_bogus"))
    cfg = SimConfig.from_dict(_cfg("cim_topo_card"))
    cfg.topology = "cim_topo_bogus"
    with pytest.raises((ValueError, NotImplementedError)):
        run(cfg)


# ---- honesty tiers: card calibrated, alpha/edge flagged ----

def test_card_default_is_calibrated_anchor():
    assert SimConfig.from_dict(_cfg("cim_topo_card")).is_calibrated_anchor()
    assert SimConfig.from_dict(_cfg("cim_topo_card", "mem_lpddr4x")).is_calibrated_anchor()


def test_alpha_and_edge_are_flagged_not_calibrated():
    alpha = SimConfig.from_dict(_cfg("cim_topo_alpha"))
    edge = SimConfig.from_dict(_cfg("cim_topo_edge"))
    assert not alpha.is_calibrated_anchor()
    assert not edge.is_calibrated_anchor()
    assert any("simulated" in p for p in edge.provenance)
