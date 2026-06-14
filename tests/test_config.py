"""Phase 2.1 — SimConfig user-input contract (simulator/runtime/config.py).

The calibrated core cannot be overridden via the config (unknown keys rejected
fail-loud); the forward-looking layer is settable and out-of-envelope inputs are
auto-tagged into provenance.

    .venv/bin/pytest tests/test_config.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.runtime.config import SimConfig  # noqa: E402

L4 = ROOT / "simulator/runtime/configs/l4_repro.json"


def test_l4_repro_loads_calibrated():
    cfg = SimConfig.from_json(L4)
    assert cfg.memory_spec == "mem_lpddr4x" and cfg.topology == "cim_topo_card"
    assert cfg.is_calibrated_anchor()           # no provenance flags
    assert cfg.provenance == []


def test_reject_calibrated_core_override():
    # an attempt to inject a calibrated quantity (memory rate) is an unknown key -> rejected
    for bad in ({"platform": {"eff_BW_GBs": 99}},
                {"platform": {"cim_params": {}}},
                {"unknownsection": {}}):
        d = {"workload": {"model": "llama-3.2-1b"}, **bad}
        try:
            SimConfig.from_dict(d)
        except ValueError:
            continue
        raise AssertionError(f"did not reject {bad}")


def test_out_of_envelope_flags_provenance():
    cfg = SimConfig.from_dict({
        "workload": {"model": "llama-3.1-8b"},
        "platform": {"memory_capacity_GB": 32, "topology": "cim_topo_edge",
                     "memory_spec": "mem_lpddr5", "bw_efficiency": 0.7,
                     "units": {"cim": True, "npu": True}},
    })
    assert not cfg.is_calibrated_anchor()
    blob = " ".join(cfg.provenance)
    assert "extrapolated" in blob and "32GB" in blob          # capacity beyond measured 16GB
    assert "simulated" in blob and "edge" in blob             # forward topology
    assert "NPU" in blob                                       # no RKNPU2 silicon


def test_pipeline_knob_default_off_is_measured():
    # default (pipeline off) = single-accelerator serial, the measured all-AIPU path -> calibrated.
    cfg = SimConfig.from_dict({"workload": {"model": "llama-3.2-1b"}})
    assert cfg.pipeline is False
    assert cfg.is_calibrated_anchor()
    # turning pipeline ON is a forward-looking SIMULATED mode (the measured Card 1c single-core
    # decode shows no cross-op overlap) -> flagged simulated, not a calibrated anchor.
    cfg2 = SimConfig.from_dict({"workload": {"model": "llama-3.2-1b"},
                                "tunables": {"pipeline": True}})
    assert cfg2.pipeline is True
    assert not cfg2.is_calibrated_anchor()
    assert any("pipeline" in p for p in cfg2.provenance)


def test_batch_must_be_one():
    try:
        SimConfig.from_dict({"workload": {"model": "llama-3.2-1b", "batch": 4}})
    except ValueError:
        return
    raise AssertionError("batch!=1 not rejected")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok {fn.__name__}")
    print(f"\n{len(fns)} config tests passed.")
