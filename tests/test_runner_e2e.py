"""Phase 2.1 — end-to-end runner sanity (simulator/runtime/runner.py).

Not a validation gate (that is validate_e2e_l4.py); just structural sanity:
positive tok/s, bigger model slower (monotonic), calibrated-anchor flag, metrics
shape. Run from repo root.

    .venv/bin/pytest tests/test_runner_e2e.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.runtime.config import SimConfig  # noqa: E402
from simulator.runtime.runner import run  # noqa: E402


def _cfg(model):
    return SimConfig.from_dict({"workload": {"model": model, "context": 1024},
                                "platform": {"memory_spec": "mem_lpddr4x", "topology": "cim_topo_card"},
                                "scheduler": {"policy": "all_cim"}})


def test_runner_positive_and_anchored():
    r = run(_cfg("llama-3.2-1b"))
    assert r["tok_s"] > 0 and r["decode_token_us"] > 0
    assert r["energy_per_token_J"] > 0 and len(r["energy_band_J"]) == 2
    assert r["calibrated_anchor"] is True
    assert r["provenance"] == []
    assert "concurrency_off" in r["ablations"]


def test_bigger_model_slower():
    t1 = run(_cfg("llama-3.2-1b"))["tok_s"]
    t3 = run(_cfg("llama-3.2-3b"))["tok_s"]
    t8 = run(_cfg("llama-3.1-8b"))["tok_s"]
    assert t1 > t3 > t8 > 0          # decode tok/s falls with model size (more weight bytes)


def test_capacity_feasibility_gate():
    # 8B INT8 weights (~8 GB) cannot fit in 1 GB or 4 GB -> fail-loud (not "calibrated")
    for cap in (1, 4):
        cfg = SimConfig.from_dict({"workload": {"model": "llama-3.1-8b"},
                                   "platform": {"memory_capacity_GB": cap}})
        try:
            run(cfg)
        except ValueError:
            continue
        raise AssertionError(f"infeasible 8B @ {cap}GB not rejected")
    # 16 GB fits -> runs, reports footprint
    r = run(SimConfig.from_dict({"workload": {"model": "llama-3.1-8b"},
                                 "platform": {"memory_capacity_GB": 16}}))
    assert r["model_footprint_GB"] > 0 and r["tok_s"] > 0


def test_per_op_provenance_summary():
    # #55: every op exposes WHICH unit model priced its compute (source_model) + a
    # compute_provenance string; the bound (compute vs memory) is the ENGINE's max() decision.
    r = run(_cfg("llama-3.2-1b"))
    prov = r["op_provenance"]
    assert prov
    for e in prov:
        assert e["source_model"] and e["compute_provenance"]
        assert e["bound"]["compute"] + e["bound"]["memory"] == e["count"]
    # matmul priced by the CIM tile model; CPU-support by m4_cpu; mem ops unmodeled
    assert any(e["category"] == "matmul" and e["source_model"] == "m1_cim_tile" for e in prov)
    assert any(e["source_model"] == "m4_cpu" for e in prov)
    assert any(e["source_model"] == "none" for e in prov)


def test_platform_price_returns_provenance():
    from simulator.runtime.platform import Platform
    from simulator.runtime.dag import OpNode
    from simulator.models.engine import Workload
    plat = Platform("llama-3.2-1b")
    mm = OpNode(id=0, category="matmul", wl=Workload(op="matmul", M=1, K=2048, N=2048), unit="cim")
    p = plat.price(mm)
    assert set(p) >= {"latency_us", "compute_provenance", "source_model"}
    assert p["source_model"] == "m1_cim_tile" and p["latency_us"] > 0
    assert plat.compute_us(mm) == p["latency_us"]   # compute_us delegates to price
    bad = OpNode(id=1, category="matmul", wl=Workload(op="matmul"), unit="bogus")
    try:
        plat.price(bad)
    except ValueError:
        return
    raise AssertionError("price did not reject unknown unit")


def test_platform_rejects_unknown_unit():
    from simulator.runtime.platform import Platform
    from simulator.runtime.dag import OpNode
    from simulator.models.engine import Workload
    plat = Platform("llama-3.2-1b")
    bad = OpNode(id=0, category="matmul", wl=Workload(op="matmul", M=1, K=64, N=64), unit="bogus")
    try:
        plat.compute_us(bad)
    except ValueError:
        return
    raise AssertionError("unknown unit not rejected by platform")


def test_unknown_scheduler_rejected():
    cfg = _cfg("llama-3.2-1b")
    cfg.scheduler = "nope"
    try:
        run(cfg)
    except ValueError:
        return
    raise AssertionError("unknown scheduler not rejected")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok {fn.__name__}")
    print(f"\n{len(fns)} runner tests passed.")
