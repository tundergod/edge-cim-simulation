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


def test_energy_excludes_cpu_cache_from_dram():
    # latency excludes cpu_cache bytes from the DRAM pool; energy must be consistent —
    # a CPU-support op's on-chip bytes must NOT be charged DRAM energy (only its cpu energy).
    from simulator.runtime.platform import Platform
    from simulator.runtime.dag import OpNode
    from simulator.models.engine import Workload
    plat = Platform("llama-3.2-1b")
    cache = OpNode(id=0, category="norm", wl=Workload(op="norm", nbytes=10000), unit="cpu",
                   bytes_streamed=10000, mem_domain="cpu_cache")
    dram = OpNode(id=1, category="matmul", wl=Workload(op="matmul", M=1, K=2048, N=2048),
                  unit="cim", bytes_streamed=10000, mem_domain="dram")
    e_cache = plat.energy_J(cache, plat.compute_us(cache))
    e_dram = plat.energy_J(dram, plat.compute_us(dram))
    assert e_cache == plat.energy.cpu_J(plat.compute_us(cache))   # ONLY cpu energy, no dram term
    assert e_dram >= plat.energy.dram_J(10000)                    # dram node IS charged dram energy


def test_gpu_attention_only_prices_bmm():
    # the GPU attn_bmm_us model is for the QK^T/S·V bmm ONLY; scale/mask elementwise
    # attention must NOT be priced as a bmm — fail loud (group-aware composite is 2.2b R2).
    from simulator.runtime.platform import Platform
    from simulator.runtime.dag import OpNode
    from simulator.models.engine import Workload
    plat = Platform("llama-3.2-1b")
    bmm = OpNode(id=0, category="attention", unit="gpu", pricing_group=0,   # grouped rep (2.2b R2)
                 wl=Workload(op="attention", kv=512, heads=32, K=64, dtype="fp16", extra={"hd": 64}))
    p = plat.price(bmm)
    assert p["latency_us"] > 0 and p["source_model"] == "m4_gpu"
    scale = OpNode(id=1, category="attention", unit="gpu",
                   wl=Workload(op="attention", kv=0, extra={"aten": "aten.mul.Tensor", "category": "attention"}))
    try:
        plat.price(scale)
    except NotImplementedError:
        return
    raise AssertionError("non-bmm GPU attention (scale/mask) was priced as a bmm, not fail-loud")


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


def test_cimhetero_requires_cim_gpu_cpu_units():
    # an impossible config (cim_hetero needs the GPU for attention) must fail loud, not
    # silently return a plausible tok/s.
    cfg = SimConfig.from_dict({"workload": {"model": "llama-3.2-1b"},
                               "scheduler": {"policy": "cim_hetero"},
                               "platform": {"units": {"cim": True, "gpu": False, "cpu": True}}})
    try:
        run(cfg)
    except ValueError:
        return
    raise AssertionError("cim_hetero with gpu disabled not rejected")


def test_run_revalidates_mutated_config():
    # SimConfig is a mutable dataclass; run() (the public boundary) must re-validate, not
    # silently emit metrics for a config broken AFTER construction.
    cfg = _cfg("llama-3.2-1b")
    cfg.batch = 4
    try:
        run(cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("run() did not re-validate a mutated batch")
    cfg2 = _cfg("llama-3.2-1b")
    cfg2.precision_boundary_placement = "prodcer"
    try:
        run(cfg2)
    except ValueError:
        return
    raise AssertionError("run() did not re-validate a mutated precision_boundary_placement")


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
