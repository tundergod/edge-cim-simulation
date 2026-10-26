"""ADR-0008 — the heterogeneous (unit x category) pricing contract of Platform.price / supports.

(a) supports() matrix == the contract (cim/gpu/npu/cpu/mem x dag.CATEGORIES + 'merge');
(b) every supported combo returns latency_us >= 0 + a provenance string on a representative node,
    for the legacy platform and BOTH profiles; every unsupported combo raises NotImplementedError;
(c) 'merge' = wl.extra['fixed_us'] on ANY unit; fixed_us is returned as its OWN field (a serial pre-op
    delay, ADR-0009 rev F1) on every category, latency_us stays the compute term (0 default);
(d) attention composite rule on gpu/npu/cim/cpu: priced once on the rep, members 0, ungrouped bmm raises,
    priced from the node's own heads/kv (a head- or kv-split sub-node prices by its own shape);
(e) pricing depends only on (unit, category, wl, pricing_group): ids and unknown extras are ignored;
(f) the new CPU GEMM / sampling and the NPU unit price sanely.

    .venv/bin/pytest tests/test_pricing_matrix.py      (from the repo root)
"""
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "trace_export"))
from simulator.models.engine import Workload  # noqa: E402
from simulator.runtime.dag import OpNode, CATEGORIES  # noqa: E402
from simulator.runtime.platform import Platform  # noqa: E402

UNITS = ("cim", "gpu", "npu", "cpu", "mem")
ALL_CATS = tuple(CATEGORIES) + ("merge",)
CONTRACT = {
    "gpu": {"matmul", "attention", "norm", "rope", "ffn", "softmax", "residual", "convert", "merge"},
    "npu": {"matmul", "attention", "norm", "rope", "ffn", "softmax", "residual", "convert", "merge"},
    "cpu": {"matmul", "attention", "norm", "rope", "ffn", "softmax", "residual", "sampling", "convert", "merge"},
    "cim": {"matmul", "attention", "convert", "merge"},
    "mem": {"embedding", "kv_cache", "convert", "merge"},
}
MODEL = "llama-3.2-1b"


def _plats():
    return {"legacy": Platform(MODEL),
            "sd8gen3_cim": Platform(MODEL, soc_profile="sd8gen3_cim"),
            "rk3588_cim": Platform(MODEL, soc_profile="rk3588_cim")}


def _node(cat, unit, **kw):
    """A representative node per category (shapes of Llama-3.2-1B decode at kv=512)."""
    base = dict(id=3, category=cat, unit=unit)
    if cat == "matmul":
        wl = Workload(op="matmul", M=kw.get("M", 1), K=2048, N=2048, dtype="int8", nbytes=4_194_304)
    elif cat == "attention":
        wl = Workload(op="attention", M=kw.get("M", 1), kv=kw.get("kv", 512), heads=kw.get("heads", 32),
                      K=64, dtype="fp16", nbytes=98_304, extra={"hd": 64})
        base["pricing_group"] = kw.get("pricing_group", 3)
    elif cat in ("norm", "rope", "ffn", "softmax", "residual", "sampling"):
        wl = Workload(op=cat, kv=512 if cat == "softmax" else 0, nbytes=8192, dtype="fp16",
                      extra={"model": MODEL, "category": cat, "aten": "aten.x"})
    elif cat == "convert":
        wl = Workload(op="convert", nbytes=6144, dtype="fp16", extra={"from": "int8", "to": "fp16"})
    elif cat == "merge":
        wl = Workload(op="merge", extra={"fixed_us": kw.get("fixed_us", 0.0)})
    elif cat == "kv_cache":
        wl = Workload(op="kv_append", nbytes=4096, dtype="int8")
    else:  # embedding
        wl = Workload(op="stream", nbytes=4096, dtype="fp16")
    return OpNode(wl=wl, **base)


# ---- (a) ---------------------------------------------------------------------------------------
def test_supports_matrix_matches_contract():
    for u in UNITS:
        got = {c for c in ALL_CATS if Platform.supports(u, c)}
        assert got == CONTRACT[u], (u, got ^ CONTRACT[u])
    assert not Platform.supports("tpu", "matmul") and not Platform.supports("gpu", "teleport")


# ---- (b) ---------------------------------------------------------------------------------------
def test_every_supported_combo_prices_and_unsupported_raises():
    for pname, plat in _plats().items():
        for u in UNITS:
            for cat in ALL_CATS:
                n = _node(cat, u)
                if Platform.supports(u, cat):
                    r = plat.price(n)
                    assert set(r) >= {"latency_us", "compute_provenance", "source_model"}, (pname, u, cat)
                    assert isinstance(r["latency_us"], float) and r["latency_us"] >= 0.0, (pname, u, cat, r)
                    assert isinstance(r["compute_provenance"], str) and r["compute_provenance"], (pname, u, cat)
                    assert r["source_model"], (pname, u, cat)
                else:
                    with pytest.raises(NotImplementedError):
                        plat.price(n)
    with pytest.raises(ValueError):
        Platform(MODEL).price(_node("matmul", "bogus"))


def test_profiles_price_gemm_on_all_compute_units_positive():
    for pname, plat in _plats().items():
        for u in ("cim", "gpu", "npu", "cpu"):
            for M in (1, 128, 256):
                assert plat.price(_node("matmul", u, M=M))["latency_us"] > 0, (pname, u, M)


# ---- (c) ---------------------------------------------------------------------------------------
def test_merge_and_fixed_us_hook():
    for pname, plat in _plats().items():
        for u in UNITS:
            assert plat.price(_node("merge", u, fixed_us=37.5)) == {
                "latency_us": 0.0, "fixed_us": 37.5, "source_model": "merge",
                "compute_provenance": "fixed-cost merge/sync (set by scheduler; serial pre-op delay)"}
            assert plat.price(_node("merge", u))["fixed_us"] == 0.0           # default 0
            assert plat.fixed_us(_node("merge", u, fixed_us=5.0)) == 5.0
        for u, cat in (("cim", "matmul"), ("gpu", "matmul"), ("npu", "matmul"), ("cpu", "matmul"),
                       ("gpu", "attention"), ("cpu", "norm"), ("mem", "kv_cache"), ("gpu", "convert")):
            base = plat.price(_node(cat, u))
            assert base["fixed_us"] == 0.0
            n = _node(cat, u)
            n.wl.extra["fixed_us"] = 12.25
            r = plat.price(n)
            assert r["latency_us"] == pytest.approx(base["latency_us"]) and r["fixed_us"] == 12.25, (pname, u, cat)
            assert "fixed_us" in r["compute_provenance"] and plat.compute_us(n) == r["latency_us"]


# ---- (d) ---------------------------------------------------------------------------------------
def test_attention_composite_rule_all_units_and_split_shapes():
    for pname, plat in _plats().items():
        for u in ("gpu", "npu", "cim", "cpu"):
            rep = _node("attention", u, pricing_group=3)
            member = _node("attention", u, pricing_group=99)             # grouped under another rep
            r_rep, r_mem = plat.price(rep), plat.price(member)
            if u == "cim" and pname == "legacy":                          # legacy CIM attention = 0 (L4 path)
                assert r_rep["latency_us"] == 0.0 and r_mem["latency_us"] == 0.0
                continue
            assert r_mem["latency_us"] == 0.0 and "grouped" in r_mem["compute_provenance"], (pname, u)
            assert r_rep["latency_us"] > 0, (pname, u, r_rep)
            # split by heads: 8 heads <= 32 heads; split by kv: 128 <= 512 (priced from the node's own shape)
            assert plat.price(_node("attention", u, heads=8))["latency_us"] <= r_rep["latency_us"]
            assert plat.price(_node("attention", u, kv=128))["latency_us"] <= r_rep["latency_us"]
            ungrouped = _node("attention", u, pricing_group=None)
            with pytest.raises(NotImplementedError):
                plat.price(ungrouped)
    # non-bmm attention (scale/mask): cpu -> 0 everywhere (legacy); gpu = the elementwise launch floor on
    # BOTH GPU models (rev F14; the Mali path used to raise)
    scale = OpNode(id=1, category="attention", unit="cpu",
                   wl=Workload(op="attention", extra={"aten": "aten.mul.Tensor", "category": "attention"}))
    assert Platform(MODEL).price(scale)["latency_us"] == 0.0
    scale.unit = "gpu"
    for pname, plat in _plats().items():
        assert plat.price(scale)["source_model"] == "elementwise_floor", pname


# ---- (e) ---------------------------------------------------------------------------------------
def test_pricing_ignores_ids_and_unknown_extras():
    for pname, plat in _plats().items():
        for u, cat in (("cim", "matmul"), ("gpu", "matmul"), ("npu", "matmul"), ("cpu", "norm"), ("gpu", "attention")):
            a = _node(cat, u)
            b = replace(a, id=4242, pricing_group=(4242 if a.pricing_group is not None else None),
                        wl=replace(a.wl, extra={**a.wl.extra, "tile_of": 17, "split": "N", "note": "junk"}))
            assert plat.price(a)["latency_us"] == plat.price(b)["latency_us"], (pname, u, cat)


# ---- (f) ---------------------------------------------------------------------------------------
def test_cpu_gemm_sampling_and_npu_units_price():
    plat = Platform(MODEL)
    g1 = plat.price(_node("matmul", "cpu", M=1))
    g256 = plat.price(_node("matmul", "cpu", M=256))
    assert g1["source_model"] == "m4_cpu_gemm" and 0 < g1["latency_us"] < g256["latency_us"]
    assert "simulated" in g1["compute_provenance"]
    s = plat.price(_node("sampling", "cpu"))
    assert s["source_model"] == "m4_cpu" and s["latency_us"] > 0
    n = plat.price(_node("matmul", "npu"))
    assert n["source_model"] == "m4_npu" and n["latency_us"] > 0 and "simulated" in n["compute_provenance"]
    e = plat.price(_node("norm", "gpu"))
    assert e["source_model"] == "elementwise_floor" and e["latency_us"] == plat.launch_us["gpu"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
