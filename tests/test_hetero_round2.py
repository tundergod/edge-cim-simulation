"""ADR-0008/0009 "Revisions 2026-10-23 round 2" (after matrix v1) — one regression test per change:

  G1  CPU GEMM model: per-kernel floor (fused-chain rule on CPU elementwise ops, profile path only),
      `cpu_only` scheduler, END-TO-END refit to the HeteroInfer Fig.12 CPU/GPU decode ratio
      (cpu_only = gpu_only / 1.33 within +/-10 % on llama-3.2-1b AND llama-3.1-8b through the real
      Platform + runner).
  G2  Solver: host=auto PORTFOLIO (solver plan per host + every single-unit plan, engine-ranked;
      "ours >= best single-unit baseline" by construction), `plan_source` surfaced; refinement passes
      (deferred / neighbours hand-back charging) so a chain of ops each cheapest on the NPU becomes
      one NPU run with one sync-in/out pair.
  G3  `platform.cim_weight_bytes_per_param` knob; price_table `--cim-n-cores`.

    .venv/bin/pytest tests/test_hetero_round2.py      (from the repo root)
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools" / "trace_export"))
sys.path.insert(0, str(ROOT / "tools" / "analysis"))
from simulator.models.engine import Workload  # noqa: E402
from simulator.models.m4_cpu import CpuGemmModel  # noqa: E402
from simulator.runtime.config import SimConfig, _KNOWN_SCHEDULERS  # noqa: E402
from simulator.runtime.dag import OpNode, Dag  # noqa: E402
from simulator.runtime.events import run_dag  # noqa: E402
from simulator.runtime.platform import Platform  # noqa: E402
from simulator.runtime.runner import run  # noqa: E402
from simulator.runtime.scheduler import (SCHEDULERS, SingleUnitScheduler, TiledHeteroScheduler,  # noqa: E402
                                         single_support_unit, tag_fused_chains)
from simulator.runtime.solver import SolverContext, _estimate_single  # noqa: E402
from simulator.runtime.workload import build_token_dag  # noqa: E402
from simulator.specs.loader import load_spec  # noqa: E402
from fake_platform import AnalyticFakePlatform  # noqa: E402

MODEL = "llama-3.2-1b"
PROFILE = "sd8gen3_cim"
WL = {"context": 1024, "prefill_len": 256, "decode_len": 128}


class Cfg:
    model = MODEL
    units = {"cim": True, "gpu": True, "npu": True, "cpu": True}
    sync_mode = "fast"
    sync_us_override = None
    partition_grid = 8
    precision_boundary_placement = "consumer"
    host = None
    allow_cpu_compute = None

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _cfg(model, policy, cim=True, **sched):
    return SimConfig.from_dict({"workload": {"model": model, **WL},
                                "platform": {"soc_profile": PROFILE,
                                             "units": {"gpu": True, "npu": True, "cim": cim, "cpu": True}},
                                "scheduler": {"policy": policy, **sched}})


def _mm(nid, unit, M, K, N, deps=(), dtype="int8"):
    return OpNode(id=nid, category="matmul", unit=unit, deps=list(deps), precision=dtype,
                  wl=Workload(op="matmul", M=M, K=K, N=N, dtype=dtype), out_elems=M * N)


def _ew(nid, unit, nbytes=0, deps=(), cat="norm"):
    return OpNode(id=nid, category=cat, unit=unit, deps=list(deps), bytes_streamed=nbytes,
                  mem_domain="dram" if nbytes else "none", precision="fp16", out_elems=max(1, nbytes // 2),
                  wl=Workload(op=cat, nbytes=nbytes, dtype="fp16", extra={"model": MODEL, "category": cat}))


# ---- G1: CPU kernel floor ----------------------------------------------------------------------------
def test_g1_cpu_kernel_floor_per_kernel_and_per_fused_chain_profile_only():
    spec = load_spec(f"soc_{PROFILE}")["units"]["cpu"]["gemm"]
    floor = spec["kernel_floor_us"]
    assert 10.0 <= floor <= 30.0                                   # the launch band the fit searched
    plat, leg = Platform(MODEL, soc_profile=PROFILE), Platform(MODEL)
    assert plat.cpu_kernel_floor_us == floor and leg.cpu_kernel_floor_us == 0.0 and leg.cpu_gemm.kernel_floor_us == 0.0
    # one GEMV kernel = floor + work; attention composite = two kernels
    g = plat.cpu_gemm
    work = 2 * 4096 * 4096 / (g.peak_ops_s("int8") * g.efficiency(1)) * 1e6
    assert plat.compute_us(_mm(0, "cpu", 1, 4096, 4096)) == pytest.approx(floor + work)
    att = OpNode(id=3, category="attention", unit="cpu", pricing_group=3,
                 wl=Workload(op="attention", M=1, kv=512, heads=32, K=64, dtype="fp16", extra={"hd": 64}))
    flops = 2.0 * 2.0 * 32 * 64 * 512
    assert plat.compute_us(att) == pytest.approx(2 * floor + flops / (g.peak_ops_s("fp16") * g.eff) * 1e6)
    # elementwise ops on the CPU: the calibrated m4_cpu cost + the floor once per fused chain head
    a = _ew(0, "cpu", 8192, cat="norm")
    b = _ew(1, "cpu", 8192, deps=[0], cat="rope")
    c = _ew(2, "cpu", 8192, deps=[1], cat="ffn")
    d = _ew(3, "cpu", 8192, deps=[2, 0], cat="residual")       # join -> head; its fan-out from 0 makes b a head too
    dag = Dag([a, b, c, d])
    assert tag_fused_chains(dag) == 3
    assert [n.wl.extra["fused_head"] for n in dag.nodes] == [True, True, False, True]
    base = {n.id: leg.compute_us(n) for n in dag.nodes}         # legacy: no tag effect, no floor
    assert all(leg.compute_us(n) == base[n.id] for n in dag.nodes)
    assert plat.compute_us(a) == pytest.approx(base[0] + floor) and plat.compute_us(d) == pytest.approx(base[3] + floor)
    assert plat.compute_us(b) == pytest.approx(base[1] + floor) and plat.compute_us(c) == pytest.approx(base[2])
    assert "fused follower" in plat.price(c)["compute_provenance"] and "kernel floor" in plat.price(a)["compute_provenance"]
    # untagged = head; sampling (control plane) never pays
    assert plat.compute_us(_ew(9, "cpu", 8192)) == pytest.approx(leg.compute_us(_ew(9, "cpu", 8192)) + floor)
    samp = OpNode(id=5, category="sampling", unit="cpu", wl=Workload(op="sampling_argmax", N=128256))
    assert plat.compute_us(samp) == leg.compute_us(samp)
    # legacy pricing of the calibrated CPU support path is byte-identical (the L4 gate): the floor is 0
    assert leg.compute_us(_ew(9, "cpu", 8192)) == Platform(MODEL).compute_us(_ew(9, "cpu", 8192))
    with pytest.raises(ValueError):
        CpuGemmModel(load_spec("cpu_rk3588"), kernel_floor_us=-1.0)


def test_g1_cpu_only_scheduler_registered_and_places_everything_on_the_cpu():
    assert "cpu_only" in SCHEDULERS and "cpu_only" in _KNOWN_SCHEDULERS and _KNOWN_SCHEDULERS == set(SCHEDULERS)
    s = SCHEDULERS["cpu_only"]
    assert isinstance(s, SingleUnitScheduler) and s.required_units == ("cpu",) and s.pipeline
    plat = Platform(MODEL, soc_profile=PROFILE)
    dag = s.assign(build_token_dag(MODEL, "decode", 512), Cfg(), platform=plat)
    assert dag.host == "cpu" and dag.plan_source == "single:cpu"
    assert {n.unit for n in dag.nodes} == {"cpu", "mem"}
    assert sum(1 for n in dag.nodes if n.wl.extra.get("sync_us")) == 0        # no cross-unit boundary
    assert all(n.mem_domain == "dram" for n in dag.nodes if n.category in ("matmul", "attention") and "hd" in n.wl.extra)
    assert all(n.bytes_streamed == n.wl.extra["bytes_int8"] - n.wl.K * n.wl.N // 2 for n in dag.nodes if n.category == "matmul")


@pytest.mark.parametrize("model", ["llama-3.2-1b", "llama-3.1-8b"])
def test_g1_cpu_only_is_1_33x_slower_than_gpu_only_end_to_end(model):
    """HeteroInfer Fig.12: llama.cpp (CPU) decode is 1.33x slower than MNN (GPU) end to end
    (19.38/14.20 s InternLM-1.8B -> llama-3.2-1b proxy; 63.39/48.76 s Llama-7B -> llama-3.1-8b proxy).
    The fit (fit_cpu_gemv.py) holds the ratio within +/-10 % on BOTH models through the real runner;
    the pre-fit model inverted it (cpu_only 30.9 vs gpu_only 21.3 tok/s at 1B, matrix v1 anomaly 1)."""
    cpu, gpu = run(_cfg(model, "cpu_only")), run(_cfg(model, "gpu_only"))
    ratio = gpu["tok_s"] / cpu["tok_s"]
    assert abs(ratio / 1.33 - 1) <= 0.10, (model, ratio)
    assert cpu["tok_s"] < gpu["tok_s"]                                           # the inversion is gone
    assert cpu["plan_source"] == "single:cpu" and cpu["host"] == "cpu" and cpu["unit_utilisation"]["cpu"] > 0.5
    assert cpu["sync_count_per_decode_token"] == 0 and abs(cpu["achieved_decode_bandwidth_GBs"] - 43.0) <= 0.43 * 2
    assert cpu["cpu_kernel_floor_us"] == load_spec(f"soc_{PROFILE}")["units"]["cpu"]["gemm"]["kernel_floor_us"]
    assert any("simulated" in p for p in cpu["provenance"])                      # never a calibrated anchor
    rep = ROOT / "validation/reports/hetero/cpu_gemv_fit.json"
    assert rep.exists()


# ---- G2: portfolio + refinement ---------------------------------------------------------------------
def test_g2_single_support_unit_rule():
    plat = Platform(MODEL, soc_profile=PROFILE)
    assert [single_support_unit(u, plat) for u in ("gpu", "npu", "cpu", "cim")] == ["gpu", "npu", "cpu", "cpu"]


def test_g2_refinement_makes_a_chain_of_npu_cheapest_ops_one_npu_run():
    """Three chained matmuls each cheaper on the NPU by 12.5 us with a 10 us sync: the per-op greedy
    (hand-back charged on every non-host placement: npu = 10 + 12.5 + 10 > gpu 25) keeps all three on
    the GPU host; the deferred pass charges each boundary once at its consumer -> one NPU run with one
    sync-in/out pair (2 syncs, not 3 pairs), and the engine confirms it is faster."""
    plat = AnalyticFakePlatform(rates={"gpu": 1.0e6, "npu": 64.0e6}, bw_GBs=1e6, knee_GBs=1e6, byte_rates={"gpu": 1e9})
    K = N = 5000                                                   # gpu 25 us; npu (M padded to 32) 12.5 us
    def chain():
        return Dag([_ew(0, None, 0, cat="norm"),
                    _mm(1, None, 1, K, N, deps=[0]), _mm(2, None, 1, K, N, deps=[1]), _mm(3, None, 1, K, N, deps=[2]),
                    _ew(4, None, 0, deps=[3], cat="norm")])
    kw = dict(units=("gpu", "npu"), support_unit="gpu", matmul_dims=(), split_attention=False, split_support=False, host="gpu")
    greedy = TiledHeteroScheduler(refine=False, **kw).assign(chain(), Cfg(), platform=plat)
    assert [greedy[i].unit for i in (1, 2, 3)] == ["gpu"] * 3 and greedy.plan_pass == "greedy"
    refined = TiledHeteroScheduler(refine=True, **kw).assign(chain(), Cfg(), platform=plat)
    assert [refined[i].unit for i in (1, 2, 3)] == ["npu"] * 3 and refined.plan_pass in ("deferred", "neighbours")
    # ONE sync-in/out pair: carried by the consumer-side int8<->fp16 converts at the chain's entry (on the
    # NPU) and at its exit (back on the GPU host), nothing in between
    syncs = [n for n in refined.nodes if n.wl.extra.get("sync_us")]
    assert len(syncs) == 2 and [n.unit for n in syncs] == ["npu", "gpu"] and all(n.category == "convert" for n in syncs)
    assert sum(1 for n in greedy.nodes if n.wl.extra.get("sync_us")) == 0
    t_g, t_r = run_dag(greedy, plat, plat.bw), run_dag(refined, plat, plat.bw)
    assert t_r < t_g and t_r == pytest.approx(3 * 12.5 + 2 * 10.0) and t_g == pytest.approx(3 * 25.0)
    assert refined.greedy_engine_us == pytest.approx(t_g) and refined.plan_source == "solver:gpu"
    # the estimate rule itself: deferred charges no hand-back when the consumer is still to be decided,
    # 'always' does; a terminal block / a fixed consumer elsewhere keeps the hand-back in every mode
    d = chain()
    d[0].unit = "gpu"
    for mode, expect in (("always", 10.0 + 12.5 + 10.0), ("deferred", 10.0 + 12.5), ("neighbours", 10.0 + 12.5)):
        ctx = SolverContext(platform=plat, units=("gpu", "npu"), host="gpu", sync_us=10.0, handback=mode,
                            fixed_units={0: "gpu", 4: "gpu"})
        assert _estimate_single(ctx, d, [1], "npu") == pytest.approx(expect)
    ctx = SolverContext(platform=plat, units=("gpu", "npu"), host="gpu", sync_us=10.0, handback="deferred",
                        fixed_units={0: "gpu", 4: "gpu"})
    d[1].unit = d[2].unit = "npu"
    assert _estimate_single(ctx, d, [3], "npu") == pytest.approx(12.5 + 10.0)      # fixed gpu consumer: hand-back
    ctx = SolverContext(platform=plat, units=("gpu", "npu"), host="gpu", sync_us=10.0, handback="neighbours",
                        fixed_units={0: "gpu"}, hints={2: "gpu"})
    d[1].unit = "gpu"
    assert _estimate_single(ctx, d, [1], "npu") == pytest.approx(10.0 + 12.5 + 10.0)   # decided consumer elsewhere
    with pytest.raises(ValueError):
        _estimate_single(SolverContext(platform=plat, units=("gpu", "npu"), host="gpu", sync_us=10.0, handback="x"), d, [1], "npu")
    # idempotent + deterministic
    again = TiledHeteroScheduler(refine=True, **kw).assign(refined, Cfg(), platform=plat)
    snap = lambda g: [(n.id, n.unit, n.wl.extra.get("fixed_us")) for n in g.nodes]   # noqa: E731
    assert snap(again) == snap(refined) and again.plan_pass == refined.plan_pass
    assert snap(TiledHeteroScheduler(refine=True, **kw).assign(chain(), Cfg(), platform=plat)) == snap(refined)


def test_g2_host_auto_portfolio_is_at_least_the_best_single_unit_plan_real_platform():
    """host=auto evaluates the per-host solver plans AND every single-unit plan with the engine and
    keeps the best: tiled_hetero (CPU allowed) is never slower than gpu/npu/cim/cpu_only under the
    same cfg — by construction, on decode and prefill. `plan_source` says which won."""
    plat = Platform(MODEL, soc_profile=PROFILE)
    cfg = Cfg(host="auto", allow_cpu_compute=True)
    for phase, L in (("decode", 320), ("prefill", 256)):
        ours = SCHEDULERS["tiled_hetero"].assign(build_token_dag(MODEL, phase, L), cfg, platform=plat)
        t_ours = run_dag(ours, plat, plat.bw)
        singles = {}
        for u in ("gpu", "npu", "cim", "cpu"):
            d = SCHEDULERS[f"{u}_only"].assign(build_token_dag(MODEL, phase, L), cfg, platform=plat)
            singles[u] = run_dag(d, plat, plat.bw)
            assert ours.host_candidates[f"single:{u}"]["engine_us"] == pytest.approx(singles[u])
        assert t_ours <= min(singles.values()) * (1 + 1e-9), (phase, t_ours, singles)
        kind, unit = ours.plan_source.split(":")
        assert kind in ("solver", "single") and unit == ours.host
        assert set(ours.host_candidates) == {f"{k}:{u}" for k in ("solver", "single") for u in ("gpu", "npu", "cim", "cpu")}
        if kind == "single":
            assert t_ours == pytest.approx(singles[unit])
        print(phase, ours.plan_source, ours.plan_pass, round(t_ours), {u: round(t) for u, t in singles.items()})


def test_g2_runner_surfaces_plan_source():
    r = run(_cfg(MODEL, "tiled_hetero", host="auto", allow_cpu_compute=True))
    for k in ("plan_source", "prefill_plan_source"):
        assert r[k].split(":")[0] in ("solver", "single") and r[k].split(":")[1] in ("gpu", "npu", "cim", "cpu")
    assert r["plan_source"].endswith(r["host"]) and r["prefill_plan_source"].endswith(r["prefill_host"])
    assert r["plan_pass"] in (None, "greedy", "deferred", "neighbours")
    assert set(r["host_candidates_est_us"]) == {f"{k}:{u}" for k in ("solver", "single") for u in ("gpu", "npu", "cim", "cpu")}
    for pol in ("gpu_only", "cim_only"):
        q = run(_cfg(MODEL, pol))
        assert q["plan_source"] == q["prefill_plan_source"] == f"single:{pol.split('_')[0]}" and q["plan_pass"] is None
    assert run(_cfg(MODEL, "heteroinfer"))["plan_source"] == "solver:gpu"
    assert run(SimConfig.from_dict({"workload": {"model": MODEL, "context": 1024}}))["plan_source"] is None


# ---- G3: cim_weight_bytes_per_param + price_table --cim-n-cores ----------------------------------
def test_g3_cim_weight_bytes_per_param_knob():
    p1, p05 = Platform(MODEL, soc_profile=PROFILE), Platform(MODEL, soc_profile=PROFILE, cim_weight_bytes_per_param=0.5)
    assert p1.cim_weight_bytes_per_param == 1.0 == p1.weight_bytes_per_param["cim"]
    assert p05.cim_weight_bytes_per_param == 0.5 == p05.weight_bytes_per_param["cim"] and p05.weight_bytes_per_param["gpu"] == 0.5
    base = build_token_dag(MODEL, "decode", 512)
    int8 = {n.id: n.bytes_streamed for n in base.nodes if n.category == "matmul"}
    d = SCHEDULERS["cim_only"].assign(base, Cfg(), platform=p05)
    assert all(n.bytes_streamed == int8[n.id] - n.wl.K * n.wl.N // 2 for n in d.nodes if n.category == "matmul")
    assert p05.compute_us(_mm(0, "cim", 1, 2048, 2048)) == p1.compute_us(_mm(0, "cim", 1, 2048, 2048))   # bytes only
    assert Platform(MODEL, cim_weight_bytes_per_param=0.5).weight_bytes_per_param["cim"] == 1.0          # legacy: ignored
    with pytest.raises(ValueError):
        Platform(MODEL, soc_profile=PROFILE, cim_weight_bytes_per_param=0)
    cfg = _cfg(MODEL, "cim_only")
    cfg05 = SimConfig.from_dict({"workload": {"model": MODEL, **WL},
                                 "platform": {"soc_profile": PROFILE, "cim_weight_bytes_per_param": 0.5},
                                 "scheduler": {"policy": "cim_only"}})
    assert any("cim_weight_bytes_per_param override = 0.5" in p and "design option" in p for p in cfg05.provenance)
    assert not any("cim_weight_bytes_per_param" in p for p in cfg.provenance)
    assert not any("cim_weight_bytes_per_param" in p for p in SimConfig.from_dict(
        {"workload": {"model": MODEL}, "platform": {"soc_profile": PROFILE, "cim_weight_bytes_per_param": 1.0}}).provenance)
    for bad in ({"cim_weight_bytes_per_param": 0}, {"cim_weight_bytes_per_param": "0.5"}, {"cim_weight_bytes_per_param": True},
                {"cim_weight_bytes_per_param": 5}):
        with pytest.raises(ValueError):
            SimConfig.from_dict({"workload": {"model": MODEL}, "platform": {"soc_profile": PROFILE, **bad}})
    with pytest.raises(ValueError):                                             # needs a profile
        SimConfig.from_dict({"workload": {"model": MODEL}, "platform": {"cim_weight_bytes_per_param": 0.5}})
    r1, r05 = run(cfg), run(cfg05)
    assert r1["cim_weight_bytes_per_param"] == 1.0 and r05["cim_weight_bytes_per_param"] == 0.5 and r1["cim_n_cores"] == 4
    assert r05["tok_s"] > r1["tok_s"]                                            # half the weight bytes on the pool


def test_g3_price_table_accepts_cim_n_cores():
    import price_table
    t8 = price_table.table(PROFILE, models=[MODEL], cim_n_cores=8)
    t4 = price_table.table(PROFILE, models=[MODEL])
    assert t8["cim_n_cores"] == 8 and not t8["cim_n_cores_is_profile_default"]
    assert t4["cim_n_cores"] == 4 and t4["cim_n_cores_is_profile_default"]
    pre8 = next(r for r in t8["rows"] if r["phase"] == "prefill" and r["category"] == "matmul")
    pre4 = next(r for r in t4["rows"] if r["phase"] == "prefill" and r["category"] == "matmul" and r["op"] == pre8["op"])
    assert pre8["units"]["cim"]["latency_us"] == pytest.approx(pre4["units"]["cim"]["latency_us"] / 2, rel=1e-3)
    assert pre8["units"]["gpu"]["latency_us"] == pre4["units"]["gpu"]["latency_us"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
