"""ADR-0008/0009 revisions 2026-10-23 (fixes F1-F15 after the independent review) — one regression
test per fix, through the REAL Platform(soc_profile="sd8gen3_cim") wherever the fix lives in a unit
model or the profile, and through toy DAGs where the fix is an engine/solver rule.

    .venv/bin/pytest tests/test_hetero_fixes.py      (from the repo root)
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools" / "trace_export"))
from simulator.models.engine import Workload  # noqa: E402
from simulator.models.m1_cim_tile import CimTileModel, CimAttentionModel  # noqa: E402
from simulator.runtime.config import SimConfig  # noqa: E402
from simulator.runtime.dag import OpNode, Dag  # noqa: E402
from simulator.runtime.events import run_dag, run_serial  # noqa: E402
from simulator.runtime.platform import Platform  # noqa: E402
from simulator.runtime.resources import SharedBandwidth  # noqa: E402
from simulator.runtime.scheduler import SCHEDULERS, HeteroInferScheduler, tag_fused_chains, apply_unit_bytes  # noqa: E402
from simulator.runtime.solver import SolverContext, solve, _node_us, _estimate_single  # noqa: E402
from simulator.runtime.workload import build_token_dag  # noqa: E402
from simulator.runtime import tiling  # noqa: E402
from simulator.specs.loader import load_spec  # noqa: E402

MODEL = "llama-3.2-1b"
PROFILE = "sd8gen3_cim"


class Cfg:
    model = MODEL
    units = {"cim": True, "gpu": True, "npu": True, "cpu": True}
    sync_mode = "fast"
    sync_us_override = None
    partition_grid = 8
    precision_boundary_placement = "consumer"
    host = None


def _mm(unit, M, K, N, **extra):
    return OpNode(id=7, category="matmul", unit=unit,
                  wl=Workload(op="matmul", M=M, K=K, N=N, dtype="int8", extra=extra))


def _ew(nid, unit, nbytes, deps=(), fixed=None, cat="norm"):
    extra = {"model": MODEL, "category": cat, "aten": "aten.x"}
    if fixed is not None:
        extra["fixed_us"] = fixed
    return OpNode(id=nid, category=cat, unit=unit, deps=list(deps), bytes_streamed=nbytes,
                  mem_domain="dram" if nbytes else "none", precision="fp16", out_elems=nbytes // 2,
                  wl=Workload(op=cat, nbytes=nbytes, dtype="fp16", extra=extra))


# ---- F1: sync = serial pre-op delay, engine == solver -------------------------------------------
def test_f1_fixed_us_is_a_serial_pre_op_delay_in_engine_and_solver():
    plat = Platform(MODEL, soc_profile=PROFILE)
    n = _ew(0, "gpu", 4_000_000, fixed=10.0)                        # 4 MB elementwise + 10 us sync
    pr = plat.price(n)
    c, m = pr["latency_us"], plat.bw.stream_us(4_000_000, 1)
    assert pr["fixed_us"] == 10.0 and c == plat.launch_us["gpu"] and m == pytest.approx(93.0233, abs=1e-3)
    expect = 10.0 + max(c, m)
    for pipeline in (True, False):
        st = {}
        t = run_dag(Dag([n]), plat, plat.bw, pipeline=pipeline, stats=st)
        assert t == pytest.approx(expect) and st["unit_busy_us"]["gpu"] == pytest.approx(10.0 + c)
        assert st["critical_path_us"] == pytest.approx(expect)
    ctx = SolverContext(platform=plat, units=("gpu", "npu"), host="gpu", sync_us=10.0)
    assert _node_us(ctx, tiling._copy(n, unit="gpu")) == pytest.approx(max(c, m))   # template: sync stripped ...
    p = _mm("npu", 1, 2048, 2048); p.id, p.bytes_streamed, p.mem_domain, p.precision = 0, 4_198_400, "dram", "int8"
    q = _ew(1, None, 4_000_000, deps=[0])
    assert _estimate_single(ctx, Dag([p, q]), [1], "gpu") == pytest.approx(expect)  # ... and re-added once
    # a merge node is fixed_us + its reduction bytes, nothing hidden under max()
    merge = OpNode(id=0, category="merge", unit="gpu", bytes_streamed=430_000, mem_domain="dram",
                   wl=Workload(op="merge", nbytes=430_000, extra={"fixed_us": 10.0, "sync_us": 10.0}))
    assert run_serial(Dag([merge]), plat, plat.bw) == pytest.approx(10.0 + plat.bw.stream_us(430_000, 1))


def test_f1_charged_sync_is_fully_visible_on_heteroinfer_decode():
    plat = Platform(MODEL, soc_profile=PROFILE)
    dag = SCHEDULERS["heteroinfer"].assign(build_token_dag(MODEL, "decode", 512), Cfg(), platform=plat)
    charged = sum(n.wl.extra.get("sync_us", 0.0) for n in dag.nodes)
    assert charged > 0
    t_sync = run_serial(dag, plat, plat.bw)
    for n in dag.nodes:                       # a merge carries its own sync from split_node: zero it too
        if n.category == "merge":
            n.wl.extra["sync_us"] = 0.0
    tiling.apply_sync(dag, 0.0)
    t_free = run_serial(dag, plat, plat.bw)
    assert t_sync - t_free == pytest.approx(charged, rel=1e-6)        # serial: every charged us is visible


# ---- F2 / F10: one memory model, per-unit caps, knee across units ------------------------------
def test_f2_f10_per_unit_stream_cap_and_cross_unit_knee():
    plat = Platform(MODEL, soc_profile=PROFILE)
    bw = plat.bw
    assert bw.share({"gpu": 2}) == {"gpu": 21.5}                          # one unit: 43 aggregate however many streams
    assert bw.share({"gpu": 1, "npu": 1}) == {"gpu": 30.0, "npu": 30.0}   # two units: the 60 knee, fair
    assert bw.share({"gpu": 1, "npu": 1, "cim": 1}) == {"gpu": 20.0, "npu": 20.0, "cim": 20.0}
    assert bw.share({"gpu": 3, "npu": 1}) == {"gpu": 10.0, "npu": 30.0}   # water-fill: gpu capped at 30/3, npu 30

    def two(u1, u2):
        a = OpNode(id=0, category="merge", unit=u1, mem_domain="dram", bytes_streamed=4_000_000,
                   wl=Workload(op="merge", extra={"fixed_us": 0.0, "sync_us": 0.0}))
        b = OpNode(id=1, category="merge", unit=u2, mem_domain="dram", bytes_streamed=4_000_000,
                   wl=Workload(op="merge", extra={"fixed_us": 0.0, "sync_us": 0.0}))
        st = {}
        t = run_dag(Dag([a, b]), plat, bw, stats=st)
        return 8e6 / t * 1e-3, st["achieved_GBs"]
    assert two("gpu", "gpu") == (pytest.approx(43.0), pytest.approx(43.0))
    assert two("gpu", "npu") == (pytest.approx(60.0), pytest.approx(60.0))
    # legacy (no profile): the uniform fair-share, byte-identical to Phase 2.1
    leg = SharedBandwidth(24.2)
    assert leg.unit_caps is None and leg.share({"gpu": 1, "npu": 1}) == {"gpu": 12.1, "npu": 12.1}
    # a P1 decode GEMV split across gpu+npu streams at the knee, each unit alone at the wall
    g = _mm("gpu", 1, 4096, 4096); g.id, g.mem_domain, g.precision = 0, "dram", "int8"
    g.bytes_streamed = tiling.unit_bytes(OpNode(id=0, category="matmul", wl=g.wl, bytes_streamed=4096 * 4096 + 8192), 0.5)
    n = _mm("npu", 1, 4096, 4096); n.id, n.mem_domain, n.precision = 1, "dram", "int8"
    n.bytes_streamed = 4096 * 4096 + 8192
    st = {}
    run_dag(Dag([g, n]), plat, bw, stats=st)
    assert 43.0 < st["achieved_GBs"] <= 60.0 + 1e-6
    # no unit model carries a DRAM term any more: compute of a GEMV is independent of the pool wall
    fast = Platform(MODEL, soc_profile=PROFILE, knee_GBs=200.0)
    for u in ("gpu", "npu", "cim", "cpu"):
        assert fast.compute_us(_mm(u, 1, 4096, 4096)) == plat.compute_us(_mm(u, 1, 4096, 4096))


# ---- F3: Table 3 through the REAL platform + solver ---------------------------------------------
def _dag(M, K, N):
    return Dag([OpNode(id=0, category="norm", wl=Workload(op="norm", nbytes=0), unit="gpu", mem_domain="none",
                       precision="fp16"),
                OpNode(id=1, category="matmul", wl=Workload(op="matmul", M=M, K=K, N=N, dtype="int8"), deps=[0],
                       bytes_streamed=M * K + K * N + M * N, precision="int8", out_elems=M * N)])


def test_table3_decisions_through_real_platform():
    """HeteroInfer Table 3 (8 Gen 3) decisions from the sd8gen3_cim models through the solver with the
    profile's minimum sub-tensor rule (units.npu.min_split_rows = 1024)."""
    plat = Platform("llama-3.1-8b", soc_profile=PROFILE)

    class C(Cfg):
        model = None
    sched = HeteroInferScheduler()
    rows = [  # (M, K, N, check)                                      paper
        (1, 4096, 4096, lambda p, g: p["strategy"] == "weight_centric" and g >= 0.5),            # 1:1
        (1, 4096, 28672, lambda p, g: p["strategy"] == "weight_centric" and 0.5 < g < 1.0),       # 3:1
        (1, 14336, 4096, lambda p, g: p["strategy"] == "gpu_only" or g >= 7 / 8),                  # GPU-only
        (128, 4096, 4096, lambda p, g: p["strategy"] == "npu_only"),                               # NPU-only
        (256, 14336, 4096, lambda p, g: p["strategy"] == "weight_centric" and abs((1 - g) - 0.6) <= 1 / 8 + 1e-9),  # 2:3
    ]
    got = {}
    for M, K, N, ok in rows:
        dag = sched.assign(_dag(M, K, N), C(), platform=plat)
        p = dag.plans[1]
        g = sum(f for u, f in p["parts"] if u == "gpu")
        got[(M, K, N)] = (p["strategy"], round(g, 3))
        assert ok(p, g), (M, K, N, p["strategy"], p["parts"], p.get("T_est_us"))
        if p["dim"] is not None:
            assert min(s for (u, _), s in zip(p["parts"], p["sizes"]) if u == "npu") >= plat.min_split_rows["npu"]
    print("Table 3 via real platform:", got)
    # decode is GPU-primary at the paper's shapes: the GPU share never drops below the NPU's
    assert all(g >= 0.5 for (M, _, _), (_, g) in got.items() if M == 1)


# ---- F4: launch floor per fused elementwise chain -----------------------------------------------
def test_f4_launch_floor_once_per_fused_chain():
    plat = Platform(MODEL, soc_profile=PROFILE)
    floor = plat.launch_us["gpu"]
    # chain: norm -> rope -> ffn on the GPU (single dependencies), then a residual join (2 deps) = new head
    a = _ew(0, "gpu", 8192, cat="norm")
    b = _ew(1, "gpu", 8192, deps=[0], cat="rope")
    c = _ew(2, "gpu", 8192, deps=[1], cat="ffn")
    d = _ew(3, "npu", 8192, deps=[2], cat="norm")           # other unit: head
    e = _ew(4, "gpu", 8192, deps=[2, 3], cat="residual")    # join: head
    dag = Dag([a, b, c, d, e])
    assert tag_fused_chains(dag) == 3
    assert [n.wl.extra["fused_head"] for n in dag.nodes] == [True, False, False, True, True]
    assert dag[2].wl.extra["fused_group"] == 0 and dag[4].wl.extra["fused_group"] == 4
    assert [plat.compute_us(n) for n in dag.nodes] == [floor, 0.0, 0.0, plat.launch_us["npu"], floor]
    assert tag_fused_chains(dag) == 3                        # idempotent
    # an untagged node pays (the pre-F4 rule), a split-off elementwise node tagged head pays once per part
    assert plat.compute_us(_ew(9, "gpu", 8192)) == floor
    # end to end: gpu_only decode charges far fewer floors than aten nodes (~476 support nodes / token)
    dag = SCHEDULERS["gpu_only"].assign(build_token_dag(MODEL, "decode", 512), Cfg(), platform=plat)
    ew = [n for n in dag.nodes if n.wl.extra.get("fused_head") is not None]
    heads = sum(1 for n in ew if n.wl.extra["fused_head"])
    # measured on the llama-3.2-1b fixture DAG: 309 chain heads of 476 elementwise nodes (the rest are
    # joins / fan-outs: residual adds, rope on q and k from one norm, ...)
    assert len(ew) >= 400 and heads < 0.7 * len(ew), (heads, len(ew))
    assert sum(plat.compute_us(n) for n in ew) == pytest.approx(heads * floor)
    # the floor is the profile parameter
    assert floor == load_spec(f"soc_{PROFILE}")["units"]["gpu"]["elementwise_launch_us"]


# ---- F5: attention vs query length M ------------------------------------------------------------
def test_f5_attention_scales_with_query_rows():
    def attn(plat, unit, M):
        n = OpNode(id=3, category="attention", unit=unit, pricing_group=3,
                   wl=Workload(op="attention", M=M, kv=256, heads=32, K=64, dtype="fp16", extra={"hd": 64}))
        return plat.price(n)["latency_us"]
    p2 = Platform(MODEL, soc_profile="rk3588_cim")
    m1, m64, m256 = (attn(p2, "gpu", M) for M in (1, 64, 256))
    assert m64 == pytest.approx(64 * m1) and m256 == pytest.approx(256 * m1)     # Mali per-row fit x M
    def attn_big(M):                      # above the 2 us dispatch floor: kv=2048, hd=128
        n = OpNode(id=3, category="attention", unit="npu", pricing_group=3,
                   wl=Workload(op="attention", M=M, kv=2048, heads=32, K=128, dtype="fp16", extra={"hd": 128}))
        return p2.price(n)["latency_us"]
    n1, n64, n256 = (attn_big(M) for M in (1, 64, 256))
    assert n64 == pytest.approx(2 * n1) and n256 == pytest.approx(4 * n64)        # ceil(M/32) steps: 1, 2, 8
    p1 = Platform(MODEL, soc_profile=PROFILE)
    s1, s64, s256 = (attn(p1, "npu", M) for M in (1, 64, 256))
    assert s1 < s64 < s256 and (s256 - p1.npu.floor_us) == pytest.approx(4 * (s64 - p1.npu.floor_us))
    g1, g256 = attn(p1, "gpu", 1), attn(p1, "gpu", 256)
    assert g1 == 2 * p1.gpu.launch_us and g256 > g1
    # legacy path (no profile): the Mali fit stays M-independent (byte-identical cim_hetero prefill)
    leg = Platform(MODEL)
    assert attn(leg, "gpu", 128) == attn(leg, "gpu", 1)


# ---- F6: CIM attention heads mode ---------------------------------------------------------------
def test_f6_cim_attention_heads_mode_is_a_profile_axis():
    tile = CimTileModel()
    ser, bat = CimAttentionModel(tile, heads_mode="serial"), CimAttentionModel(tile, heads_mode="batched")
    t_s, _ = ser.composite_us(512, heads=32, hd=128)
    t_b, _ = bat.composite_us(512, heads=32, hd=128)
    floor = ser.qk_a + ser.sv_a
    work = (ser.qk_b + ser.sv_b) * 512
    assert t_s == pytest.approx((floor + work) * 32) and t_b == pytest.approx(floor + work * 32)
    assert t_b < t_s and ser.composite_us(512, heads=1, hd=128)[0] == pytest.approx(bat.composite_us(512, heads=1, hd=128)[0])
    with pytest.raises(ValueError):
        CimAttentionModel(tile, heads_mode="parallel")
    plat = Platform(MODEL, soc_profile=PROFILE)
    assert plat.cim_heads_mode == "serial" == load_spec(f"soc_{PROFILE}")["units"]["cim"]["attention_heads_mode"]
    assert "heads serial" in plat.price(OpNode(id=3, category="attention", unit="cim", pricing_group=3,
                                               wl=Workload(op="attention", kv=512, heads=32, K=128, extra={"hd": 128})))["compute_provenance"]


# ---- F8: one sync table -------------------------------------------------------------------------
def test_f8_single_sync_table():
    for name in ("sd8gen3_cim", "rk3588_cim"):
        assert "sync_us" not in load_spec(f"soc_{name}")
        assert Platform(MODEL, soc_profile=name).sync_us == tiling.SYNC_US
    assert tiling.resolve_sync_us("fast", "decode") == tiling.SYNC_US["fast"]["decode"] == 10.0


# ---- F11: legacy label ----------------------------------------------------------------------------
def test_f11_legacy_schedulers_labelled():
    from simulator.runtime.scheduler import AllCimScheduler
    assert "NO SYNC CHARGED" in AllCimScheduler.__doc__ and "NOT a fair baseline" in AllCimScheduler.__doc__
    assert "out_elems" in tiling._attn_M.__doc__ and "wl.M = Sq" in tiling._attn_M.__doc__


# ---- F12: CIM n_cores physics on the profile path -----------------------------------------------
def test_f12_cim_n_cores_scale_physically_on_profile_path():
    cores = (1, 2, 4, 8)
    plats = {n: Platform(MODEL, soc_profile=PROFILE, cim_n_cores=n) for n in cores}
    pre = {n: plats[n].compute_us(_mm("cim", 256, 2048, 2048)) for n in cores}
    for n in cores:
        assert pre[n] == pytest.approx(pre[4] * 4 / n)                       # prefill latency ∝ 1/n, not 1/n^2
    # decode: a 12M-element weight set fits SRAM at 8 cores (knee 8.16M x 2) but spills at 2 (x 0.5)
    K, N = 4096, 3072
    assert plats[2].cim.spills(K, N) and not plats[8].cim.spills(K, N)
    assert plats[8].compute_us(_mm("cim", 1, K, N)) < plats[2].compute_us(_mm("cim", 1, K, N))
    assert plats[8].cim.mt_knee == pytest.approx(plats[4].cim.mt_knee_kn * 2)
    # single-tile decode throughput rises with cores (s x G_eff(N/s, K)), sub-linearly (per-core fill)
    t = {n: plats[n].compute_us(_mm("cim", 1, 2048, 2048)) for n in cores}
    assert t[8] < t[4] < t[2] and t[8] > t[4] / 2
    # at the calibrated count physical == legacy numerics WITHIN the resident capacity; beyond it the
    # profile path re-programs the non-resident weights instead of the Card's spill floor (R3 2026-10-26,
    # tests/test_cim_residency.py)
    leg = Platform(MODEL)
    for node in (_mm("cim", 1, 2048, 2048), _mm("cim", 1, 2048, 3900), _mm("cim", 256, 2048, 2048)):
        assert plats[4].compute_us(node) == leg.compute_us(node)
    assert plats[4].compute_us(_mm("cim", 1, 8192, 2048)) < leg.compute_us(_mm("cim", 1, 8192, 2048))
    with pytest.raises(ValueError):
        CimTileModel(scale_mode="quadratic")


# ---- F13: per-unit weight bytes ------------------------------------------------------------------
def test_f13_per_unit_weight_bytes_applied_once_and_seen_by_solver():
    plat = Platform(MODEL, soc_profile=PROFILE)
    assert plat.weight_bytes_per_param == {"gpu": 0.5, "npu": 1.0, "cpu": 0.5, "cim": 1.0}
    base = build_token_dag(MODEL, "decode", 512)
    int8 = {n.id: n.bytes_streamed for n in base.nodes if n.category == "matmul"}
    dag = SCHEDULERS["gpu_only"].assign(base, Cfg(), platform=plat)
    for n in dag.nodes:
        if n.category == "matmul":
            assert n.wl.extra["bytes_int8"] == int8[n.id]
            assert n.bytes_streamed == int8[n.id] - n.wl.K * n.wl.N // 2        # weight half, activations unchanged
    snap = {n.id: n.bytes_streamed for n in dag.nodes}
    dag = SCHEDULERS["gpu_only"].assign(dag, Cfg(), platform=plat)              # idempotent
    assert {n.id: n.bytes_streamed for n in dag.nodes} == snap
    npu = SCHEDULERS["npu_only"].assign(build_token_dag(MODEL, "decode", 512), Cfg(), platform=plat)
    assert all(n.bytes_streamed == int8[n.id] for n in npu.nodes if n.category == "matmul")
    # a tiled DAG: sub-node bytes derive from the INT8 basis, then rescale per unit; the solver's view
    # of a candidate equals what the engine meters
    het = SCHEDULERS["heteroinfer"].assign(build_token_dag(MODEL, "decode", 512), Cfg(), platform=plat)
    from simulator.runtime.solver import _dram_bytes
    ctx = SolverContext(platform=plat, units=("gpu", "npu"), host="gpu", sync_us=10.0)
    parts = [n for n in het.nodes if n.category == "matmul" and n.tile]
    assert parts and all(_dram_bytes(ctx, n) == n.bytes_streamed for n in parts)
    for n in parts:
        if n.unit == "gpu":
            assert n.bytes_streamed < n.wl.extra["bytes_int8"]
    # legacy (no profile): untouched
    leg = SCHEDULERS["gpu_only"].assign(build_token_dag(MODEL, "decode", 512), Cfg(), platform=Platform(MODEL))
    assert all(n.bytes_streamed == int8[n.id] for n in leg.nodes if n.category == "matmul")
    # converts (ADR-0004) still exist at the GPU int8<->fp16 boundaries of a split
    assert any(n.category == "convert" for n in het.nodes)


# ---- F14: P2 GPU in the unit set runs ----------------------------------------------------------
def test_f14_rk3588_profile_runs_with_gpu_in_unit_set():
    from simulator.runtime.runner import run
    cfg = SimConfig.from_dict({"workload": {"model": MODEL, "context": 1024, "prefill_len": 128, "decode_len": 4},
                               "platform": {"soc_profile": "rk3588_cim",
                                            "units": {"cim": True, "gpu": True, "npu": True, "cpu": True}},
                               "scheduler": {"policy": "gpu_only"}})
    r = run(cfg)
    assert r["tok_s"] > 0 and r["prefill_tok_s"] > 0 and r["achieved_decode_bandwidth_GBs"] <= 29.97 + 0.3


# ---- F15: mem streams count against the consuming unit --------------------------------------------
def test_f15_single_unit_baselines_do_not_exceed_the_wall():
    plat = Platform(MODEL, soc_profile=PROFILE)
    for pol in ("gpu_only", "cim_only"):
        dag = SCHEDULERS[pol].assign(build_token_dag(MODEL, "decode", 512), Cfg(), platform=plat)
        st = {}
        run_dag(dag, plat, plat.bw, stats=st)
        assert st["achieved_GBs"] <= 43.0 * 1.01, (pol, st["achieved_GBs"])
    from simulator.runtime.events import _stream_owner
    emb = next(n for n in dag.nodes if n.category == "embedding")
    assert emb.unit == "mem" and _stream_owner(dag, emb) == dag[dag.successors(emb.id)[0]].unit != "mem"


# ---- CPU GEMV kernel time is distinct from the pool time (F4 second half) -----------------------
def test_cpu_gemv_kernel_time_distinct_from_pool():
    """Round 2: the per-op anchor 'GEMV = 1.33x its W4 pool time' is gone — the kernel time is
    floor + work/eff with (floor, eff) fitted END-TO-END (fit_cpu_gemv.py, tests/test_hetero_round2.py);
    a decode GEMV is now DEARER than its W4 pool time (the CPU is kernel-bound, as HeteroInfer Fig.12
    implies), and the compute term stays independent of the pool wall."""
    plat = Platform(MODEL, soc_profile=PROFILE)
    g, spec = plat.cpu_gemm, load_spec(f"soc_{PROFILE}")["units"]["cpu"]["gemm"]
    assert g.efficiency(1) == pytest.approx(spec["gemv_efficiency"]) and g.efficiency(10_000) == pytest.approx(0.5, abs=3e-3)
    assert g.kernel_floor_us == spec["kernel_floor_us"] == plat.cpu_kernel_floor_us and 10.0 <= g.kernel_floor_us <= 30.0
    t = plat.compute_us(_mm("cpu", 1, 4096, 4096))
    pool_w4 = 4096 * 4096 * 0.5 / (43.0 * 1e3)
    work = 2 * 4096 * 4096 / (g.peak_ops_s("int8") * g.efficiency(1)) * 1e6
    assert t == pytest.approx(g.kernel_floor_us + work) and 1.33 * pool_w4 < t < 4 * pool_w4
    assert SCHEDULERS["tiled_hetero"].allow_cpu_compute is False      # same unit set as gpu_only/npu_only


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
