"""ADR-0009 "Revisions 2026-10-26 (cooperation)" — two mechanisms so the GPU/NPU stop idling while the
CIM works. All SIMULATED (no silicon).

  C1  capacity-aware weight split (`capacity_split` candidate): the CIM takes the output columns that
      fit its residency budget (N_cim = min(N, floor(budget / K)) aligned down to the core width), the
      other units take the rest. Decode: the CIM part is RESIDENT (pinned across tokens: no per-token
      weight stream) against ONE global budget per DAG allocated by saving density (two passes);
      prefill: a per-op cap, weights written once per forward. Plans record `cim_resident_elems`,
      the DAG `cim_budget_used`.
  C2  prefill token-chunk pipelining (`scheduler.prefill_chunks` in {1, 2, 4, 8, "auto"}): the forward
      as C chunk sub-DAGs (block-causal attention, bytes conserved, GEMM FLOPs conserved exactly) the
      event engine overlaps; "auto" = the engine-ranked portfolio over {1, 2, 4, 8}.

    .venv/bin/pytest tests/test_cooperation.py      (from the repo root)
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools" / "trace_export"))
from simulator.models.engine import Workload  # noqa: E402
from simulator.runtime import tiling  # noqa: E402
from simulator.runtime.config import SimConfig  # noqa: E402
from simulator.runtime.dag import OpNode, Dag  # noqa: E402
from simulator.runtime.events import run_dag  # noqa: E402
from simulator.runtime.platform import Platform  # noqa: E402
from simulator.runtime.runner import run  # noqa: E402
from simulator.runtime.scheduler import SCHEDULERS, TiledHeteroScheduler, _allocate_cim_budget  # noqa: E402
from simulator.runtime.solver import SolverContext, candidates, cim_cols, solve  # noqa: E402
from simulator.runtime.workload import (build_token_dag, build_prefill_dag, chunk_rows,  # noqa: E402
                                        weight_order_violations)
from fake_platform import AnalyticFakePlatform  # noqa: E402

MODEL = "llama-3.2-1b"
PROFILE = "sd8gen3_cim"
MiB = 2 ** 20


class Cfg:
    model = MODEL
    units = {"cim": True, "gpu": True, "npu": True, "cpu": True}
    sync_mode = "fast"
    sync_us_override = None
    partition_grid = 8
    precision_boundary_placement = "consumer"
    host = None
    allow_cpu_compute = False

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _mm(nid, M, K, N, deps=(), unit=None):
    b = M * K + K * N + M * N
    return OpNode(id=nid, category="matmul", unit=unit, wl=Workload(op="matmul", M=M, K=K, N=N, dtype="int8", nbytes=b),
                  deps=list(deps), in_values=list(deps), out_value=nid, bytes_streamed=b, out_elems=M * N, precision="int8")


def _ew(nid, deps=(), unit=None, nbytes=1024, cat="norm"):
    return OpNode(id=nid, category=cat, unit=unit,
                  wl=Workload(op=cat, nbytes=nbytes, dtype="fp16", extra={"category": cat}),
                  deps=list(deps), in_values=list(deps), out_value=nid, bytes_streamed=nbytes, out_elems=nbytes // 2,
                  precision="fp16")


def _act_bytes(n):
    return n.wl.M * n.wl.K + n.wl.M * n.wl.N


# ---- C1: the capacity_split candidate ---------------------------------------------------------------
def test_c1_platform_budget_and_cim_cols():
    p = Platform(MODEL, soc_profile=PROFILE)
    assert p.cim_weight_budget_frac == 1.0 and p.cim_weight_budget_elems == pytest.approx(p.cim.mt_knee)
    assert p.cim_align_cols == p.cim.core_width == 512
    assert Platform(MODEL, soc_profile=PROFILE, cim_n_cores=8).cim_weight_budget_elems == pytest.approx(2 * p.cim.mt_knee)
    leg = Platform(MODEL)
    assert leg.cim_weight_budget_elems is None and leg.cim_weight_budget_frac is None   # legacy: no residency model
    half = Platform(MODEL, soc_profile=PROFILE, cim_weight_budget_frac=0.5)
    assert half.cim_weight_budget_frac == 0.5 and half.cim_weight_budget_elems == pytest.approx(0.5 * p.cim.mt_knee)
    assert Platform(MODEL, soc_profile=PROFILE, cim_weight_budget_frac=0).cim_weight_budget_elems == 0.0   # disables C1
    # the config knob: platform.cim_weight_budget_frac (needs a profile; flagged when it differs from the profile)
    d = {"workload": {"model": MODEL, "context": 1024}, "platform": {"soc_profile": PROFILE, "cim_weight_budget_frac": 0.0},
         "scheduler": {"policy": "tiled_hetero"}}
    c = SimConfig.from_dict(d)
    assert c.cim_weight_budget_frac == 0.0 and any("cim_weight_budget_frac override = 0.0" in x for x in c.provenance)
    d["platform"]["cim_weight_budget_frac"] = 1.0
    assert not any("cim_weight_budget_frac" in x for x in SimConfig.from_dict(d).provenance)
    for bad in (-0.1, 1.5, "0.5", True):
        d["platform"]["cim_weight_budget_frac"] = bad
        with pytest.raises(ValueError):
            SimConfig.from_dict(d)
    with pytest.raises(ValueError):
        SimConfig.from_dict({"workload": {"model": MODEL, "context": 1024}, "platform": {"cim_weight_budget_frac": 0.5}})
    ctx = SolverContext(platform=p, units=("gpu", "npu", "cim"), host="gpu", sync_us=10.0,
                        cim_cap_elems=8 * MiB, cim_align=512, cim_resident=True)
    assert cim_cols(ctx, 0, 4096, 14336) == 2048          # floor(8 Mi / 4096) = 2048, aligned
    assert cim_cols(ctx, 0, 4096, 4096) == 2048
    assert cim_cols(ctx, 0, 2048, 2048) == 2048           # fits entirely -> the whole N
    assert cim_cols(ctx, 0, 4096, 1500) == 1500
    ctx.cim_cap_elems = p.cim_weight_budget_elems          # the profile knee 8159232 -> 1991 -> 1536
    assert cim_cols(ctx, 0, 4096, 14336) == 1536
    ctx.cim_alloc = {7: 1024}                              # pass 2: exactly the allocation (absent -> 0)
    assert cim_cols(ctx, 7, 4096, 14336) == 1024 and cim_cols(ctx, 8, 4096, 14336) == 0
    assert cim_cols(SolverContext(platform=p, units=("cim",), host="cim", sync_us=0.0), 0, 4096, 4096) == 0


def test_c1_candidates_cim_takes_what_fits_rest_split_resident_in_decode_only():
    p = Platform(MODEL, soc_profile=PROFILE)
    dag = Dag([_ew(0, unit="gpu"), _mm(1, 1, 4096, 14336, deps=[0])])
    ctx = SolverContext(platform=p, units=("gpu", "npu", "cim"), host="gpu", sync_us=10.0,
                        align={u: (tiling.DEFAULT_ALIGN.get(u, (1, 1))[0], max(1, r)) for u, r in p.min_split_rows.items()},
                        cim_cap_elems=8 * MiB, cim_align=512, cim_resident=True)
    cands, units = candidates(ctx, dag, 1)
    caps = [c for c in cands if c.strategy == "capacity_split"]
    assert caps and all(c.dim == "N" and c.plan[0] == ("cim", 2048 / 14336) and c.resident and c.cim_elems == 4096 * 2048
                        for c in caps)
    rests = {tuple(sorted(u for u, _ in c.plan[1:])) for c in caps}
    assert rests == {("gpu",), ("npu",), ("gpu", "npu")}            # the rest on one unit or over the ratio grid
    assert all(abs(sum(f for _, f in c.plan) - 1.0) < 1e-9 for c in caps)
    tp = tiling.tile_parts(dag, 1, "N", caps[0].plan, align=ctx.align, resident=(0,))
    assert tp.sizes[0] == 2048 and sum(tp.sizes) == 14336 and tp.units[0] == "cim"
    cim_part, other = tp.parts[0][0], tp.parts[1][0]
    assert cim_part.wl.extra["cim_resident"] is True and "cim_resident" not in other.wl.extra
    assert tiling.unit_bytes(cim_part, 1.0) == _act_bytes(cim_part)          # pinned: no weight stream
    assert tiling.unit_bytes(other, 1.0) == tiling.base_bytes(other)
    # with the PROFILE budget (the 8 159 232-element knee) the CIM part is 1536 columns = within capacity:
    # priced by the resident law, no R3 program term
    ctx.cim_cap_elems = p.cim_weight_budget_elems
    real = [c for c in candidates(ctx, dag, 1)[0] if c.strategy == "capacity_split"]
    assert real and all(c.plan[0] == ("cim", 1536 / 14336) for c in real)
    rp = tiling.tile_parts(dag, 1, "N", real[0].plan, align=ctx.align, resident=(0,)).parts[0][0]
    assert rp.wl.N == 1536 and not p.cim.spills(4096, 1536) and p.cim.program_us(4096, 1536) == 0.0
    assert p.compute_us(rp) == pytest.approx(p.cim._decode_resident_lat_us(4096, 1536))
    ctx.cim_cap_elems = 8 * MiB
    # an op that fits entirely: decode -> a RESIDENT single placement besides the plain cim_only
    dag2 = Dag([_ew(0, unit="gpu"), _mm(1, 1, 2048, 2048, deps=[0])])
    cands2, _ = candidates(ctx, dag2, 1)
    res = [c for c in cands2 if c.strategy == "capacity_split"]
    assert len(res) == 1 and res[0].dim is None and res[0].plan == [("cim", 1.0)] and res[0].resident
    assert any(c.strategy == "cim_only" for c in cands2)
    # prefill (not resident): the split candidate is offered but never tagged; a fitting op adds nothing
    pre = SolverContext(platform=p, units=("gpu", "npu", "cim"), host="gpu", sync_us=30.0, align=ctx.align,
                        cim_cap_elems=8 * MiB, cim_align=512, cim_resident=False)
    dag3 = Dag([_ew(0, unit="gpu"), _mm(1, 256, 4096, 14336, deps=[0])])
    c3 = [c for c in candidates(pre, dag3, 1)[0] if c.strategy == "capacity_split"]
    assert c3 and not any(c.resident for c in c3) and all(c.plan[0] == ("cim", 2048 / 14336) for c in c3)
    assert not [c for c in candidates(pre, Dag([_ew(0, unit="gpu"), _mm(1, 256, 2048, 2048, deps=[0])]), 1)[0]
                if c.strategy == "capacity_split"]
    # no budget (legacy platform / fakes / cim not in the unit set): no capacity candidates
    none = SolverContext(platform=p, units=("gpu", "npu", "cim"), host="gpu", sync_us=10.0, align=ctx.align)
    assert not [c for c in candidates(none, dag, 1)[0] if c.strategy == "capacity_split"]
    best, n = solve(ctx, dag, 1)
    assert n > 0 and best.T_est < float("inf")


def test_c1_allocation_by_saving_density():
    req = {1: {"saving_us": 100.0, "K": 4096, "N_cim": 2048, "elems": 4096 * 2048},     # density 1.19e-5
           2: {"saving_us": 100.0, "K": 4096, "N_cim": 1024, "elems": 4096 * 1024},     # density 2.38e-5 (best)
           3: {"saving_us": -5.0, "K": 4096, "N_cim": 1024, "elems": 4096 * 1024},      # no saving: nothing
           4: {"saving_us": 1.0, "K": 2048, "N_cim": 2048, "elems": 2048 * 2048}}       # density 2.4e-7
    alloc = _allocate_cim_budget(req, 4096 * 2048 + 4096 * 512, 512)
    assert alloc == {2: 1024, 1: 512 * 3}        # op 2 whole, op 1 the aligned remainder (1536 cols), 3/4 nothing
    assert sum(req[n]["K"] * c for n, c in alloc.items()) <= 4096 * 2048 + 4096 * 512
    assert _allocate_cim_budget(req, 4096 * 1024 + 100, 512) == {2: 1024}
    assert _allocate_cim_budget(req, 0, 512) == {}


def test_c1_decode_global_budget_two_ops_competing_fake():
    """Two independent GEMVs on a fake where the GPU streams W4 (0.25 B/param) and the CIM pins: op A
    (4096 x 1024) saves more per pinned element than op B (4096 x 512) -> with a budget of exactly A's
    size A is pinned, B is not; with room for both, both are. Plans record the pinned elements, the
    DAG the budget used; the pinned node streams only its activation bytes."""
    plat = AnalyticFakePlatform(rates={"gpu": 1.0e6, "npu": 1.0e6, "cim": 8.0e6}, bw_GBs=24.2, knee_GBs=48.0)
    plat.weight_bytes_per_param = {"gpu": 0.25, "npu": 1.0, "cim": 1.0}
    plat.cim_align_cols = 512

    def dag():
        return Dag([_ew(0, cat="norm"), _mm(1, 1, 4096, 1024, deps=[0]), _mm(2, 1, 4096, 512, deps=[0]),
                    _ew(3, deps=[1, 2], cat="residual")])
    kw = dict(units=("gpu", "cim"), support_unit="gpu", matmul_dims=(), split_attention=False,
              split_support=False, host="gpu", refine=False)
    plat.cim_weight_budget_elems = 4096 * 1024
    out = TiledHeteroScheduler(**kw).assign(dag(), Cfg(sync_us_override=10.0), platform=plat)
    assert out.plans[1]["strategy"] == "capacity_split" and out.plans[1]["cim_resident_elems"] == 4096 * 1024
    assert out.plans[2]["cim_resident_elems"] == 0 and out.plans[2]["strategy"] != "capacity_split"
    assert out.cim_budget_used == 4096 * 1024 <= out.cim_budget_elems and out.cim_alloc == {1: 1024}
    assert out[1].unit == "cim" and out[1].wl.extra["cim_resident"] is True and out[1].bytes_streamed == _act_bytes(out[1])
    assert "cim_resident" not in out[2].wl.extra
    # the requests the allocation came from: A's saving density is the larger one
    # room for both -> both pinned, sum within the budget
    plat.cim_weight_budget_elems = 4096 * 1024 + 4096 * 512
    both = TiledHeteroScheduler(**kw).assign(dag(), Cfg(sync_us_override=10.0), platform=plat)
    assert both.plans[1]["cim_resident_elems"] == 4096 * 1024 and both.plans[2]["cim_resident_elems"] == 4096 * 512
    assert both.cim_budget_used == 4096 * 1536 <= both.cim_budget_elems
    # the engine runs the pinned plan faster than the unpinned one (no weight stream on the pinned op)
    plat.cim_weight_budget_elems = None
    base = TiledHeteroScheduler(**kw).assign(dag(), Cfg(sync_us_override=10.0), platform=plat)
    assert not hasattr(base, "cim_budget_used")
    assert run_dag(both, plat, plat.bw) < run_dag(base, plat, plat.bw)
    # idempotent re-assign keeps the plan and the budget record
    again = TiledHeteroScheduler(**kw).assign(both, Cfg(sync_us_override=10.0), platform=plat)
    assert again.cim_budget_used == both.cim_budget_used and [n.unit for n in again.nodes] == [n.unit for n in both.nodes]


def test_c1_real_platform_decode_dag_budget_respected_and_bytes_consistent():
    plat = Platform(MODEL, soc_profile=PROFILE)
    cap = plat.cim_weight_budget_elems
    out = TiledHeteroScheduler(refine=False).assign(build_token_dag(MODEL, "decode", 512), Cfg(host="gpu"), platform=plat)
    pinned = {k: p["cim_resident_elems"] for k, p in out.plans.items() if p.get("cim_resident_elems", 0) > 0}
    assert out.cim_budget_elems == cap and out.cim_budget_used == sum(pinned.values()) <= cap
    assert all(p["strategy"] == "capacity_split" for k, p in out.plans.items() if k in pinned)
    res_nodes = [n for n in out.nodes if n.wl.extra.get("cim_resident")]
    assert all(n.unit == "cim" and n.bytes_streamed == _act_bytes(n) for n in res_nodes)
    assert sum(n.wl.K * n.wl.N for n in res_nodes) == out.cim_budget_used
    assert all(not plat.cim.spills(n.wl.K, n.wl.N) for n in res_nodes)
    # the pass-1 allocation exists (the 1B layer-0 projections fit); the engine decides whether the pinned
    # plan is kept ("with C1 >= without C1" per host by construction): pinned iff it was faster
    t = out.cim_pin_engine_us
    assert t["pinned"] > 0 and t["unpinned"] > 0
    assert (out.cim_budget_used > 0) == (t["pinned"] < t["unpinned"] - 1e-9)
    assert bool(res_nodes) == (out.cim_budget_used > 0)
    st = {}
    assert run_dag(out, plat, plat.bw, stats=st) == pytest.approx(min(t.values()))
    assert all(0 <= u <= 1 + 1e-9 for u in st["unit_utilisation"].values())
    # the runner surfaces the decode budget
    r = run(SimConfig.from_dict({"workload": {"model": MODEL, "context": 1024, "prefill_len": 64, "decode_len": 4},
                                 "platform": {"soc_profile": PROFILE, "units": {"gpu": True, "npu": True, "cim": True, "cpu": True}},
                                 "scheduler": {"policy": "tiled_hetero", "host": "gpu"}}))
    assert r["cim_weight_budget_elems"] == cap and 0 <= r["cim_budget_used_decode"] <= cap
    off = run(SimConfig.from_dict({"workload": {"model": MODEL, "context": 1024, "prefill_len": 64, "decode_len": 4},
                                   "platform": {"soc_profile": PROFILE, "cim_weight_budget_frac": 0.0,
                                                "units": {"gpu": True, "npu": True, "cim": True, "cpu": True}},
                                   "scheduler": {"policy": "tiled_hetero", "host": "gpu"}}))
    assert off["cim_weight_budget_frac"] == 0.0 and off["cim_weight_budget_elems"] == 0.0
    assert off["cim_budget_used_decode"] is None and off["cim_resident_matmuls_decode"] == 0   # C1 disabled: no record
    assert (r["cim_resident_matmuls_decode"] >= 1) == (r["cim_budget_used_decode"] > 0)
    assert r["prefill_chunks"] == 1 and r["prefill_chunk_candidates_us"] is None


# ---- C2: prefill token-chunk pipelining -------------------------------------------------------------------
def _flops(dag, cat):
    return sum(tiling.flops(n) for n in dag.nodes if n.category == cat)


def test_c2_chunked_prefill_dag_shapes_and_conservation():
    assert chunk_rows(256, 4) == [64] * 4 and chunk_rows(250, 4) == [62, 62, 62, 64] and chunk_rows(3, 8) == [1, 1, 1]
    one = build_prefill_dag(MODEL, 256, 1)
    four = build_prefill_dag(MODEL, 256, 4)
    assert len(one.nodes) == len(build_token_dag(MODEL, "prefill", 256).nodes) and not hasattr(one, "prefill_chunks")
    assert four.prefill_chunks == 4 and four.chunk_rows == [64] * 4 and len(four.nodes) == 4 * len(one.nodes)
    assert four.is_acyclic()
    assert _flops(four, "matmul") == _flops(one, "matmul")                       # GEMM FLOPs conserved exactly
    reps = [n for n in four.nodes if tiling.is_attention_rep(n)]
    H, hd = reps[0].wl.heads, reps[0].wl.extra["hd"]
    n_blocks = len([n for n in one.nodes if tiling.is_attention_rep(n)])
    causal = n_blocks * sum(2 * 2 * H * 64 * hd * 64 * (c + 1) for c in range(4))   # QK^T + S.V per block
    assert _flops(four, "attention") == causal < _flops(one, "attention")         # block-causal total (analytic)
    for r in reps:
        c = r.wl.extra["chunk"]
        assert r.wl.M == 64 and r.wl.kv == 64 * (c + 1) and r.out_elems == H * 64 * r.wl.kv
        sv = next(n for n in four.nodes if n.pricing_group == r.id and n.id != r.id)
        assert sv.wl.kv == r.wl.kv and sv.out_elems == H * 64 * hd
        block = tiling._block(four, r.id)
        for cid in block[1:-1]:
            assert four[cid].out_elems == H * 64 * r.wl.kv                       # scale / mask / softmax widened
        ext = [four[d].wl.extra["chunk"] for d in r.deps]
        assert sorted(set(ext)) == list(range(c + 1))                              # K of every chunk <= c
        assert sorted({four[d].wl.extra["chunk"] for d in sv.deps}) == list(range(c + 1))   # V of every chunk <= c
    # bytes conserved: chunk 0 streams the weights, later chunks only their activation rows
    mm = lambda d: sum(n.bytes_streamed for n in d.nodes if n.category == "matmul")   # noqa: E731
    assert mm(four) == mm(one)
    for n in four.nodes:
        if n.category == "matmul" and n.wl.extra["chunk"] > 0:
            assert n.wl.extra["chunk_weights_shared"] and n.bytes_streamed == _act_bytes(n)
            assert n.wl.extra["bytes_int8"] == _act_bytes(n) + n.wl.K * n.wl.N
    assert sum(n.bytes_streamed for n in four.nodes if n.category != "matmul") == pytest.approx(
        sum(n.bytes_streamed for n in one.nodes if n.category != "matmul") * 0.0 + sum(
            n.bytes_streamed for n in four.nodes if n.category != "matmul"))
    # remainder absorbed by the last chunk; chunk ids chunk-major
    rem = build_prefill_dag(MODEL, 250, 4)
    assert rem.chunk_rows == [62, 62, 62, 64] and _flops(rem, "matmul") == _flops(build_prefill_dag(MODEL, 250, 1), "matmul")
    assert [n.wl.extra["chunk"] for n in four.nodes] == sorted(n.wl.extra["chunk"] for n in four.nodes)


class _OverlapFake(AnalyticFakePlatform):
    """CIM GEMMs slow, GPU attention slow and priced PER QUERY ROW (kv-independent, so C = 1 and C = 4 do
    identical attention work and only the overlap differs); bytes irrelevant."""

    def __init__(self):
        super().__init__(rates={"gpu": 1.0e5, "cim": 2.0e6}, byte_rates={"gpu": 1e9}, bw_GBs=1e6, knee_GBs=1e6)

    def price(self, node):
        r = super().price(node)
        if node.category == "attention" and "hd" in node.wl.extra and node.unit == "gpu":
            r["latency_us"] = 0.0 if node.pricing_group != node.id else 40.0 * tiling._attn_M(node)   # 40 us per row
            # (GPU attention work ~= the CIM GEMM work, so a pipeline can hide most of it)
        return r


def test_c2_engine_overlaps_chunks_on_a_fake():
    plat = _OverlapFake()
    kw = dict(units=("cim", "gpu"), support_unit="gpu", matmul_dims=(), split_attention=False, split_support=False,
              host="gpu", refine=False)
    lat, busy = {}, {}
    for C in (1, 4):
        out = TiledHeteroScheduler(**kw).assign(build_prefill_dag(MODEL, 256, C), Cfg(sync_us_override=0.0), platform=plat)
        assert all(n.unit == "cim" for n in out.nodes if n.category == "matmul")
        st = {}
        lat[C] = run_dag(out, plat, plat.bw, stats=st)
        busy[C] = st["unit_busy_us"]
    # same work (GEMM FLOPs conserved, attention per row, no sync), C = 4 overlaps it across the two units
    # (rel 1e-3: the fixture's per-row byte templates carry small additive constants, so a 64-row chunk's
    # support-op bytes are not exactly a quarter of the 256-row ones)
    assert busy[4]["cim"] == pytest.approx(busy[1]["cim"]) and busy[4]["gpu"] == pytest.approx(busy[1]["gpu"], rel=1e-3)
    serial = busy[1]["cim"] + busy[1]["gpu"]
    assert lat[1] > 0.95 * serial                                           # C = 1: (almost) a serial chain
    # overlap shows (10 % here: the engine dispatches lowest-id-first with no cross-chunk priority, so the
    # host GPU runs chunk c's long attention before chunk c+1's short norm that would unblock the CIM)
    assert lat[4] < 0.95 * lat[1] and lat[4] < 0.95 * (busy[4]["cim"] + busy[4]["gpu"])
    assert lat[4] >= max(busy[4].values())


def test_c2_config_knob_and_provenance():
    base = {"workload": {"model": MODEL, "context": 1024, "prefill_len": 64, "decode_len": 4},
            "platform": {"soc_profile": PROFILE, "units": {"gpu": True, "npu": True, "cim": True, "cpu": True}},
            "scheduler": {"policy": "tiled_hetero", "host": "gpu"}}
    cfg = SimConfig.from_dict(base)
    assert cfg.prefill_chunks == 1 and not any("prefill_chunks" in p for p in cfg.provenance)
    for ok in (2, 4, 8, "auto"):
        base["scheduler"]["prefill_chunks"] = ok
        c = SimConfig.from_dict(base)
        assert c.prefill_chunks == ok and any("prefill_chunks" in p for p in c.provenance)
    for bad in (3, 0, "x", True, 16):
        base["scheduler"]["prefill_chunks"] = bad
        with pytest.raises(ValueError):
            SimConfig.from_dict(base)
    l4 = SimConfig.from_dict({"workload": {"model": MODEL, "context": 1024}})
    assert l4.prefill_chunks == 1 and l4.is_calibrated_anchor()


def test_c2_runner_auto_picks_the_engine_best_chunk_count():
    def cfg(pc):
        return SimConfig.from_dict({"workload": {"model": MODEL, "context": 1024, "prefill_len": 128, "decode_len": 4},
                                    "platform": {"soc_profile": PROFILE,
                                                 "units": {"gpu": True, "npu": True, "cim": True, "cpu": True}},
                                    "scheduler": {"policy": "tiled_hetero", "host": "gpu", "prefill_chunks": pc}})
    auto = run(cfg("auto"))
    cand = auto["prefill_chunk_candidates_us"]
    assert set(cand) == {1, 2, 4, 8} and auto["prefill_chunks_config"] == "auto"
    best = min(cand.values())
    assert auto["prefill_chunks"] == min(C for C, t in cand.items() if abs(t - best) <= 1e-3)
    assert auto["ttft_s_reported_not_gated"] == pytest.approx(best / 1e6 + auto["prefill_critical_path_us"] * 0, abs=2e-3)
    assert auto["prefill_weight_order_checked"] >= 0 and auto["prefill_weight_order_violations"] <= auto["prefill_weight_order_checked"]
    one = run(cfg(1))
    assert one["prefill_chunks"] == 1 and one["prefill_chunk_candidates_us"] is None
    assert one["ttft_s_reported_not_gated"] == pytest.approx(cand[1] / 1e6, abs=2e-3)
    assert one["prefill_weight_order_checked"] == 0 == one["prefill_weight_order_violations"]
    four = run(cfg(4))
    assert four["prefill_chunks"] == 4 and four["ttft_s_reported_not_gated"] == pytest.approx(cand[4] / 1e6, abs=2e-3)
    assert four["prefill_weight_order_checked"] > 0
    assert one["tok_s"] == pytest.approx(auto["tok_s"]) == pytest.approx(four["tok_s"])    # decode untouched


def test_c2_weight_order_check():
    four = build_prefill_dag(MODEL, 64, 2)
    for n in four.nodes:
        n.unit = "cim" if n.category == "matmul" else "gpu"
    mats = [n.id for n in four.nodes if n.category == "matmul"]
    half = len(mats) // 2
    stationary = {"cim": [i for pair in zip(mats[:half], mats[half:]) for i in pair]}   # G_i(0), G_i(1), ...
    assert weight_order_violations(four, stationary) == (0, half)
    assert weight_order_violations(four, {"cim": mats}) == (half, half)                 # chunk 0 fully, then chunk 1
    assert weight_order_violations(four, {"cim": []}) == (0, 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
