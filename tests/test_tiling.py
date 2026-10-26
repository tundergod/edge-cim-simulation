"""ADR-0009 — op tiling (simulator/runtime/tiling.py).

Invariants per split dim: FLOPs conserved exactly, bytes = original + replicated operand +
explicit merge bytes, acyclic, successors rewired to the merge node, deterministic ids,
alignment/minimum handling, single-part no-op, attention block split with fresh pricing
groups, per-boundary sync charging (idempotent), and the engine runs the tiled DAG.

    .venv/bin/pytest tests/test_tiling.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from simulator.models.engine import Workload  # noqa: E402
from simulator.runtime.dag import OpNode, Dag  # noqa: E402
from simulator.runtime.workload import build_token_dag  # noqa: E402
from simulator.runtime import tiling  # noqa: E402
from simulator.runtime.tiling import split_node, split_dag, partition, apply_sync, resolve_sync_us  # noqa: E402
from simulator.runtime.events import run_dag  # noqa: E402
from fake_platform import AnalyticFakePlatform  # noqa: E402

MODEL = "llama-3.2-1b"


def _dag(phase="decode", L=512):
    return build_token_dag(MODEL, phase, L)


def _first(dag, pred):
    return next(n for n in dag.nodes if pred(n))


def _check_common(orig, new, nid, k):
    """Structural invariants shared by every dim."""
    assert new.is_acyclic()
    assert nid not in new._by_id                                   # original removed
    plan = new.plans[nid]
    subs = [new[i] for i in plan["sub_ids"]]
    merge = new[plan["merge_id"]]
    assert merge.category == "merge" and merge.tile["role"] == "merge"
    assert len({s.unit for s in subs}) == k and all(s.tile["of"] == nid for s in subs)
    last = plan["block"][-1]
    for s in orig.successors(last):                                # successors now depend on the merge
        assert plan["merge_id"] in new[s].deps and last not in new[s].deps
    assert set(merge.deps) == {s.id for s in subs if s.tile["src"] == last}
    assert plan["sub_ids"] + [plan["merge_id"]] == list(range(max(n.id for n in orig.nodes) + 1,
                                                              max(n.id for n in orig.nodes) + 1 + len(subs) + 1))
    return subs, merge


def _bytes_identity(orig, new, nid, replicated_expected):
    plan = new.plans[nid]
    orig_bytes = sum(orig[c].bytes_streamed for c in plan["block"])
    sub_bytes = sum(new[i].bytes_streamed for i in plan["sub_ids"])
    merge = new[plan["merge_id"]]
    assert sub_bytes + merge.bytes_streamed == orig_bytes + plan["replicated_bytes"] + plan["merge_bytes"]
    assert plan["replicated_bytes"] == replicated_expected


def test_partition_rounds_aligns_and_fails_loud():
    assert partition(100, [0.5, 0.5]) == [50, 50]
    assert partition(2048, [0.3, 0.7], quanta=[32, 1]) == [608, 1440]        # 614.4 -> 608 (x32), remainder last
    assert partition(7, [1.0]) == [7]
    for bad in ([0.5, 0.6], [0.0, 1.0], []):
        try:
            partition(10, bad)
        except ValueError:
            continue
        raise AssertionError(f"bad fractions {bad} accepted")
    try:
        partition(64, [0.1, 0.9], quanta=[32, 1], minima=[32, 1])           # 6.4 -> 0 rows on part 0
    except ValueError:
        pass
    else:
        raise AssertionError("infeasible aligned plan not rejected")
    try:
        partition(100, [0.5, 0.5], minima=[1, 60])
    except ValueError:
        return
    raise AssertionError("minimum-size violation not rejected")


def test_split_matmul_N_M_K_invariants():
    for dim, phase, L in (("N", "decode", 512), ("M", "prefill", 256), ("K", "decode", 512), ("N", "prefill", 256)):
        orig = _dag(phase, L)
        mm = _first(orig, lambda n: n.category == "matmul")
        M, K, N = mm.wl.M, mm.wl.K, mm.wl.N
        plan = [("npu", 0.375), ("cim", 0.125), ("gpu", 0.5)]
        new = split_node(_dag(phase, L), mm.id, dim, plan, merge_unit="gpu", sync_us=10.0)
        subs, merge = _check_common(orig, new, mm.id, 3)
        assert sum(tiling.flops(s) for s in subs) == tiling.flops(mm)        # FLOPs conserved exactly
        axis = {"N": "N", "M": "M", "K": "K"}[dim]
        assert sum(getattr(s.wl, axis) for s in subs) == getattr(mm.wl, axis)
        npu = next(s for s in subs if s.unit == "npu")
        assert getattr(npu.wl, axis) % 32 == 0                               # NPU quantum (default align)
        for s in subs:                                                      # the other two dims untouched
            for other in "MKN".replace(axis, ""):
                assert getattr(s.wl, other) == getattr(mm.wl, other)
        k = 3
        rep = {"N": (k - 1) * M * K, "M": (k - 1) * K * N, "K": (k - 1) * M * N}[dim]   # op_profile: 1 B/elem
        _bytes_identity(orig, new, mm.id, rep)
        if dim == "K":
            assert merge.bytes_streamed == k * M * N * tiling.ACC_BYTES + M * N and merge.mem_domain == "dram"
        else:
            assert merge.bytes_streamed == 0 and merge.mem_domain == "none"
        assert merge.wl.extra["fixed_us"] == 10.0 and merge.unit == "gpu"
        assert all(s.category == "matmul" and s.precision == mm.precision for s in subs)


def test_split_attention_block_heads_and_kv():
    for dim in ("heads", "kv"):
        orig = _dag("decode", 512)
        qk = _first(orig, tiling.is_attention_rep)
        sv = _first(orig, lambda n: n.pricing_group == qk.id and n.id != qk.id)
        align = {"heads": (4, 4)}                                           # llama-3.2-1b GQA group = 32/8
        new = split_node(_dag("decode", 512), qk.id, dim, [("gpu", 0.5), ("cim", 0.5)],
                         merge_unit="gpu", sync_us=10.0, align=align)
        subs, merge = _check_common(orig, new, qk.id, 2)
        plan = new.plans[qk.id]
        assert plan["block"][0] == qk.id and plan["block"][-1] == sv.id and len(plan["block"]) >= 3
        # every part copies the whole chain (QK^T, scale/mask/softmax, S.V) on ONE unit
        for part in (0, 1):
            members = [s for s in subs if s.tile["part"] == part]
            assert [m.tile["src"] for m in members] == plan["block"]
            assert len({m.unit for m in members}) == 1
            q = members[0]; v = members[-1]
            assert q.pricing_group == q.id and v.pricing_group == q.id          # FRESH group per sub-pair
            assert v.deps.count(q.id) == 0 and members[1].deps == [q.id]         # chain rewired inside the part
        qks = [s for s in subs if s.tile["src"] == qk.id]
        svs = [s for s in subs if s.tile["src"] == sv.id]
        assert sum(tiling.flops(s) for s in qks) == tiling.flops(qk)
        assert sum(tiling.flops(s) for s in svs) == tiling.flops(sv)
        H, hd, kv = qk.wl.heads, qk.wl.extra["hd"], qk.wl.kv
        if dim == "heads":
            assert [s.wl.heads for s in qks] == [16, 16] and all(s.wl.heads % 4 == 0 for s in qks)
            assert all(s.wl.kv == kv for s in qks)
            _bytes_identity(orig, new, qk.id, 0)
            assert merge.bytes_streamed == 0
        else:
            assert sum(s.wl.kv for s in qks) == kv and all(s.wl.heads == H for s in qks)
            sm = [s for s in subs if s.category == "softmax"]
            assert sum(s.wl.kv for s in sm) == kv                             # softmax rows follow the chunk
            # Q replicated per chunk for QK^T; a full partial output per chunk for S.V (M = 1)
            _bytes_identity(orig, new, qk.id, 2 * H * hd)
            assert merge.bytes_streamed == 2 * H * (hd + 2) * tiling.ACC_BYTES + H * hd
        assert merge.mem_domain == ("dram" if merge.bytes_streamed else "none")


def test_split_rows_support_op_prefill_only():
    orig = _dag("prefill", 256)
    norm = _first(orig, lambda n: n.category == "norm" and n.bytes_streamed > 1000)
    new = split_node(_dag("prefill", 256), norm.id, "rows", [("gpu", 0.25), ("cpu", 0.75)], rows=256)
    subs, merge = _check_common(orig, new, norm.id, 2)
    assert [s.bytes_streamed for s in subs] == [norm.bytes_streamed // 4, norm.bytes_streamed - norm.bytes_streamed // 4]
    assert sum(s.out_elems for s in subs) == norm.out_elems and merge.bytes_streamed == 0
    _bytes_identity(orig, new, norm.id, 0)
    assert new.plans[norm.id]["sizes"] == [64, 192]
    try:                                                                   # decode: 1 row -> unsplittable
        split_node(_dag("decode", 512), norm.id, "rows", [("gpu", 0.5), ("cpu", 0.5)], rows=1)
    except ValueError:
        return
    raise AssertionError("rows split with rows=1 not rejected")


def test_single_part_plan_is_noop():
    dag = _dag("decode", 512)
    mm = _first(dag, lambda n: n.category == "matmul")
    n0 = len(dag.nodes)
    out = split_node(dag, mm.id, "N", [("npu", 1.0), ("gpu", 0.0)])
    assert out is dag and len(dag.nodes) == n0 and dag[mm.id].unit == "npu"
    assert not any(n.category == "merge" for n in dag.nodes)


def test_deterministic_ids_and_split_dag():
    a = split_dag(_dag("decode", 512), {13: ("N", [("npu", 0.5), ("gpu", 0.5)]),
                                        33: ("K", [("cim", 0.5), ("gpu", 0.5)])}, merge_unit="gpu")
    b = split_dag(_dag("decode", 512), {13: ("N", [("npu", 0.5), ("gpu", 0.5)]),
                                        33: ("K", [("cim", 0.5), ("gpu", 0.5)])}, merge_unit="gpu")
    assert [(n.id, n.category, n.unit, n.deps) for n in a.nodes] == [(n.id, n.category, n.unit, n.deps) for n in b.nodes]
    assert a.plans[13]["merge_id"] < a.plans[33]["sub_ids"][0]           # ids allocated in nid order
    assert a.is_acyclic()


def test_illegal_splits_fail_loud():
    dag = _dag("decode", 512)
    mm = _first(dag, lambda n: n.category == "matmul")
    sv = _first(dag, lambda n: n.category == "attention" and "hd" in n.wl.extra and n.pricing_group != n.id)
    for nid, dim in ((mm.id, "heads"), (mm.id, "rows"), (sv.id, "heads"), (0, "N")):
        try:
            split_node(_dag("decode", 512), nid, dim, [("gpu", 0.5), ("npu", 0.5)])
        except ValueError:
            continue
        raise AssertionError(f"split of node {nid} on {dim!r} not rejected")


def _toy(units):
    nodes = [OpNode(id=0, category="embedding", wl=Workload(op="stream", nbytes=8), unit="mem", bytes_streamed=8),
             OpNode(id=1, category="norm", wl=Workload(op="norm"), deps=[0], unit=units[0]),
             OpNode(id=2, category="matmul", wl=Workload(op="matmul", M=1, K=64, N=64), deps=[1], unit=units[1]),
             OpNode(id=3, category="residual", wl=Workload(op="residual"), deps=[1, 2], unit=units[2])]
    for n in nodes:
        n.mem_domain = "dram" if n.bytes_streamed else "none"
    return Dag(nodes)


def test_apply_sync_charges_each_cross_unit_consumer_once_and_is_idempotent():
    dag = _toy(("gpu", "npu", "gpu"))
    assert apply_sync(dag, 10.0) == 2                     # npu<-gpu, gpu<-{gpu,npu}; mem->gpu is free
    assert dag[1].wl.extra.get("fixed_us") is None
    assert dag[2].wl.extra["fixed_us"] == 10.0 and dag[3].wl.extra["fixed_us"] == 10.0   # once per consumer
    assert apply_sync(dag, 10.0) == 2 and dag[3].wl.extra["fixed_us"] == 10.0           # idempotent
    assert apply_sync(dag, 0.0) == 0 and "fixed_us" not in dag[3].wl.extra               # cleared
    assert apply_sync(_toy(("gpu", "gpu", "gpu")), 400.0) == 0                           # no boundary


def test_merge_node_carries_its_own_sync():
    dag = split_node(_dag("decode", 512), 13, "N", [("npu", 0.5), ("gpu", 0.5)], merge_unit="gpu", sync_us=10.0)
    for n in dag.nodes:
        if n.unit is None:
            n.unit = "gpu"
    assert apply_sync(dag, 10.0) >= 1
    merge = dag[dag.plans[13]["merge_id"]]
    assert merge.wl.extra["fixed_us"] == 10.0 and merge.wl.extra["sync_us"] == 10.0


def test_resolve_sync_us_table_and_override():
    assert resolve_sync_us("fast", "decode") == 10.0 and resolve_sync_us("fast", "prefill") == 30.0
    assert resolve_sync_us("clfinish", "decode") == 400.0 and resolve_sync_us("none", "prefill") == 0.0
    assert resolve_sync_us("clfinish", "decode", override=5.0) == 5.0
    for bad in (("bogus", "decode", None), ("fast", "bogus", None), ("fast", "decode", -1.0)):
        try:
            resolve_sync_us(*bad[:2], override=bad[2])
        except ValueError:
            continue
        raise AssertionError(f"{bad} accepted")


def test_engine_runs_tiled_dag_and_reports_stats():
    plat = AnalyticFakePlatform()
    dag = split_dag(_dag("decode", 512), {13: ("N", [("npu", 0.5), ("gpu", 0.5)]),
                                          33: ("K", [("cim", 0.5), ("gpu", 0.5)])}, merge_unit="gpu", sync_us=10.0)
    from simulator.runtime.scheduler import _residency
    for n in dag.nodes:
        if n.unit is None:
            n.unit = {"matmul": "gpu", "attention": "gpu", "kv_cache": "mem", "embedding": "mem"}.get(n.category, "gpu")
        if n.category != "merge":
            n.mem_domain = _residency(n)
    apply_sync(dag, 10.0)
    stats = {}
    t = run_dag(dag, plat, plat.bw, stats=stats)
    assert t > 0 and stats["latency_us"] == t
    assert stats["n_merge"] == 2 and stats["n_sync"] >= 2 and stats["sync_us"] >= 20.0
    assert set(stats["unit_busy_us"]) >= {"gpu", "npu", "cim"}
    assert 0 < stats["achieved_GBs"] <= plat.bw.knee + 1e-9
    assert 0 < stats["critical_path_us"] <= t + 1e-6
    assert stats["dram_bytes"] == sum(n.bytes_streamed for n in dag.nodes if n.mem_domain == "dram")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok {fn.__name__}")
    print(f"\n{len(fns)} tiling tests passed.")
