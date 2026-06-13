"""Phase 2.1 — tests for the op-DAG types (simulator/runtime/dag.py).

Covers: acyclicity detection, successor/roots index, fail-loud on duplicate
ids / dangling deps, and Workload shape sanity. Plain asserts + __main__ runner
(matches tests/test_engine_iface.py; also pytest-collectable).

    .venv/bin/pytest tests/test_dag.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.models.engine import Workload  # noqa: E402
from simulator.runtime.dag import OpNode, Dag, wl_is_sane  # noqa: E402


def _wl(op="matmul", **kw):
    return Workload(op=op, **kw)


def _node(i, deps=(), cat="matmul"):
    return OpNode(id=i, category=cat, wl=_wl(M=1, K=64, N=64), deps=list(deps))


def test_linear_chain_acyclic():
    dag = Dag([_node(0), _node(1, [0]), _node(2, [1])])
    assert dag.is_acyclic()
    assert dag.roots() == [0]
    assert dag.successors(0) == [1]
    assert dag.successors(2) == []
    assert len(dag) == 3


def test_fanout_join_acyclic():
    # 0 -> {1,2,3} (QKV) -> 4 (attention join)
    nodes = [_node(0), _node(1, [0]), _node(2, [0]), _node(3, [0]),
             _node(4, [1, 2, 3], cat="attention")]
    dag = Dag(nodes)
    assert dag.is_acyclic()
    assert sorted(dag.successors(0)) == [1, 2, 3]
    assert dag.roots() == [0]


def test_cycle_detected():
    dag = Dag([_node(0, [1]), _node(1, [0])])   # both ids exist -> builds, but cyclic
    assert not dag.is_acyclic()


def test_duplicate_id_rejected():
    try:
        Dag([_node(0), _node(0)])
    except ValueError:
        return
    raise AssertionError("duplicate id not rejected")


def test_dangling_dep_rejected():
    try:
        Dag([_node(0, [99])])
    except ValueError:
        return
    raise AssertionError("dangling dep not rejected")


def test_wl_sanity():
    assert wl_is_sane(_wl(M=1, K=4096, N=4096, dtype="int8"))
    assert wl_is_sane(_wl(op="attention", kv=512, heads=32, dtype="fp16"))
    assert not wl_is_sane(_wl(M=1, K=-1, N=64))          # negative dim
    assert not wl_is_sane(_wl(K=64, N=64, dtype="bf16"))  # unknown dtype
    assert not wl_is_sane(_wl(op="", K=64))               # empty op


def test_every_node_wl_sane():
    dag = Dag([_node(0), _node(1, [0], cat="attention")])
    assert all(wl_is_sane(n.wl) for n in dag.nodes)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"\n{len(fns)} dag tests passed.")
