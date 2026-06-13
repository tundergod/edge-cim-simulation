"""Phase 2.1 — M3 event loop (simulator/runtime/events.py).

Hand-computed toy DAGs: serial-chain finish time, cross-unit concurrency,
concurrency-off == serial sum, and node duration = max(compute, memory).

    .venv/bin/pytest tests/test_event_engine.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.models.engine import Workload  # noqa: E402
from simulator.runtime.dag import OpNode, Dag  # noqa: E402
from simulator.runtime.resources import SharedBandwidth  # noqa: E402
from simulator.runtime.events import run_dag  # noqa: E402


class _StubPlatform:
    """Prices each node's compute from a {node_id: us} map (no real unit models)."""
    def __init__(self, compute_map):
        self.compute_map = compute_map

    def compute_us(self, node):
        return self.compute_map.get(node.id, 0.0)


def _node(i, unit, deps=(), bytes_streamed=0):
    return OpNode(id=i, category="matmul", wl=Workload(op="matmul", M=1, K=64, N=64),
                  deps=list(deps), unit=unit, bytes_streamed=bytes_streamed)


def test_serial_chain_sums():
    dag = Dag([_node(0, "cim"), _node(1, "gpu", [0]), _node(2, "cpu", [1])])
    plat = _StubPlatform({0: 10.0, 1: 5.0, 2: 3.0})
    bw = SharedBandwidth(eff_BW_GBs=24.2)
    assert abs(run_dag(dag, plat, bw) - 18.0) < 1e-9     # 10 -> 5 -> 3 chained = 18


def test_concurrency_overlap_vs_off():
    # three independent compute nodes on three units
    dag = Dag([_node(0, "cim"), _node(1, "gpu"), _node(2, "cpu")])
    plat = _StubPlatform({0: 10.0, 1: 5.0, 2: 3.0})
    bw = SharedBandwidth(eff_BW_GBs=24.2)
    assert abs(run_dag(dag, plat, bw, concurrency=True) - 10.0) < 1e-9   # max, overlapped
    assert abs(run_dag(dag, plat, bw, concurrency=False) - 18.0) < 1e-9  # serial sum


def test_single_memory_stream():
    dag = Dag([_node(0, "cim", bytes_streamed=int(10e9))])    # 10 GB
    plat = _StubPlatform({0: 0.0})
    bw = SharedBandwidth(eff_BW_GBs=10.0)                     # 10 GB/s -> 1 s = 1e6 us
    assert abs(run_dag(dag, plat, bw) - 1e6) < 1e-3


def test_node_dur_is_max_compute_memory():
    # compute (2e6 us) dominates memory (1e6 us)
    dag = Dag([_node(0, "cim", bytes_streamed=int(10e9))])
    plat = _StubPlatform({0: 2e6})
    bw = SharedBandwidth(eff_BW_GBs=10.0)
    assert abs(run_dag(dag, plat, bw) - 2e6) < 1e-3


def test_fluid_fairshare_unequal_costart():
    # two co-started memory streams (5 GB + 10 GB) sharing a 10 GB/s channel.
    # fair-share: both at 5 GB/s; A (5 GB) finishes at 1s, then B re-shares to full
    # 10 GB/s for its last 5 GB -> +0.5s -> 1.5s. (The old dispatch-time-k model gave 2.0s.)
    dag = Dag([_node(0, "a", bytes_streamed=int(5e9)), _node(1, "b", bytes_streamed=int(10e9))])
    plat = _StubPlatform({0: 0.0, 1: 0.0})
    bw = SharedBandwidth(eff_BW_GBs=10.0, knee_GBs=10.0)
    assert abs(run_dag(dag, plat, bw) - 1.5e6) < 1e-3


def test_fluid_fairshare_equal_costart():
    # three equal 10 GB streams sharing a 15 GB/s knee -> 5 GB/s each -> all finish at 2s
    dag = Dag([_node(i, f"u{i}", bytes_streamed=int(10e9)) for i in range(3)])
    plat = _StubPlatform({0: 0.0, 1: 0.0, 2: 0.0})
    bw = SharedBandwidth(eff_BW_GBs=10.0, knee_GBs=15.0)
    assert abs(run_dag(dag, plat, bw) - 2.0e6) < 1e-3


def test_cyclic_dag_raises():
    # 0 <-> 1 (both ids exist so Dag builds, but it is cyclic) -> fail loud, no latency
    dag = Dag([_node(0, "cim", deps=[1]), _node(1, "gpu", deps=[0])])
    plat = _StubPlatform({0: 1.0, 1: 1.0})
    try:
        run_dag(dag, plat, SharedBandwidth(24.2))
    except ValueError:
        return
    raise AssertionError("cyclic DAG not rejected")


def test_zero_bandwidth_raises():
    # a memory-streaming op with zero bandwidth would stall forever -> fail loud
    dag = Dag([_node(0, "cim", bytes_streamed=int(1e9))])
    try:
        run_dag(dag, _StubPlatform({0: 0.0}), SharedBandwidth(eff_BW_GBs=0.0))
    except ValueError:
        return
    raise AssertionError("zero-bandwidth memory stream not rejected")


def test_unscheduled_node_raises():
    # unit=None (no scheduler run) must fail loud, not fall back to "cpu"
    dag = Dag([OpNode(id=0, category="matmul", wl=Workload(op="matmul"), deps=[], unit=None)])
    try:
        run_dag(dag, _StubPlatform({0: 1.0}), SharedBandwidth(24.2))
    except ValueError:
        return
    raise AssertionError("unscheduled node (unit=None) not rejected")


def test_empty_dag():
    dag = Dag([])
    assert run_dag(dag, _StubPlatform({}), SharedBandwidth(24.2)) == 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok {fn.__name__}")
    print(f"\n{len(fns)} event-engine tests passed.")
