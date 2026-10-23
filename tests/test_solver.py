"""ADR-0009 — per-op partition solver (simulator/runtime/solver.py).

Table 3 reproduction (HeteroInfer, 8 Gen 3 [LIT]): with the GPU/NPU anchor latencies scaled
linearly in N for sub-shapes, a 1/8 ratio grid, the fast-sync costs and a minimum
sub-tensor size of 1024 output rows per unit, the solver reproduces the paper's decisions
(ratio tolerance: one grid step). HONESTY: the minimum sub-tensor size is HeteroInfer §4.4
profiler constraint (2) ("NPU stage-performance minimum sub-tensor sizes"), whose value the
paper does not state; 1024 is a test choice. Under PURE linear scaling with no minimum, the
[4096,14336] M=1 case prefers a 7:1 weight-centric split (saves ~130 us of 1467) and the
[4096,4096] M=128 case is within 2 us of a 1:7 split — i.e. two of the paper's "no partition"
rows are NOT reproduced by linear scaling alone (see ADR-0009 limitations).

    .venv/bin/pytest tests/test_solver.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from simulator.models.engine import Workload  # noqa: E402
from simulator.runtime.dag import OpNode, Dag  # noqa: E402
from simulator.runtime.solver import SolverContext, solve, candidates  # noqa: E402
from simulator.runtime.scheduler import HeteroInferScheduler  # noqa: E402
from fake_platform import Table3FakePlatform, AnalyticFakePlatform  # noqa: E402

MIN_ROWS = 1024
ALIGN = {"npu": (32, MIN_ROWS), "gpu": (1, MIN_ROWS)}


def _dag(M, K, N):
    """producer (norm on the GPU = HeteroInfer's host graph) -> matmul [M,K]x[K,N]."""
    return Dag([OpNode(id=0, category="norm", wl=Workload(op="norm", nbytes=0), unit="gpu", mem_domain="none",
                       precision="fp16"),
                OpNode(id=1, category="matmul", wl=Workload(op="matmul", M=M, K=K, N=N), deps=[0],
                       bytes_streamed=M * K + K * N + M * N, precision="int8", out_elems=M * N)])


def _ctx(plat, M):
    return SolverContext(platform=plat, units=("gpu", "npu"), host="gpu",
                         sync_us=(30.0 if M > 1 else 10.0), grid=8, align=ALIGN, rows=M,
                         dims={"matmul": ("N",)})


def _gpu_frac(c):
    return sum(f for u, f in c.plan if u == "gpu")


TABLE3_CASES = [
    # (M, K, N, n_full, expected strategy, expected GPU fraction)       Table 3 row
    (1, 4096, 4096, 4096, "weight_centric", 0.5),                         # [4096,4096] x [4096,1]   1:1
    (1, 4096, 28672, 28672, "weight_centric", 0.75),                      # [28672,4096] x [4096,1]  3:1
    (1, 14336, 4096, 4096, "gpu_only", 1.0),                              # [4096,14336] x [14336,1] GPU-only
    (128, 4096, 4096, 4096, "npu_only", 0.0),                             # [4096,4096] x [4096,128] NPU-only
    (256, 14336, 4096, 4096, "weight_centric", 0.4),                      # [4096,14336] x [14336,256] 2:3
]


def test_table3_decisions_reproduced():
    for M, K, N, n_full, strategy, gfrac in TABLE3_CASES:
        plat = Table3FakePlatform(n_full=n_full)
        best, n = solve(_ctx(plat, M), _dag(M, K, N), 1)
        assert n > 1
        assert best.strategy == strategy, (M, K, N, best)
        assert abs(_gpu_frac(best) - gfrac) <= 1 / 8 + 1e-9, (M, K, N, best)


def test_table3_through_the_heteroinfer_scheduler():
    # the same decisions through HeteroInferScheduler.assign (its solver context), plans recorded
    sched = HeteroInferScheduler(align=ALIGN)
    class Cfg:
        model = None; units = {"gpu": True, "npu": True, "cim": False, "cpu": True}
        sync_mode = "fast"; sync_us_override = None; partition_grid = 8; precision_boundary_placement = "consumer"
    for M, K, N, n_full, strategy, gfrac in TABLE3_CASES:
        dag = sched.assign(_dag(M, K, N), Cfg(), platform=Table3FakePlatform(n_full=n_full))
        p = dag.plans[1]
        assert p["strategy"] == strategy and abs(sum(f for u, f in p["parts"] if u == "gpu") - gfrac) <= 1 / 8 + 1e-9
        if p["dim"] is not None:                                     # a split: sub-nodes + merge on the GPU host
            assert dag[p["merge_id"]].unit == "gpu" and dag[p["merge_id"]].wl.extra["fixed_us"] == (30.0 if M > 1 else 10.0)
            assert dag.is_acyclic()


def test_linear_scaling_alone_does_not_give_gpu_only():
    # the honesty check behind the docstring: without the minimum-size constraint the solver
    # splits the [4096,14336] M=1 matmul (a documented deviation, not a tolerance issue).
    ctx = _ctx(Table3FakePlatform(), 1)
    ctx.align = {"npu": (32, 32)}
    best, _ = solve(ctx, _dag(1, 14336, 4096), 1)
    assert best.strategy == "weight_centric" and _gpu_frac(best) == 0.875


def test_activation_centric_candidates_and_static_npu():
    # M=300 is not a standard length: HeteroInfer chunks 256 onto the NPU (pre-built graph)
    # and runs the 44-token remainder on the GPU. With a static-graph NPU (non-standard M
    # unpriceable), the hybrid/N-split candidates at M=300 drop out and the chunked plan wins.
    plat = AnalyticFakePlatform(rates={"gpu": 1.0e6, "npu": 8.0e6}, static_npu=True)
    ctx = SolverContext(platform=plat, units=("gpu", "npu"), host="gpu", sync_us=30.0, grid=8,
                        align={"npu": (32, 32)}, rows=300, dims={"matmul": ("N",)})
    cands, _ = candidates(ctx, _dag(300, 2048, 2048), 1)
    ac = [c for c in cands if c.strategy == "activation_centric"]
    assert [tuple(c.plan) for c in ac] == [(("npu", 128 / 300), ("npu", 128 / 300), ("gpu", 44 / 300)),
                                           (("npu", 256 / 300), ("gpu", 44 / 300))]
    assert any(c.strategy == "hybrid" for c in cands)                 # N-split at a non-standard M is labelled hybrid
    best, _ = solve(ctx, _dag(300, 2048, 2048), 1)
    assert best.strategy == "activation_centric" and best.plan[0] == ("npu", 256 / 300)


def test_solver_estimate_includes_sync_and_hand_back():
    # gpu-only pays no sync (host); npu-only pays sync in + hand-back; the estimate is the
    # engine's per-boundary rule, so a tiny NPU advantage (~80 us at M=32) loses to 2 x 100 us.
    plat = AnalyticFakePlatform(rates={"gpu": 1.0e6, "npu": 1.01e6}, bw_GBs=1e6, knee_GBs=1e6)
    ctx = SolverContext(platform=plat, units=("gpu", "npu"), host="gpu", sync_us=100.0, grid=8,
                        align={"npu": (32, 32)}, rows=32, dims={"matmul": ()})
    best, n = solve(ctx, _dag(32, 512, 512), 1)
    assert n == 2 and best.strategy == "gpu_only"
    ctx.sync_us = 0.0
    best, _ = solve(ctx, _dag(32, 512, 512), 1)
    assert best.strategy == "npu_only"


def test_solver_fails_loud_without_supports_or_units():
    class NoSupports:
        bw = AnalyticFakePlatform().bw
        def compute_us(self, n): return 0.0
    try:
        solve(SolverContext(platform=NoSupports(), units=("gpu",), host="gpu", sync_us=0.0), _dag(1, 64, 64), 1)
    except ValueError:
        pass
    else:
        raise AssertionError("missing supports() not rejected")
    try:
        solve(SolverContext(platform=AnalyticFakePlatform(), units=("mem",), host="mem", sync_us=0.0), _dag(1, 64, 64), 1)
    except ValueError:
        return
    raise AssertionError("no supporting unit not rejected")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok {fn.__name__}")
    print(f"\n{len(fns)} solver tests passed.")
