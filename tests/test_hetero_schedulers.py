"""ADR-0009 — heterogeneous schedulers (gpu_only / npu_only / heteroinfer / tiled_hetero) +
the config/runner wiring. Unit-level tests use the FakePlatform (tests/fake_platform.py);
the end-to-end runner tests use a TEMPORARY SHIM over the real Platform (below) because
main's Platform still raises for npu / gpu-matmul pricing ("npu pricing is Wave 2.2") —
worker A's pricing replaces the shim on merge. The shim lives ONLY in this test file.

    .venv/bin/pytest tests/test_hetero_schedulers.py
"""
import contextlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from simulator.runtime.workload import build_token_dag  # noqa: E402
from simulator.runtime.scheduler import (  # noqa: E402
    SCHEDULERS, Scheduler, SingleUnitScheduler, HeteroInferScheduler, TiledHeteroScheduler, plans_summary,
)
from simulator.runtime.config import SimConfig, _KNOWN_SCHEDULERS  # noqa: E402
from simulator.runtime.platform import Platform  # noqa: E402
from simulator.runtime import runner as runner_mod  # noqa: E402
from simulator.runtime import tiling  # noqa: E402
from fake_platform import AnalyticFakePlatform  # noqa: E402

MODEL = "llama-3.2-1b"
_SUPPORT = ("norm", "rope", "ffn", "softmax", "residual")


class Cfg:
    """Minimal cfg stand-in (SimConfig fields the schedulers read)."""
    model = MODEL
    units = {"cim": True, "gpu": True, "npu": True, "cpu": True}
    sync_mode = "fast"
    sync_us_override = None
    partition_grid = 8
    precision_boundary_placement = "consumer"


def _snap(dag):
    return [(n.id, n.category, n.unit, n.mem_domain, n.wl.extra.get("fixed_us")) for n in dag.nodes]


def test_registry_and_config_mirror():
    for name in ("gpu_only", "npu_only", "heteroinfer", "tiled_hetero"):
        assert name in SCHEDULERS and isinstance(SCHEDULERS[name], Scheduler) and SCHEDULERS[name].pipeline
    assert _KNOWN_SCHEDULERS == set(SCHEDULERS)
    assert SCHEDULERS["all_cim"].pipeline is False
    assert SCHEDULERS["heteroinfer"].required_units == ("gpu", "npu", "cpu")


def test_single_unit_schedulers_place_everything_on_the_unit():
    plat = AnalyticFakePlatform()
    for unit in ("gpu", "npu"):
        dag = SCHEDULERS[f"{unit}_only"].assign(build_token_dag(MODEL, "decode", 512), Cfg(), platform=plat)
        units = {n.category: set() for n in dag.nodes}
        for n in dag.nodes:
            units[n.category].add(n.unit)
        for c in ("matmul", "attention") + _SUPPORT:
            assert units[c] == {unit}
        assert units["kv_cache"] == {"mem"} and units["embedding"] == {"mem"}
        assert not any(n.category == "merge" for n in dag.nodes)
        assert sum(1 for n in dag.nodes if n.wl.extra.get("sync_us")) == 0      # no cross-unit boundary
        assert plans_summary(dag) == {"mem_fixed": 33, f"{unit}_fixed": 621}


def test_heteroinfer_placement_rules():
    plat = AnalyticFakePlatform(rates={"gpu": 1.0e6, "npu": 1.5e6})
    dag = SCHEDULERS["heteroinfer"].assign(build_token_dag(MODEL, "decode", 512), Cfg(), platform=plat)
    assert dag.is_acyclic()
    for n in dag.nodes:
        if n.category in _SUPPORT or n.category == "attention":
            assert n.unit == "gpu"                                    # the GPU block (§4.1)
        elif n.category == "matmul":
            assert n.unit in ("gpu", "npu")
        elif n.category == "merge":
            assert n.unit == "gpu" and n.wl.extra["fixed_us"] == 10.0   # merge on the host; decode fast sync
        elif n.category in ("kv_cache", "embedding"):
            assert n.unit == "mem"
    assert not any(n.category == "attention" and n.tile for n in dag.nodes)   # attention never split
    strategies = {p["strategy"] for p in dag.plans.values() if p["category"] == "matmul"}
    assert strategies <= {"gpu_only", "npu_only", "weight_centric"} and "weight_centric" in strategies
    assert all(p["dim"] in (None, "N") for p in dag.plans.values())
    # every NPU sub-node consumes from the GPU host -> pays one sync (charged on its convert or itself)
    for n in dag.nodes:
        if n.unit == "npu":
            payer = n if n.wl.extra.get("sync_us") else next(dag[d] for d in n.deps if dag[d].category == "convert")
            assert payer.wl.extra["sync_us"] == 10.0


def test_heteroinfer_prefill_uses_prefill_sync_and_activation_centric_when_needed():
    plat = AnalyticFakePlatform(rates={"gpu": 1.0e6, "npu": 8.0e6}, static_npu=True)
    dag = SCHEDULERS["heteroinfer"].assign(build_token_dag(MODEL, "prefill", 300), Cfg(), platform=plat)
    merges = [n for n in dag.nodes if n.category == "merge"]
    assert merges and all(m.wl.extra["fixed_us"] == 30.0 for m in merges)
    assert plans_summary(dag).get("activation_centric", 0) > 0        # 300 = 256 (NPU) + 44 (GPU)
    p = next(p for p in dag.plans.values() if p["strategy"] == "activation_centric")
    assert [u for u, _ in p["parts"]] == ["npu", "gpu"] and p["sizes"] == [256, 44]


def test_tiled_hetero_uses_all_dims_and_records_plans():
    plat = AnalyticFakePlatform(rates={"gpu": 1.0e6, "npu": 3.0e6, "cim": 6.0e6, "cpu": 0.1e6})
    dag = SCHEDULERS["tiled_hetero"].assign(build_token_dag(MODEL, "prefill", 256), Cfg(), platform=plat)
    assert dag.is_acyclic()
    strategies = plans_summary(dag)
    assert sum(strategies.values()) == len([p for p in dag.plans.values()])
    assert any(p["dim"] is not None for p in dag.plans.values())
    assert {p["category"] for p in dag.plans.values()} >= {"matmul", "attention", "norm", "embedding"}
    for p in dag.plans.values():
        assert "strategy" in p and "parts" in p
        if p["dim"]:
            assert p["dim"] in tiling.DIMS and "T_est_us" in p and p["n_candidates"] > 1
    units = {n.unit for n in dag.nodes}
    assert "cim" in units and "cpu" not in units                       # CIM used; cpu compute off by default


def test_tiled_hetero_respects_enabled_units_and_allow_flags():
    plat = AnalyticFakePlatform(rates={"gpu": 1.0e6, "npu": 3.0e6, "cim": 6.0e6, "cpu": 0.1e6})
    class NoCim(Cfg):
        units = {"cim": False, "gpu": True, "npu": True, "cpu": True}
    dag = SCHEDULERS["tiled_hetero"].assign(build_token_dag(MODEL, "decode", 512), NoCim(), platform=plat)
    assert "cim" not in {n.unit for n in dag.nodes}
    dag = TiledHeteroScheduler(allow_cim=False).assign(build_token_dag(MODEL, "decode", 512), Cfg(), platform=plat)
    assert "cim" not in {n.unit for n in dag.nodes}
    dag = TiledHeteroScheduler(allow_cpu_compute=True).assign(build_token_dag(MODEL, "decode", 512), Cfg(), platform=plat)
    assert all(n.unit in ("gpu", "npu", "cim", "cpu", "mem") for n in dag.nodes)
    class Nothing(Cfg):
        units = {"cim": False, "gpu": False, "npu": False, "cpu": True}
    try:
        SCHEDULERS["tiled_hetero"].assign(build_token_dag(MODEL, "decode", 512), Nothing(), platform=plat)
    except ValueError:
        pass
    else:
        raise AssertionError("no enabled compute unit not rejected")
    try:
        SCHEDULERS["tiled_hetero"].assign(build_token_dag(MODEL, "decode", 512), Cfg())
    except ValueError:
        return
    raise AssertionError("missing platform not rejected")


def test_tiled_hetero_reduces_to_heteroinfer():
    # BY CONSTRUCTION (same code path): tiled_hetero restricted to HeteroInfer's search space
    # (units {gpu,npu}, support + attention pinned to the GPU, matmul N-split only, no
    # attention/support splitting) yields identical plans and an identical DAG.
    plat = AnalyticFakePlatform(rates={"gpu": 1.0e6, "npu": 2.0e6})
    reduced = TiledHeteroScheduler(units=("gpu", "npu"), allow_cim=False, allow_cpu_compute=False,
                                   support_unit="gpu", matmul_dims=("N",), split_attention=False,
                                   split_support=False, host="gpu")
    for phase, L in (("decode", 512), ("prefill", 256), ("prefill", 300)):
        a = HeteroInferScheduler().assign(build_token_dag(MODEL, phase, L), Cfg(), platform=plat)
        b = reduced.assign(build_token_dag(MODEL, phase, L), Cfg(), platform=plat)
        assert _snap(a) == _snap(b)
        assert {k: (v["strategy"], v["dim"], v["parts"]) for k, v in a.plans.items()} == \
               {k: (v["strategy"], v["dim"], v["parts"]) for k, v in b.plans.items()}
        assert any(p["dim"] for p in a.plans.values())                      # some split happened


def test_assign_is_idempotent_for_tiling_schedulers():
    plat = AnalyticFakePlatform(rates={"gpu": 1.0e6, "npu": 3.0e6, "cim": 6.0e6})
    for name in ("heteroinfer", "tiled_hetero", "gpu_only"):
        d1 = SCHEDULERS[name].assign(build_token_dag(MODEL, "decode", 512), Cfg(), platform=plat)
        snap, plans = _snap(d1), dict(d1.plans)
        d2 = SCHEDULERS[name].assign(d1, Cfg(), platform=plat)
        assert _snap(d2) == snap and len(d2.nodes) == len(d1.nodes)
        assert {k: v["strategy"] for k, v in d2.plans.items()} == {k: v["strategy"] for k, v in plans.items()}


def test_sync_mode_changes_decisions_and_cost():
    # decode matmuls at the HeteroInfer scale (~400 us on the GPU; the NPU, padded to M=32, ~2x
    # faster): a fast sync (10 us) leaves the split worthwhile, clFinish (2 x 400 us) does not.
    plat = AnalyticFakePlatform(rates={"gpu": 1.0e4, "npu": 6.4e5}, bw_GBs=1e6, knee_GBs=1e6)
    class ClFinish(Cfg):
        sync_mode = "clfinish"
    class NoSync(Cfg):
        sync_mode = "none"
    fast = SCHEDULERS["heteroinfer"].assign(build_token_dag(MODEL, "decode", 512), Cfg(), platform=plat)
    slow = SCHEDULERS["heteroinfer"].assign(build_token_dag(MODEL, "decode", 512), ClFinish(), platform=plat)
    free = SCHEDULERS["heteroinfer"].assign(build_token_dag(MODEL, "decode", 512), NoSync(), platform=plat)
    n_split = lambda d: sum(1 for p in d.plans.values() if p["dim"])   # noqa: E731
    assert n_split(free) >= n_split(fast) >= n_split(slow)
    assert n_split(slow) < n_split(fast)                               # 400 us kills decode splits (§3.1)
    assert sum(n.wl.extra.get("sync_us", 0) for n in free.nodes) == 0
    assert all(m.wl.extra["fixed_us"] == 400.0 for m in slow.nodes if m.category == "merge")


def test_config_sync_and_grid_fields():
    cfg = SimConfig.from_dict({"workload": {"model": MODEL},
                               "scheduler": {"policy": "heteroinfer", "sync_mode": "clfinish", "partition_grid": 4}})
    assert cfg.sync_mode == "clfinish" and cfg.partition_grid == 4 and cfg.sync_us_override is None
    assert any("sync_mode" in p for p in cfg.provenance)
    for bad in ({"sync_mode": "bogus"}, {"sync_us_override": -1}, {"partition_grid": 1}, {"nope": 1}):
        try:
            SimConfig.from_dict({"workload": {"model": MODEL}, "scheduler": {"policy": "tiled_hetero", **bad}})
        except ValueError:
            continue
        raise AssertionError(f"{bad} accepted")


# --------------------------------------------------------------------------------------------
# TEMPORARY SHIM (runner e2e only): analytic npu / gpu-matmul / gpu-support / cpu-matmul pricing
# + supports(). Replaced by worker A's per-unit pricing + SoC profiles on merge. NOT silicon.
# --------------------------------------------------------------------------------------------
class _ShimPlatform(Platform):
    _MAC_US = {"gpu": 0.5e6, "npu": 2.0e6, "cpu": 0.05e6}       # MAC/us: ~1 TFLOPS GPU plateau, NPU 4x, CPU NEON

    def supports(self, unit, cat):
        if cat in ("merge", "convert"):
            return True
        if unit == "mem":
            return cat in ("embedding", "kv_cache")
        if unit == "cim":
            return cat in ("matmul", "attention")
        return cat in ("matmul", "attention", "sampling") + _SUPPORT

    def price(self, node):
        w, u, cat = node.wl, node.unit, node.category
        fixed = float(w.extra.get("fixed_us", 0.0))
        def out(lat, prov):
            return {"latency_us": fixed + lat, "compute_provenance": f"TEMP SHIM {prov}", "source_model": "shim"}
        if cat == "merge":
            return out(0.0, "merge = fixed_us")
        if u == "cim" or (u == "cpu" and cat in _SUPPORT) or (u == "gpu" and cat == "attention" and "hd" in w.extra):
            p = dict(super().price(node))
            p["latency_us"] += fixed
            return p
        if cat == "matmul":
            M = -(-w.M // 32) * 32 if u == "npu" else w.M
            return out(M * w.K * w.N / self._MAC_US[u] + (50.0 if u == "npu" else 0.0), f"{u} matmul roofline")
        if cat == "attention" and "hd" in w.extra:
            if node.pricing_group != node.id:
                return out(0.0, "grouped attention member")
            return out(2 * w.heads * tiling._attn_M(node) * w.extra["hd"] * w.kv / self._MAC_US.get(u, 0.5e6), f"{u} bmm")
        if cat == "convert":
            return out(0.0, "convert (memory-bound)")
        return out(5.0 + w.nbytes / 2e4, f"{u} elementwise (bytes/20 GB/s + launch)")


@contextlib.contextmanager
def _shimmed():
    orig = runner_mod.Platform
    runner_mod.Platform = _ShimPlatform
    try:
        yield
    finally:
        runner_mod.Platform = orig


def _run(policy, **sched):
    # knee_GBs=48: multi-unit bandwidth headroom (HeteroInfer §3.3: one unit ~40-45 GB/s, GPU+NPU
    # ~60 GB/s). With the default knee == eff_BW a split decode matvec gains nothing, so the
    # solver (correctly) keeps every decode matmul on the GPU and no sync is ever charged.
    cfg = SimConfig.from_dict({"workload": {"model": MODEL, "context": 1024, "prefill_len": 256},
                               "platform": {"units": {"cim": True, "gpu": True, "npu": True, "cpu": True}},
                               "tunables": {"knee_GBs": 48.0},
                               "scheduler": {"policy": policy, **sched}})
    with _shimmed():
        return runner_mod.run(cfg)


def test_runner_end_to_end_all_hetero_schedulers_with_shim():
    for policy in ("gpu_only", "npu_only", "heteroinfer", "tiled_hetero"):
        r = _run(policy)
        assert r["tok_s"] > 0 and r["prefill_tok_s"] > 0 and r["ttft_s_reported_not_gated"] > 0
        assert r["calibrated_anchor"] is False and any("simulated" in p for p in r["provenance"])
        assert set(r["plans_summary"]) == {"decode", "prefill"} and r["plans_summary"]["decode"]
        assert 0 <= r["achieved_decode_bandwidth_GBs"] <= 48.0 + 1e-6
        assert 0 < r["decode_critical_path_us"] and 0 < r["prefill_critical_path_us"]
        assert r["sync_mode"] == "fast"
        for u, util in r["unit_utilisation"].items():
            assert 0.0 <= util <= 1.0 + 1e-9, (policy, u, util)
        if policy in ("gpu_only", "npu_only"):
            assert r["sync_count_per_decode_token"] == 0 and r["merge_count_per_decode_token"] == 0
        else:
            assert r["sync_count_per_decode_token"] > 0 and r["sync_us_per_decode_token"] > 0
    # existing metric keys unchanged (AllCim path untouched; shim not involved in CIM pricing)
    r_all = runner_mod.run(SimConfig.from_dict({"workload": {"model": MODEL, "context": 1024}}))
    assert r_all["sync_count_per_decode_token"] == 0 and r_all["plans_summary"] == {"decode": {}, "prefill": {}}
    assert r_all["calibrated_anchor"] is True


def test_runner_rejects_missing_required_units():
    cfg = SimConfig.from_dict({"workload": {"model": MODEL}, "scheduler": {"policy": "heteroinfer"},
                               "platform": {"units": {"cim": True, "gpu": True, "npu": False, "cpu": True}}})
    with _shimmed():
        try:
            runner_mod.run(cfg)
        except ValueError:
            return
    raise AssertionError("heteroinfer with npu disabled not rejected")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok {fn.__name__}")
    print(f"\n{len(fns)} hetero-scheduler tests passed.")
