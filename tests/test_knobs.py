"""Scalable / flexible knobs on top of ADR-0008/0009: `scheduler.host` (gpu|npu|cim|auto),
`platform.cim_n_cores`, `platform.npu_scale`, and the runner's echo of the values actually
used (host, cim_n_cores, npu_scale, knee_GBs, soc_profile). All SIMULATED / EXTRAPOLATED.

    .venv/bin/pytest tests/test_knobs.py
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from simulator.models.engine import Workload  # noqa: E402
from simulator.runtime.config import SimConfig  # noqa: E402
from simulator.runtime.dag import OpNode  # noqa: E402
from simulator.runtime.platform import Platform  # noqa: E402
from simulator.runtime.runner import run  # noqa: E402
from simulator.runtime.scheduler import SCHEDULERS, TiledHeteroScheduler  # noqa: E402
from simulator.runtime.workload import build_token_dag  # noqa: E402
from fake_platform import AnalyticFakePlatform  # noqa: E402

MODEL = "llama-3.2-1b"


class Cfg:
    model = MODEL
    units = {"cim": True, "gpu": True, "npu": True, "cpu": True}
    sync_mode = "fast"
    sync_us_override = None
    partition_grid = 8
    precision_boundary_placement = "consumer"
    host = None

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _cfg(**kw):
    d = {"workload": {"model": MODEL, "context": 1024, "prefill_len": 256, "decode_len": 8},
         "platform": {"soc_profile": "sd8gen3_cim", "units": {"gpu": True, "npu": True, "cim": True, "cpu": True}},
         "scheduler": {"policy": "tiled_hetero"}}
    for k, v in kw.items():
        d.setdefault(k, {}).update(v)
    return SimConfig.from_dict(d)


# ---- config contract ----------------------------------------------------------------------
def test_config_accepts_knobs_and_flags_provenance():
    cfg = _cfg(scheduler={"host": "auto"}, platform={"cim_n_cores": 8, "npu_scale": 2.0})
    assert cfg.host == "auto" and cfg.cim_n_cores == 8 and cfg.npu_scale == 2.0
    assert any("cim_n_cores override = 8" in p for p in cfg.provenance)
    assert any("npu_scale = 2.0" in p for p in cfg.provenance)
    # the profile default / scale 1 are NOT flagged (no extra off-anchor claim)
    cfg = _cfg(platform={"cim_n_cores": 4, "npu_scale": 1.0})
    assert not any("cim_n_cores" in p or "npu_scale" in p for p in cfg.provenance)
    # defaults unchanged + the L4 config stays a calibrated anchor
    base = SimConfig.from_dict({"workload": {"model": MODEL, "context": 1024}})
    assert base.host is None and base.cim_n_cores is None and base.npu_scale == 1.0 and base.calibrated_anchor is True if hasattr(base, "calibrated_anchor") else base.is_calibrated_anchor()


@pytest.mark.parametrize("bad", [
    {"scheduler": {"host": "cpu"}}, {"scheduler": {"host": ""}},
    {"platform": {"cim_n_cores": 0}}, {"platform": {"cim_n_cores": 2.5}}, {"platform": {"cim_n_cores": True}},
    {"platform": {"npu_scale": 0}}, {"platform": {"npu_scale": -1}}, {"platform": {"npu_scale": "2"}},
])
def test_config_rejects_bad_knobs(bad):
    with pytest.raises(ValueError):
        _cfg(**bad)


# ---- platform: the knobs reach the unit models --------------------------------------------
def _mm(M, K, N, unit):
    return OpNode(id=0, category="matmul", unit=unit, wl=Workload(op="matmul", M=M, K=K, N=N, dtype="int8"),
                  bytes_streamed=K * N, deps=[])


def test_cim_n_cores_override_changes_tile_width_and_latency():
    p4 = Platform(MODEL, soc_profile="sd8gen3_cim")
    p8 = Platform(MODEL, soc_profile="sd8gen3_cim", cim_n_cores=8)
    p1 = Platform(MODEL, soc_profile="sd8gen3_cim", cim_n_cores=1)
    assert (p4.cim_n_cores, p8.cim_n_cores, p1.cim_n_cores) == (4, 8, 1)
    assert p8.cim.width == 8 * p4.cim.core_width and p1.cim.width == p4.cim.core_width
    # decode (M=1): the core count sets the pass width W = n_cores*512 (single-tile fit when both dims
    # <= W, else the multi-tile cliff model, which does NOT depend on n_cores) -> n=1 (W=512) prices a
    # K=1024 GEMV through the cliff model, n=4 / n=8 through the single-tile fit (identical: one pass).
    n = _mm(1, 1024, 2048, "cim")
    t4, t8, t1 = (p.compute_us(n) for p in (p4, p8, p1))
    assert t8 == pytest.approx(t4) and t1 != pytest.approx(t4)
    # prefill (M>1): tile area K*N/W^2 x (a + b*M) -> the per-GEMM latency scales as 1/n_cores^2
    m = _mm(128, 2048, 2048, "cim")
    u4, u8, u1 = (p.compute_us(m) for p in (p4, p8, p1))
    assert u8 == pytest.approx(u4 / 4) and u1 == pytest.approx(u4 * 16)
    # legacy (no profile) path: default untouched, override honoured
    assert Platform(MODEL).cim_n_cores == 4 and Platform(MODEL, cim_n_cores=2).cim.width == 2 * p4.cim.core_width


def test_npu_scale_multiplies_compute_throughput_only():
    p1 = Platform(MODEL, soc_profile="sd8gen3_cim")
    p2 = Platform(MODEL, soc_profile="sd8gen3_cim", npu_scale=2.0)
    assert p1.npu_scale == 1.0 and p2.npu_scale == 2.0
    assert p2.npu.tops_eff == pytest.approx(2.0 * p1.npu.tops_eff) and p2.npu.bw_w == p1.npu.bw_w
    pre = _mm(256, 4096, 4096, "npu")                 # compute-heavy prefill GEMM: shrinks
    dec = _mm(1, 4096, 4096, "npu")                   # memory-bound GEMV: barely moves
    assert p2.compute_us(pre) < 0.75 * p1.compute_us(pre)
    # decode GEMV: only the per-step streamed-activation term shrinks (the weight-load term is the
    # bandwidth, untouched) -> a smaller relative gain than the compute-bound prefill GEMM
    assert p2.compute_us(dec) < p1.compute_us(dec)
    assert p2.compute_us(dec) / p1.compute_us(dec) > p2.compute_us(pre) / p1.compute_us(pre)
    # the RKNPU2 analytic model (rk3588 profile) scales its datasheet TOPS the same way
    q2 = Platform(MODEL, soc_profile="rk3588_cim", npu_scale=4.0)
    assert q2.npu.tops == pytest.approx(4.0 * Platform(MODEL, soc_profile="rk3588_cim").npu.tops)
    with pytest.raises(ValueError):
        Platform(MODEL, soc_profile="sd8gen3_cim", npu_scale=0.0)


# ---- scheduler host -----------------------------------------------------------------------
def test_cfg_host_sets_merge_unit_and_is_recorded():
    plat = AnalyticFakePlatform()
    for host in ("gpu", "npu", "cim"):
        dag = SCHEDULERS["tiled_hetero"].assign(build_token_dag(MODEL, "decode", 512), Cfg(host=host), platform=plat)
        assert dag.host == host and dag.host_candidates is None if hasattr(dag, "host_candidates") else dag.host == host
        merges = [n for n in dag.nodes if n.category == "merge"]
        assert merges and all(n.unit == host for n in merges)
    with pytest.raises(ValueError):
        SCHEDULERS["tiled_hetero"].assign(build_token_dag(MODEL, "decode", 512), Cfg(host="cim", units={"gpu": True, "npu": True, "cim": False, "cpu": True}), platform=plat)
    # constructor host wins over cfg.host (heteroinfer stays GPU-hosted)
    dag = SCHEDULERS["heteroinfer"].assign(build_token_dag(MODEL, "decode", 512), Cfg(host="npu"), platform=plat)
    assert dag.host == "gpu"
    # single-unit baselines record their unit as the host
    assert SCHEDULERS["npu_only"].assign(build_token_dag(MODEL, "decode", 512), Cfg(), platform=plat).host == "npu"


def _snap(dag):
    return sorted((n.id, n.category, n.unit, n.wl.extra.get("fixed_us")) for n in dag.nodes)


def test_host_auto_picks_lowest_estimate_and_is_idempotent():
    plat = AnalyticFakePlatform()
    sched = SCHEDULERS["tiled_hetero"]
    auto = sched.assign(build_token_dag(MODEL, "decode", 512), Cfg(host="auto"), platform=plat)
    est = auto.host_candidates
    assert set(est) == {"gpu", "npu", "cim"} and all(v > 0 for v in est.values())
    assert auto.host == min(est, key=est.get)
    # == the explicit-host run of the chosen host (same decisions, same estimate)
    fixed = sched.assign(build_token_dag(MODEL, "decode", 512), Cfg(host=auto.host), platform=plat)
    assert _snap(fixed) == _snap(auto) and fixed.host_est_us == pytest.approx(auto.host_est_us)
    for h, e in est.items():                      # every candidate estimate is reproducible
        other = sched.assign(build_token_dag(MODEL, "decode", 512), Cfg(host=h), platform=plat)
        assert other.host_est_us == pytest.approx(e)
    # idempotent: a re-assign keeps the chosen host and changes nothing
    again = sched.assign(auto, Cfg(host="auto"), platform=plat)
    assert again.host == auto.host and _snap(again) == _snap(auto)
    # the input DAG is not mutated by the candidate solves (deep copies)
    fresh = build_token_dag(MODEL, "decode", 512)
    before = _snap(fresh)
    sched.assign(fresh, Cfg(host="auto"), platform=plat)
    assert _snap(fresh) == before


def test_tiled_hetero_constructor_host_still_wins():
    plat = AnalyticFakePlatform()
    dag = TiledHeteroScheduler(host="npu").assign(build_token_dag(MODEL, "decode", 512), Cfg(host="auto"), platform=plat)
    assert dag.host == "npu"


# ---- runner echo --------------------------------------------------------------------------
def test_runner_echoes_knobs_actually_used():
    r = run(_cfg(scheduler={"host": "auto"}, platform={"cim_n_cores": 2, "npu_scale": 0.5}, tunables={"knee_GBs": 50.0}))
    assert r["host"] in ("gpu", "npu", "cim") and r["prefill_host"] in ("gpu", "npu", "cim")
    assert set(r["host_candidates_est_us"]) == {"gpu", "npu", "cim"}
    assert r["cim_n_cores"] == 2 and r["npu_scale"] == 0.5 and r["knee_GBs"] == 50.0 and r["soc_profile"] == "sd8gen3_cim"
    assert r["tok_s"] > 0 and r["prefill_tok_s"] > 0
    r = run(_cfg())
    assert r["host"] == "gpu" and r["host_candidates_est_us"] is None and r["cim_n_cores"] == 4
    assert r["npu_scale"] == 1.0 and r["knee_GBs"] == 60.0
    # legacy AllCim path: knobs absent, echo = defaults, no new provenance
    r = run(SimConfig.from_dict({"workload": {"model": MODEL, "context": 1024}}))
    assert r["host"] is None and r["cim_n_cores"] == 4 and r["npu_scale"] == 1.0 and r["soc_profile"] is None
    assert r["knee_GBs"] == r["memory_eff_BW_GBs"] and r["calibrated_anchor"] is True
