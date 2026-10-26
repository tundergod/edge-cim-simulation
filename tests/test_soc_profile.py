"""ADR-0008 — SoC profiles (sd8gen3_cim / rk3588_cim) + the legacy byte-identity guard.

(a) profile specs are well-formed and self-consistent (P2 memory == the edge topology's resolved wall;
    P2 knee == the Card-calibrated 4c/1c ratio applied to it);
(b) SimConfig.soc_profile: resolves the topology, rejects conflicts/unknowns/bw_efficiency, is
    provenance-flagged (never a calibrated anchor);
(c) Platform(soc_profile=): memory from the profile, knee override, unit model types per profile;
(d) P1 anchors (HeteroInfer Table 3) reproduced within the tolerance table below through
    Platform.price on gpu/npu matmul nodes; Fig.3 staircase monotone in 32-row steps; NPU order flag;
(e) soc_profile UNSET => pricing identical to main 9581de24 for the AllCim/CimHetero DAGs (snapshot of
    per-(unit,category) compute sums captured from the main checkout) and the L4 tok/s unchanged.

    .venv/bin/pytest tests/test_soc_profile.py      (from the repo root)
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "trace_export"))
from simulator.specs.loader import load_spec  # noqa: E402
from simulator.models.m2_memory import MemoryModel  # noqa: E402
from simulator.models.engine import Workload  # noqa: E402
from simulator.models.m4_gpu import MaliGpuModel  # noqa: E402
from simulator.models.m4_gpu_anchor import GpuAnchorRooflineModel  # noqa: E402
from simulator.models.m4_npu import NpuModel, NpuStaircaseModel  # noqa: E402
from simulator.runtime.config import SimConfig  # noqa: E402
from simulator.runtime.dag import OpNode  # noqa: E402
from simulator.runtime.platform import Platform, SOC_PROFILES  # noqa: E402
from simulator.runtime.workload import build_token_dag  # noqa: E402
from simulator.runtime.scheduler import SCHEDULERS  # noqa: E402

PROFILES = ("sd8gen3_cim", "sd8gen3_cim_paper", "rk3588_cim")
TOL = 0.15   # the ADR-0008 acceptance band on every HeteroInfer Table 3 anchor


# ---- (a) specs ------------------------------------------------------------------------------
def test_profile_specs_well_formed():
    assert set(SOC_PROFILES) == set(PROFILES)
    for name in PROFILES:
        s = load_spec(f"soc_{name}")
        assert s["profile"] == name and s["honesty"] and s["provenance"]
        for u in ("gpu", "npu", "cpu", "cim"):
            assert u in s["units"] and s["units"][u]["default_dtype"]
        assert s["memory"]["single_stream_eff_BW_GBs"] > 0 and s["memory"]["knee_GBs"] >= s["memory"]["single_stream_eff_BW_GBs"]
        assert "sync_us" not in s                      # rev F8: tiling.SYNC_US is the one sync table
        assert s["units"]["cim"]["topology"] in ("cim_topo_edge", "cim_topo_card", "cim_topo_alpha")
        for u in ("gpu", "npu", "cpu", "cim"):           # rev F2/F3/F13 per-unit parameters
            assert 0 < s["units"][u]["stream_eff"] <= 1.0 and s["units"][u]["min_split_rows"] >= 1
            assert s["units"][u]["weight_bytes_per_param"] in (0.5, 1.0)


def test_rk3588_profile_memory_matches_edge_topology_and_contention_ratio():
    s = load_spec("soc_rk3588_cim")
    edge = MemoryModel(load_spec("cim_topo_edge")).eff_BW_GBs
    assert s["memory"]["single_stream_eff_BW_GBs"] == pytest.approx(edge, abs=1e-6)   # 33.3 x 0.9 = 29.97 (pinned)
    rep = json.loads((ROOT / "validation/reports/phase2/contention.json").read_text())
    assert s["memory"]["knee_GBs"] == pytest.approx(edge * rep["card_4c_1c_mean"], rel=1e-3)


def test_sd8gen3_profile_numbers_are_the_published_ones():
    s = load_spec("soc_sd8gen3_cim")
    assert s["memory"]["single_stream_eff_BW_GBs"] == 43.0 and s["memory"]["knee_GBs"] == 60.0
    assert s["memory"]["practical_max_GBs"] == 61.9 and s["memory"]["peak_GBs"] == 68.0
    assert "borrowed" in s["honesty"]


# ---- (b) config -------------------------------------------------------------------------------
def test_config_soc_profile_resolves_topology_and_flags_provenance():
    for name in PROFILES:
        cfg = SimConfig.from_dict({"workload": {"model": "llama-3.2-1b"}, "platform": {"soc_profile": name}})
        assert cfg.soc_profile == name and cfg.topology == "cim_topo_edge"
        assert cfg.resolved_memory_spec() == "mem_lpddr5"
        assert not cfg.is_calibrated_anchor()
        assert any(name in p for p in cfg.provenance)
    # explicit matching topology is fine; a conflicting one is rejected
    SimConfig.from_dict({"workload": {"model": "llama-3.2-1b"},
                         "platform": {"soc_profile": "sd8gen3_cim", "topology": "cim_topo_edge"}})
    for bad in ({"soc_profile": "sd8gen3_cim", "topology": "cim_topo_card"},
                {"soc_profile": "nope"},
                {"soc_profile": "rk3588_cim", "bw_efficiency": 0.9}):
        with pytest.raises(ValueError):
            SimConfig.from_dict({"workload": {"model": "llama-3.2-1b"}, "platform": bad})
    # mutation after construction is caught at validate() (the run() boundary)
    cfg = SimConfig.from_dict({"workload": {"model": "llama-3.2-1b"}, "platform": {"soc_profile": "rk3588_cim"}})
    cfg.topology = "cim_topo_card"
    with pytest.raises(ValueError):
        cfg.validate()


def test_config_without_profile_unchanged():
    cfg = SimConfig.from_dict({"workload": {"model": "llama-3.2-1b"}})
    assert cfg.soc_profile is None and cfg.topology == "cim_topo_card" and cfg.is_calibrated_anchor()


# ---- (c) platform -------------------------------------------------------------------------------
def test_platform_profile_memory_units_and_precedence():
    p1 = Platform("llama-3.2-1b", soc_profile="sd8gen3_cim")
    assert p1.bw.eff_BW == 43.0 and p1.bw.knee == 60.0 and p1.per_call_floor_us == 0.0
    assert isinstance(p1.gpu, GpuAnchorRooflineModel) and isinstance(p1.npu, NpuStaircaseModel)
    assert p1.cim_attn is not None and p1.cim.n_cores == 4
    from simulator.runtime import tiling
    assert p1.sync_us == tiling.SYNC_US and p1.sync_us["fast"] == {"decode": 10.0, "prefill": 30.0}
    assert p1.bw.unit_caps == {"gpu": 43.0, "npu": 43.0, "cpu": 43.0, "cim": 43.0, "mem": 43.0}
    assert p1.weight_bytes_per_param == {"gpu": 0.5, "npu": 1.0, "cpu": 0.5, "cim": 1.0}
    assert p1.min_split_rows["npu"] == 1024 and p1.cim.scale_mode == "physical"
    p2 = Platform("llama-3.2-1b", soc_profile="rk3588_cim")
    assert p2.bw.eff_BW == 29.97 and p2.bw.knee == pytest.approx(33.03)
    assert p2.bw.unit_cap("npu") == pytest.approx(29.97 * 0.67) and p2.bw.unit_cap("gpu") == 29.97
    legacy = Platform("llama-3.2-1b")
    assert legacy.bw.unit_caps is None and legacy.weight_bytes_per_param == {u: 1.0 for u in ("gpu", "npu", "cpu", "cim")}
    assert legacy.cim.scale_mode == "legacy" and legacy.sync_us == tiling.SYNC_US
    assert isinstance(p2.gpu, MaliGpuModel) and type(p2.npu) is NpuModel
    # explicit knee (sensitivity knob) wins over the profile knee; conflicts are rejected
    assert Platform("llama-3.2-1b", soc_profile="sd8gen3_cim", knee_GBs=50.0).bw.knee == 50.0
    assert Platform("llama-3.2-1b", soc_profile="sd8gen3_cim", topology="cim_topo_edge").bw.eff_BW == 43.0
    with pytest.raises(ValueError):
        Platform("llama-3.2-1b", soc_profile="sd8gen3_cim", topology="cim_topo_card")
    with pytest.raises(ValueError):
        Platform("llama-3.2-1b", soc_profile="sd8gen3_cim", bw_efficiency=0.9)
    with pytest.raises(ValueError):
        Platform("llama-3.2-1b", soc_profile="bogus")


# ---- (d) P1 anchors --------------------------------------------------------------------------
# HeteroInfer SOSP'25 Table 3 (8 Gen 3): weight [N_out, K_in], activation [K_in, M]; latency us.
ANCHORS = [  # (unit, N_out, K_in, M, latency_us, tolerance)
    ("gpu", 4096, 4096, 1, 511, TOL), ("gpu", 4096, 4096, 128, 7306, TOL), ("gpu", 4096, 4096, 256, 10841, TOL),
    ("gpu", 4096, 14336, 1, 1467, TOL), ("gpu", 4096, 14336, 256, 35231, TOL), ("gpu", 28672, 4096, 1, 1903, TOL),
    ("npu", 4096, 4096, 1, 693, TOL), ("npu", 4096, 4096, 128, 912, TOL), ("npu", 4096, 4096, 256, 1884, TOL),
    ("npu", 4096, 14336, 1, 6506, TOL), ("npu", 4096, 14336, 256, 23445, TOL), ("npu", 28672, 4096, 1, 3886, TOL),
]


def _mm(unit, N, K, M, **extra):
    return OpNode(id=7, category="matmul", unit=unit,
                  wl=Workload(op="matmul", M=M, K=K, N=N, dtype="fp16", extra=extra))


def test_p1_table3_anchors_within_tolerance():
    """Anchors under the runtime's rule (rev F2): pred = max(kernel, K*N*wbpp / 43 GB/s). The residual
    is reported per unit so a refit that drifts shows up, and the GPU fit (rev F3) must beat the band
    by a margin (max residual <= 0.06, was 0.147 at the band edge)."""
    plat = Platform("llama-3.1-8b", soc_profile="sd8gen3_cim")
    worst = {"gpu": 0.0, "npu": 0.0}
    for unit, N, K, M, t, tol in ANCHORS:
        r = plat.price(_mm(unit, N, K, M))
        pred = max(r["latency_us"], K * N * plat.weight_bytes_per_param[unit] / (plat.bw.eff_BW * 1e3))
        resid = abs(pred / t - 1)
        worst[unit] = max(worst[unit], resid)
        assert resid <= tol, (unit, N, K, M, t, pred)
        assert "borrowed" in r["compute_provenance"]
    print(f"P1 anchor max residual: gpu {worst['gpu']:.3f} npu {worst['npu']:.3f}")
    assert worst["gpu"] <= 0.06 and worst["npu"] <= TOL
    rep = json.loads((ROOT / "validation/reports/hetero/anchor_fit.json").read_text())
    assert rep["pass_all"] and rep["gpu_sd8gen3"]["params"]["max_abs_rel_residual"] == pytest.approx(worst["gpu"], abs=1e-3)
    # small-shape plausibility (rev F3): a 1 MB (INT8) matvec is a launch + its work, not a 300 us constant
    small = plat.price(_mm("gpu", 512, 2048, 1))["latency_us"]
    assert plat.gpu.gemv_floor_us <= 60.0 and small <= 0.5 * plat.price(_mm("gpu", 4096, 4096, 1))["latency_us"]
    assert small < 200.0, small


def test_p1_npu_staircase_and_order_flag():
    plat = Platform("llama-3.1-8b", soc_profile="sd8gen3_cim")
    lat = lambda M, **e: plat.price(_mm("npu", 4096, 14336, M, **e))["latency_us"]  # noqa: E731
    assert lat(1) == lat(32) < lat(33) == lat(64) < lat(65)          # Fig.3: flat within a 32-row tile
    f = plat.npu.floor_us                                                 # Fig.4 penalty on the work, not the floor
    assert (lat(32, order="activation_stationary") - f) / (lat(32) - f) == pytest.approx(6.0)
    with pytest.raises(ValueError):
        lat(32, order="sideways")
    # Fig.3 cross-check (not fitted): levels for [4096,14336] at steps 1..6 within +/-10%
    for i, lvl in enumerate(load_spec("npu_hexagon_sd8gen3")["fig3_staircase_4096x14336_us"]):
        assert abs(lat(32 * (i + 1)) / lvl - 1) <= 0.10


def test_p1_gpu_regimes_monotone():
    plat = Platform("llama-3.1-8b", soc_profile="sd8gen3_cim")
    lat = lambda M: plat.price(_mm("gpu", 4096, 4096, M))["latency_us"]  # noqa: E731
    xs = [lat(M) for M in (1, 2, 4, 8, 16, 64, 128, 256, 512)]
    assert all(a < b for a, b in zip(xs, xs[1:]))                       # never cheaper with more rows
    assert lat(2) <= 2 * lat(1)                                          # bridging = min(loop, gemm)


# ---- (e) legacy byte-identity -------------------------------------------------------------------
# per-(unit,category) compute sums (us) captured on main 9581de24 (AllCim + CimHetero placements).
SNAPSHOT = {
    "llama-3.2-1b/all_cim/decode@512": {
        "cim/attention": 0.0, "cim/matmul": 32425.317402, "cpu/ffn": 3184.724009, "cpu/norm": 4053.836169,
        "cpu/residual": 48.665088, "cpu/rope": 4468.006873, "cpu/softmax": 3351.082092,
        "mem/embedding": 0.0, "mem/kv_cache": 0.0},
    "llama-3.2-1b/cim_hetero/decode@512": {
        "cim/convert": 0.0, "cim/matmul": 32425.317402, "cpu/attention": 0.0, "cpu/ffn": 3184.724009,
        "cpu/norm": 4053.836169, "cpu/residual": 48.665088, "cpu/rope": 4468.006873, "cpu/softmax": 3351.082092,
        "gpu/attention": 130318.86848, "gpu/convert": 0.0, "mem/embedding": 0.0, "mem/kv_cache": 0.0},
    "llama-3.1-8b/all_cim/prefill@128": {
        "cim/attention": 0.0, "cim/matmul": 94616.9714, "cpu/ffn": 11146.534031, "cpu/norm": 9411.807637,
        "cpu/residual": 144.164353, "cpu/rope": 10205.889045, "cpu/softmax": 2053.506264, "mem/embedding": 0.0},
    "llama-3.1-8b/cim_hetero/decode@512": {
        "cim/convert": 0.0, "cim/matmul": 209566.126246, "cpu/attention": 0.0, "cpu/ffn": 11146.534031,
        "cpu/norm": 9411.807637, "cpu/residual": 144.164353, "cpu/rope": 10205.889045, "cpu/softmax": 6702.164184,
        "gpu/attention": 260637.73696, "gpu/convert": 0.0, "mem/embedding": 0.0, "mem/kv_cache": 0.0},
}
L4_TOK_S = {"llama-3.2-1b": 14.4727, "llama-3.2-3b": 5.9659, "llama-3.1-8b": 2.7841}


def test_legacy_pricing_identical_to_main_snapshot():
    for key, expect in SNAPSHOT.items():
        model, pol, rest = key.split("/")
        phase, L = rest.split("@")
        cfg = SimConfig.from_dict({"workload": {"model": model}, "scheduler": {"policy": pol}})
        plat = Platform(model, memory_spec="mem_lpddr4x", topology="cim_topo_card")
        assert plat.profile is None and plat.cim_attn is None
        dag = SCHEDULERS[pol].assign(build_token_dag(model, phase, int(L)), cfg)
        got = {}
        for n in dag.nodes:
            got[f"{n.unit}/{n.category}"] = got.get(f"{n.unit}/{n.category}", 0.0) + plat.price(n)["latency_us"]
        assert set(got) == set(expect), key
        for k, v in expect.items():
            assert round(got[k], 6) == v, (key, k, got[k], v)


def test_l4_tok_s_unchanged_without_profile():
    from simulator.runtime.runner import run
    for model, tok_s in L4_TOK_S.items():
        r = run(SimConfig.from_dict({"workload": {"model": model, "context": 1024},
                                     "platform": {"memory_spec": "mem_lpddr4x", "topology": "cim_topo_card"},
                                     "scheduler": {"policy": "all_cim"}}))
        assert r["tok_s"] == tok_s, (model, r["tok_s"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
