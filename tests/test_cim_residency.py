"""ADR-0008 "Revisions 2026-10-26" (R3) — CIM beyond-capacity on the profile path: the Card's ~70 GOP/s
DRAM spill floor is a Metis AIPU artifact; physical mode re-programs the non-resident weights from the
shared pool instead: latency = max(resident compute, bytes_not_resident / program_BW_GBs). Legacy path
(no profile) keeps the Card cliff byte-identically (the L4 gate).

    .venv/bin/pytest tests/test_cim_residency.py      (from the repo root)
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "trace_export"))
from simulator.models.engine import Workload  # noqa: E402
from simulator.models.m1_cim_tile import CimTileModel  # noqa: E402
from simulator.runtime.dag import OpNode  # noqa: E402
from simulator.runtime.platform import Platform  # noqa: E402
from simulator.specs.loader import load_spec  # noqa: E402

MODEL = "llama-3.1-8b"
PROFILE = "sd8gen3_cim"


def _mm(unit, M, K, N):
    return OpNode(id=7, category="matmul", unit=unit,
                  wl=Workload(op="matmul", M=M, K=K, N=N, dtype="int8", extra={}))


def _plat(n_cores=None, **kw):
    return Platform(MODEL, soc_profile=PROFILE, cim_n_cores=n_cores, **kw)


def test_r3_profile_parameters_and_defaults():
    spec = load_spec(f"soc_{PROFILE}")["units"]["cim"]
    p = _plat()
    assert spec["program_BW_GBs"] == 43.0 == p.cim_program_BW_GBs == p.cim.program_BW_GBs == p.bw.unit_cap("cim")
    assert spec["residency_policy"] == "pin_first" == p.cim_residency_policy == p.cim.residency_policy
    assert p.cim.weight_bytes_per_param == 1.0
    p2 = Platform(MODEL, soc_profile="rk3588_cim")
    assert p2.cim.program_BW_GBs == pytest.approx(29.97) == p2.bw.unit_cap("cim")
    leg = Platform(MODEL)
    assert leg.cim_program_BW_GBs is None and leg.cim_residency_policy is None and leg.cim.program_BW_GBs is None
    with pytest.raises(ValueError):
        CimTileModel(scale_mode="physical")                       # physical mode needs program_BW_GBs
    with pytest.raises(ValueError):
        CimTileModel(scale_mode="physical", program_BW_GBs=0)
    with pytest.raises(ValueError):
        CimTileModel(residency_policy="lru")
    with pytest.raises(ValueError):
        CimTileModel(weight_bytes_per_param=0)


def test_r3_decode_beyond_capacity_is_max_of_resident_compute_and_reprogramming_no_floor():
    K, N = 4096, 4096                                     # K*N 16.8M > knee 8.16M at 4 cores
    p4 = _plat(4)
    c = p4.cim
    assert c.spills(K, N) and c.mt_knee == pytest.approx(c.mt_knee_kn)
    bytes_nr = (K * N - c.mt_knee) * 1.0
    program = bytes_nr / (43.0 * 1e9) * 1e6
    resident = c.mt_resident_a_us + c.mt_resident_b_us * K * N
    assert c.bytes_not_resident(K, N) == pytest.approx(bytes_nr) and c.program_us(K, N) == pytest.approx(program)
    t = p4.compute_us(_mm("cim", 1, K, N))
    assert t == pytest.approx(max(resident, program)) and t == pytest.approx(program) and program > resident
    assert t == pytest.approx(bytes_nr / 43e9 * 1e6)      # = bytes_not_resident / program_BW at 1 B/param
    floor = 2.0 * K * N / (c.mt_floor_gops * 1e9) * 1e6   # the Card's DRAM-spill floor (legacy only)
    assert t < floor and Platform(MODEL).compute_us(_mm("cim", 1, K, N)) == pytest.approx(floor)
    prov = p4.price(_mm("cim", 1, K, N))["compute_provenance"]
    assert "BEYOND RESIDENT CAPACITY" in prov and "EXTRAPOLATED" in prov and "no silicon" in prov
    # enough cores to make the weight set resident (knee x s >= K*N): the cost drops to the tile law
    p16 = _plat(16)
    assert not p16.cim.spills(K, N) and p16.cim.program_us(K, N) == 0.0
    t16 = p16.compute_us(_mm("cim", 1, K, N))                # W = 8192: a single tile, the G_eff tiling law
    assert p16.cim.width == 8192 and t16 == pytest.approx(p16.cim._decode_lat_us(K, N)) and t16 < t
    p6 = _plat(6)                                             # W = 3072: multi-tile, resident (knee 12.2M) for 4096x2048
    assert not p6.cim.spills(4096, 2048) and p6.compute_us(_mm("cim", 1, 4096, 2048)) == pytest.approx(
        c.mt_resident_a_us + c.mt_resident_b_us / 1.5 * 4096 * 2048)
    assert "BEYOND RESIDENT CAPACITY" not in p16.price(_mm("cim", 1, K, N))["compute_provenance"]
    # within capacity the physical path at s = 1 is still the legacy numerics
    for node in (_mm("cim", 1, 2048, 2048), _mm("cim", 1, 4096, 1024)):
        assert p4.compute_us(node) == Platform(MODEL).compute_us(node)


def test_r3_no_cliff_at_the_knee_and_program_bw_policy_bytes_sensitivity():
    K = 4096
    c = _plat(4).cim
    n_lo = int(c.mt_knee // K)                            # K*N just below / just above the knee
    lo, hi = c.dev_lat_us(1, K, n_lo), c.dev_lat_us(1, K, n_lo + 1)
    assert lo <= hi < lo * 1.01                           # continuous: no 3.5x throughput collapse
    K, N = 4096, 4096
    # a slow array-write path: program term scales 1/program_BW and dominates
    slow = CimTileModel(n_cores=4, scale_mode="physical", program_BW_GBs=10.0)
    assert slow.program_us(K, N) == pytest.approx(c.program_us(K, N) * 43.0 / 10.0)
    assert slow.dev_lat_us(1, K, N) == pytest.approx(slow.program_us(K, N))
    # residency_policy 'none': the whole weight set is re-programmed per use
    none = CimTileModel(n_cores=4, scale_mode="physical", program_BW_GBs=43.0, residency_policy="none")
    assert none.bytes_not_resident(K, N) == pytest.approx(K * N * 1.0) and K * N > c.bytes_not_resident(K, N)
    assert none.bytes_not_resident(2048, 2048) == 0.0     # within capacity nothing is re-programmed
    # the cim_weight_bytes_per_param knob (W4 storage) halves the re-programmed bytes
    p05 = _plat(4, cim_weight_bytes_per_param=0.5)
    assert p05.cim.weight_bytes_per_param == 0.5
    assert p05.cim.bytes_not_resident(K, N) == pytest.approx(c.bytes_not_resident(K, N) / 2)
    assert p05.compute_us(_mm("cim", 1, K, N)) == pytest.approx(max(c._decode_resident_lat_us(K, N), c.program_us(K, N) / 2))
    # physical mode: capacity is capacity, a single-tile shape past knee x s is beyond it too (8 cores, W=4096)
    p8 = _plat(8)
    assert p8.cim.width == 4096 and p8.cim.spills(4096, 4096) and 0 < p8.cim.program_us(4096, 4096) < 20


def test_r3_prefill_beyond_capacity_same_rule_once_per_forward():
    p4 = _plat(4)
    c = p4.cim
    W = c.n_cal * c.core_width
    for M, K, N in ((256, 4096, 4096), (256, 4096, 14336), (128, 4096, 14336)):
        tile_law = (K * N) / (W * W) * (c.prefill_a_us + c.prefill_b_us * M)
        assert p4.compute_us(_mm("cim", M, K, N)) == pytest.approx(max(tile_law, c.program_us(K, N)))
    assert p4.compute_us(_mm("cim", 256, 4096, 4096)) == pytest.approx(
        (4096 * 4096) / (W * W) * (c.prefill_a_us + c.prefill_b_us * 256))   # tile law dominates
    assert p4.compute_us(_mm("cim", 256, 4096, 14336)) == pytest.approx(c.program_us(4096, 14336))   # program dominates
    assert p4.compute_us(_mm("cim", 256, 4096, 14336)) > p4.compute_us(_mm("cim", 128, 4096, 14336)) * 0.99
    # the program term is per forward, not per row: M=256 and M=128 share it
    assert c.program_us(4096, 14336) == pytest.approx((4096 * 14336 - c.mt_knee) / 43e9 * 1e6)


def test_r3_legacy_path_keeps_the_card_cliff():
    leg = Platform(MODEL).cim
    assert leg.scale_mode == "legacy" and leg.program_us(8192, 2048) == 0.0 and leg.program_BW_GBs is None
    assert leg.spills(8192, 2048) and not leg.spills(2048, 2048)
    assert leg.bytes_not_resident(8192, 2048) == 8192 * 2048 - leg.mt_knee_kn    # informational: the spilled bytes
    assert leg.dev_lat_us(1, 8192, 2048) == pytest.approx(2.0 * 8192 * 2048 / (leg.mt_floor_gops * 1e9) * 1e6)
    assert leg.dev_lat_us(1, 2048, 3900) == pytest.approx(leg.mt_resident_a_us + leg.mt_resident_b_us * 2048 * 3900)   # 7.99M < knee
    assert leg.dev_lat_us(256, 4096, 4096) == pytest.approx(4.0 * (leg.prefill_a_us + leg.prefill_b_us * 256))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
