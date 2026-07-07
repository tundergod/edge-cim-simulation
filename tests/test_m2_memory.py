"""Phase 2.4 hardening — M2 memory model (simulator/models/m2_memory.py):
MemoryModel.predict fail-loud paths + the _bw_tag() honesty-tag branches.

    .venv/bin/pytest tests/test_m2_memory.py
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import simulator.models.m2_memory as m2_memory  # noqa: E402
from simulator.models.m2_memory import MemoryModel  # noqa: E402
from simulator.models.engine import Workload  # noqa: E402
from simulator.specs.loader import load_spec  # noqa: E402


def test_pcie_on_non_topology_spec_raises():
    mm = MemoryModel(load_spec("mem_lpddr4x"))          # a MEMORY spec: no 'topology' key
    with pytest.raises(ValueError):
        mm.predict(Workload(op="pcie", nbytes=1000))


def test_stream_raises_when_eff_bw_falsy():
    # cim_topo_alpha: no on-card DRAM and no mem_spec_ref -> eff_BW_GBs resolves to None
    mm = MemoryModel(load_spec("cim_topo_alpha"))
    assert not mm.eff_BW_GBs
    with pytest.raises(ValueError):
        mm.predict(Workload(op="stream", nbytes=1000))


# ---------------------------------------------------------------------------
# _bw_tag(): 7 branches. All 7 are reachable with real committed specs (no crafted
# spec dicts needed) — see the per-test comment for which branch each hits.
# ---------------------------------------------------------------------------

def test_bw_tag_lpddr4x_is_calibrated():
    assert "calibrated" in MemoryModel(load_spec("mem_lpddr4x"))._bw_tag()


def test_bw_tag_lpddr5_is_simulated():
    assert "simulated (eff 0.65" in MemoryModel(load_spec("mem_lpddr5"))._bw_tag()


def test_bw_tag_lpddr4_hits_default_assumption_branch():
    # memory_type 'LPDDR4' (not '4x') matches neither the LPDDR4x nor the LPDDR5 branch.
    assert MemoryModel(load_spec("mem_lpddr4"))._bw_tag() == \
        "assumption (eff derived from measured-4x efficiency)"


def test_bw_tag_edge_topology_is_assumption():
    tag = MemoryModel(load_spec("cim_topo_edge"))._bw_tag()
    assert tag.startswith("assumption (edge:")


def test_bw_tag_card_topology_is_calibrated():
    # cim_topo_card carries dram_type=LPDDR4x -> same calibrated tag as the bare mem_lpddr4x spec.
    assert "calibrated" in MemoryModel(load_spec("cim_topo_card"))._bw_tag()


def test_bw_tag_ramulator2_lpddr5_uses_cached_heavy_sim():
    # the Ramulator2 LPDDR5 cache IS committed in this repo -> real cache-hit branch.
    tag = MemoryModel(load_spec("mem_lpddr5"), engine="ramulator2")._bw_tag()
    assert "Ramulator2 LPDDR5 heavy sim" in tag


def test_bw_tag_ramulator2_na_for_non_lpddr5_spec():
    tag = MemoryModel(load_spec("mem_lpddr4x"), engine="ramulator2")._bw_tag()
    assert "N/A for this spec" in tag


def test_bw_tag_ramulator2_deferred_when_cache_absent():
    # simulate the Ramulator2 C++ build/cache being absent (a fresh checkout without the
    # build would hit this; the cache IS committed here, so patch the module path constant).
    orig = m2_memory._RAM2
    m2_memory._RAM2 = Path("/nonexistent/not_built.json")
    try:
        tag = MemoryModel(load_spec("mem_lpddr5"), engine="ramulator2")._bw_tag()
    finally:
        m2_memory._RAM2 = orig
    assert "cache is absent" in tag


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)
           and v.__code__.co_argcount == 0]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"\n{len(fns)} m2 memory tests passed (fixture-parametrized tests need pytest).")
