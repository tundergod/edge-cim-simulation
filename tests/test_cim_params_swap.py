"""Phase 2.5 — the CIM compute-engine swap surface (cim_compute_params).

Proves the MECHANISM (compute params are resolved from the topology spec and actually
drive compute; a non-Metis params file is provenance-tagged `simulated`) WITHOUT
committing any example non-Metis architecture as a result — all assertions use the
committed Metis specs or in-test fixtures.

    .venv/bin/pytest tests/test_cim_params_swap.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.runtime.platform import Platform  # noqa: E402
from simulator.runtime.config import SimConfig  # noqa: E402
from simulator.models.m1_cim_tile import CimTileModel  # noqa: E402
import simulator.runtime.config as cfgmod  # noqa: E402
import simulator.runtime.platform as platmod  # noqa: E402

_M1 = ROOT / "simulator/models/params/m1_cim.json"


def _cfg(topology, memory_spec=None):
    plat = {"topology": topology}
    if memory_spec is not None:
        plat["memory_spec"] = memory_spec
    return SimConfig.from_dict({"workload": {"model": "llama-3.1-8b", "context": 1024},
                                "platform": plat, "scheduler": {"policy": "all_cim"}})


def test_card_cim_params_equal_metis_default():
    """(a) card resolves cim_compute_params to m1_cim.json -> byte-identical to CimTileModel()."""
    p = Platform("llama-3.1-8b", topology="cim_topo_card", memory_spec="mem_lpddr4x")
    d = CimTileModel()
    assert p.cim.dev_lat_us(1, 2048, 2048) == d.dev_lat_us(1, 2048, 2048)
    assert p.cim.Gmax == d.Gmax and p.cim.core_width == d.core_width and p.cim.n_cores == d.n_cores


def test_different_params_change_compute():
    """(b) a DIFFERENT params set actually drives a different dev_lat (params are load-bearing)."""
    base = json.loads(_M1.read_text())              # full, complete params (constructor uses p["..."])
    modified = dict(base)
    modified["core_width"] = base.get("core_width", 512) // 2
    modified["G_eff_Gmax_gops"] = base["G_eff_Gmax_gops"] * 0.5
    metis = CimTileModel()
    swapped = CimTileModel(params=modified)
    assert swapped.dev_lat_us(1, 2048, 2048) != metis.dev_lat_us(1, 2048, 2048)


def test_platform_resolves_params_through_topology(monkeypatch):
    """Pins the platform.py resolver itself: a topology pointing (REPO-RELATIVE, the required contract)
    at a DIFFERENT params file makes Platform's CimTileModel compute differently. Without this, reverting
    platform.py to a bare CimTileModel() would go undetected (card->m1_cim.json IS the default). The alt
    params file must live INSIDE the repo (the resolver rejects absolute/escaping paths), so it is
    written to a repo-relative path and removed in finally."""
    base = json.loads(_M1.read_text())
    base["core_width"] = base.get("core_width", 512) // 2
    base["G_eff_Gmax_gops"] = base["G_eff_Gmax_gops"] * 0.5
    rel = "simulator/models/params/_pytest_swap_arch.json"   # repo-relative (the contract)
    abs_p = ROOT / rel
    abs_p.write_text(json.dumps(base))
    try:
        real = platmod.load_spec

        def fake(name):
            s = dict(real(name))
            if name == "cim_topo_card":
                s["cim_compute_params"] = rel
            return s

        monkeypatch.setattr(platmod, "load_spec", fake)
        p = Platform("llama-3.1-8b", topology="cim_topo_card", memory_spec="mem_lpddr4x")
        assert p.cim.dev_lat_us(1, 2048, 2048) != CimTileModel().dev_lat_us(1, 2048, 2048)
    finally:
        abs_p.unlink(missing_ok=True)


def test_platform_rejects_out_of_repo_params(monkeypatch, tmp_path):
    """The resolver must fail-loud on an absolute or repo-escaping cim_compute_params: a topology spec
    may only reference params committed INSIDE the repo (reproducibility/safety — a bare absolute path
    would read a machine-local file that isn't in the commit)."""
    real = platmod.load_spec

    def make(bad):
        def fake(name):
            s = dict(real(name))
            if name == "cim_topo_card":
                s["cim_compute_params"] = bad
            return s
        return fake

    for bad in [str(tmp_path / "x.json"), "/etc/passwd", "../../../../../../etc/passwd"]:
        monkeypatch.setattr(platmod, "load_spec", make(bad))
        try:
            Platform("llama-3.1-8b", topology="cim_topo_card", memory_spec="mem_lpddr4x")
            raise AssertionError(f"out-of-repo cim_compute_params {bad!r} was not rejected")
        except ValueError:
            pass


def test_nonmetis_params_tagged_simulated(monkeypatch):
    """(c) a topology whose cim_compute_params != m1_cim.json is tagged `simulated` and is NOT a
    calibrated anchor. Uses a VALID topology name (cim_topo_card) with load_spec monkeypatched so
    the field differs — a novel topology name would be rejected by validate() before provenance runs."""
    real = cfgmod.load_spec

    def fake(name):
        s = dict(real(name))
        if name == "cim_topo_card":
            s["cim_compute_params"] = "simulator/models/params/some_other_arch.json"
        return s

    monkeypatch.setattr(cfgmod, "load_spec", fake)
    c = _cfg("cim_topo_card", "mem_lpddr4x")
    assert any("CIM compute params" in x and x.startswith("simulated:") for x in c.provenance), c.provenance
    assert c.is_calibrated_anchor() is False


def test_wired_topologies_unchanged():
    """(d) card/alpha/edge (all point at m1_cim.json) keep their current anchor/provenance; none gets
    a CIM-compute-params flag."""
    for topo, mem, anchor in [("cim_topo_card", "mem_lpddr4x", True),
                              ("cim_topo_edge", "mem_lpddr5", False),
                              ("cim_topo_alpha", None, False)]:
        c = _cfg(topo, mem)
        assert c.is_calibrated_anchor() is anchor, (topo, c.provenance)
        assert not any("CIM compute params" in x for x in c.provenance), (topo, c.provenance)
