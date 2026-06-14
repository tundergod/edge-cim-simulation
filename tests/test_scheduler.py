"""Phase 2.2b Step A — Scheduler ABC (simulator/runtime/scheduler.py).

`Scheduler(ABC).assign(dag, cfg) -> dag` is a pure, idempotent annotator (sets only
node.unit + node.mem_domain). `AllCimScheduler` is the 2.1/2.2a all-CIM placement
behind the ABC; the thin `all_cim_assign` wrapper is kept for existing callers. The
hard regression (AllCim L4 byte-identical) is checked in validate_e2e_l4, not here.

    .venv/bin/pytest tests/test_scheduler.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "trace_export"))
from simulator.runtime.workload import build_token_dag  # noqa: E402
from simulator.runtime.scheduler import (  # noqa: E402
    Scheduler, AllCimScheduler, CimHeteroScheduler, SCHEDULERS, all_cim_assign,
)


def test_allcim_scheduler_assigns_units_and_domains():
    dag = build_token_dag("llama-3.2-1b", "decode", 512)
    out = AllCimScheduler().assign(dag)
    assert out is dag                                  # annotates in place, returns the dag
    for n in dag.nodes:
        assert n.unit in ("cim", "cpu", "mem")
        assert n.mem_domain in ("dram", "cpu_cache", "none")
    cats = {n.category: n.unit for n in dag.nodes}
    assert cats.get("matmul") == "cim" and cats.get("attention") == "cim"
    assert cats.get("softmax") == "cpu" and cats.get("kv_cache") == "mem"


def test_assign_is_pure_and_idempotent():
    # only unit + mem_domain change; re-assign gives an identical result; other fields untouched.
    dag = build_token_dag("llama-3.2-1b", "decode", 512)
    cats0 = [n.category for n in dag.nodes]
    deps0 = [list(n.deps) for n in dag.nodes]
    AllCimScheduler().assign(dag)
    snap = [(n.unit, n.mem_domain) for n in dag.nodes]
    AllCimScheduler().assign(dag)
    assert [(n.unit, n.mem_domain) for n in dag.nodes] == snap     # idempotent
    assert [n.category for n in dag.nodes] == cats0                # category untouched
    assert [list(n.deps) for n in dag.nodes] == deps0             # deps untouched


def test_allcim_scheduler_matches_legacy_wrapper():
    d1 = all_cim_assign(build_token_dag("llama-3.2-1b", "decode", 512))
    d2 = AllCimScheduler().assign(build_token_dag("llama-3.2-1b", "decode", 512))
    assert [(n.unit, n.mem_domain) for n in d1.nodes] == [(n.unit, n.mem_domain) for n in d2.nodes]


def test_scheduler_registry():
    assert "all_cim" in SCHEDULERS
    assert isinstance(SCHEDULERS["all_cim"], Scheduler)
    assert isinstance(SCHEDULERS["all_cim"], AllCimScheduler)


def test_cimhetero_scheduler_placement_and_conversions():
    dag = CimHeteroScheduler().assign(build_token_dag("llama-3.2-1b", "decode", 512))
    by_cat = {}
    for n in dag.nodes:
        by_cat.setdefault(n.category, set()).add(n.unit)
    assert by_cat["matmul"] == {"cim"}
    assert by_cat["softmax"] == {"cpu"} and by_cat["kv_cache"] == {"mem"}
    bmm_units = {n.unit for n in dag.nodes if n.category == "attention" and "hd" in n.wl.extra}
    assert bmm_units == {"gpu"}                                   # QK^T/S·V bmm on GPU
    assert any(n.category == "convert" for n in dag.nodes)         # conversions inserted
    assert all(n.mem_domain in ("dram", "cpu_cache", "none") for n in dag.nodes)  # incl converts


def test_cimhetero_registry_and_pipeline_mode():
    assert "cim_hetero" in SCHEDULERS and isinstance(SCHEDULERS["cim_hetero"], CimHeteroScheduler)
    assert SCHEDULERS["cim_hetero"].pipeline is True              # genuine multi-unit overlap
    assert SCHEDULERS["all_cim"].pipeline is False               # single-accelerator serial


def test_cimhetero_runs_end_to_end_simulated():
    from simulator.runtime.config import SimConfig
    from simulator.runtime.runner import run
    r = run(SimConfig.from_dict({"workload": {"model": "llama-3.2-1b", "context": 1024},
                                 "scheduler": {"policy": "cim_hetero"}}))
    assert r["tok_s"] > 0
    assert r["calibrated_anchor"] is False                        # SIMULATED (no concurrent silicon)
    assert any("simulated" in p for p in r["provenance"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"\n{len(fns)} scheduler tests passed.")
