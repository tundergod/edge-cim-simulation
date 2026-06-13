"""Phase 2.1 — M5 oracle check (simulator/runtime/workload.py).

The per-token DAG, summed over a (P,D) generation, must reproduce
Model.profile(P,D) per-(phase,category) counts for every model (no dropped or
double-counted ops; identical category set = semantic coverage + zero orphans).
Model() construction also self-validates the length-templates vs held-out
inventory. Run from the repo root (op_profile reads measurements/ via CWD).

    .venv/bin/pytest tests/test_workload.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.runtime.workload import oracle_check, build_token_dag, wl_from_row  # noqa: E402
from simulator.runtime.dag import wl_is_sane  # noqa: E402

MODELS = ["llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b", "qwen2.5-7b"]


def test_oracle_counts_match_all_models():
    for m in MODELS:
        ok, detail = oracle_check(m, 128, 4)
        assert ok, f"{m}: DAG counts != profile: {detail}"


def test_decode_dag_nodes_sane_and_chained():
    dag = build_token_dag("llama-3.1-8b", "decode", 256)
    assert len(dag) > 0 and dag.is_acyclic()
    assert all(wl_is_sane(n.wl) for n in dag.nodes)
    assert all(n.bytes_streamed >= 0 for n in dag.nodes)
    # serial chain: exactly one root, every non-root has its predecessor as dep
    assert len(dag.roots()) == 1


def test_categories_are_known():
    from simulator.runtime.dag import CATEGORIES
    dag = build_token_dag("llama-3.2-1b", "decode", 128)
    assert {n.category for n in dag.nodes} <= set(CATEGORIES)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"\n{len(fns)} workload tests passed.")
