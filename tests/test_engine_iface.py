"""Conformance test for the shared UnitEngine interface (Phase 1.2 step 2b).

A dummy engine must (a) bind its spec at construction, take `engine=` and (b) return
EXACTLY the frozen keys from predict(). Run this BEFORE the WP fan-out — every WP engine
fills in the same signature, so a green run here is the contract the parallel agents build
against. No pytest in this venv -> plain asserts + a __main__ runner (also pytest-friendly).

    .venv/bin/python tests/test_engine_iface.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.models.engine import UnitEngine, Workload, check_return, RETURN_KEYS  # noqa: E402
from simulator.specs.loader import load_spec, provenance  # noqa: E402


class _DummyEngine(UnitEngine):
    """Minimal conformant engine: latency proportional to K, always compute-bound."""

    def predict(self, wl):
        return {"latency_us": 1.0 * wl.K, "bound": "compute", "provenance": "dummy"}


def test_binds_spec_and_engine_at_construction():
    e = _DummyEngine(spec={"x": 1}, engine="analytic")
    assert e.spec == {"x": 1}
    assert e.engine == "analytic"
    # default engine is analytic (Phase 1.2 backend)
    assert _DummyEngine(spec={"x": 1}).engine == "analytic"


def test_predict_returns_exactly_frozen_keys():
    e = _DummyEngine(spec={})
    out = e.predict(Workload(op="matmul", M=1, K=2048, N=2048))
    assert set(out) == set(RETURN_KEYS), f"{sorted(out)} != {sorted(RETURN_KEYS)}"
    check_return(out)
    assert out["latency_us"] == 2048.0


def test_check_return_rejects_bad_contracts():
    for bad in (
        {"latency_us": 1.0, "bound": "compute"},                      # missing key
        {"latency_us": 1.0, "bound": "nonsense", "provenance": "x"},  # bad bound
        {"latency_us": -1.0, "bound": "compute", "provenance": "x"},  # negative latency
        {"latency_us": 1.0, "bound": "compute", "provenance": ""},    # empty provenance
    ):
        try:
            check_return(bad)
        except AssertionError:
            continue
        raise AssertionError(f"check_return should have rejected {bad}")


def test_loader_roundtrips_when_specs_exist():
    # Loader contract: a known-missing spec raises; provenance() reads the map. Spec files
    # themselves are written in the next step, so only the missing-file path is asserted here.
    try:
        load_spec("__definitely_missing__")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("load_spec should raise on a missing spec")
    assert provenance({"provenance": {"peak_GBs": "assumption"}}, "peak_GBs") == "assumption"


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nOK — {len(fns)} conformance tests passed")


if __name__ == "__main__":
    _run()
