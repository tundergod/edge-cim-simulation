"""Phase 1.3 cross-check: the heavy engines are drop-in for the Phase 1.2 analytic engine.

Verifies that `engine='analytic' | 'ramulator2' | 'onnxim'` are interchangeable behind the SAME
constructor + frozen predict() contract (the Phase 1.3 deliverable: interface-ready). When the
C++ heavy sims are NOT built (this session: external builds were not authorized), the heavy
engines fall back to the analytic result with an honest provenance note — and this check confirms
(a) the contract holds, (b) the fallback is faithful (same number), (c) the honesty tag says so.

When the heavy sims ARE built (simulator/engines/ramulator2/lpddr5_eff.json, simulator/engines/onnxim/
rknpu2_sim_matmul.json present), the same check confirms the heavy path is used and still conforms.

Run: ./.venv/bin/python tools/analysis/check_phase1_3.py   (exit 0 = pass)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from simulator.specs.loader import load_spec                 # noqa: E402
from simulator.models.engine import Workload, check_return  # noqa: E402
from simulator.models.m2_memory import MemoryModel           # noqa: E402
from simulator.models.m4_npu import NpuModel                 # noqa: E402

RAM2 = ROOT / "simulator/engines/ramulator2/lpddr5_eff.json"
ONNX = ROOT / "simulator/engines/onnxim/rknpu2_sim_matmul.json"
fails = []


def chk(cond, msg):
    print(f"  {'OK  ' if cond else 'FAIL'} {msg}")
    if not cond:
        fails.append(msg)


def main():
    print("=== preflight: the 1.2 spec/engine=/predict API is live ===")
    base = MemoryModel(load_spec("mem_lpddr5"), engine="analytic").predict(Workload(op="stream", nbytes=1_000_000))
    chk(set(base) == {"latency_us", "bound", "provenance"}, "MemoryModel(spec, engine='analytic').predict() frozen keys")

    print("\n=== Ramulator2 drop-in (memory heavy) ===")
    wl = Workload(op="stream", nbytes=1_000_000)
    a = MemoryModel(load_spec("mem_lpddr5"), engine="analytic").predict(wl)
    r = MemoryModel(load_spec("mem_lpddr5"), engine="ramulator2").predict(wl)
    chk(set(r) == {"latency_us", "bound", "provenance"}, "engine='ramulator2' returns the frozen contract")
    if RAM2.exists():
        chk("Ramulator2 LPDDR5 heavy sim" in r["provenance"], "Ramulator2 build present -> heavy path used")
        chk("NOT silicon" in r["provenance"], "Ramulator2 honestly tagged simulated, NOT silicon")
        # numeric (not string-only): ramulator device-eff (~0.92) > analytic system-eff (0.65) -> FASTER,
        # but within a sane factor (~0.71x). A wrong ramulator number outside (0.5x, 1.0x) of analytic fails.
        chk(0.5 * a["latency_us"] < r["latency_us"] < a["latency_us"],
            f"Ramulator2 latency {r['latency_us']:.1f}us in (0.5x,1.0x) of analytic {a['latency_us']:.1f}us")
    else:
        chk(abs(a["latency_us"] - r["latency_us"]) < 1e-9, "build deferred -> faithful analytic fallback (same latency)")
        chk("ANALYTIC fallback" in r["provenance"] and "ramulator2" in r["provenance"], "fallback honestly noted in provenance")

    print("\n=== ONNXim drop-in (NPU heavy) ===")
    wl = Workload(op="matmul", M=1, K=2048, N=2048)
    a = NpuModel(load_spec("npu_rknpu2"), engine="analytic").predict(wl)
    o = NpuModel(load_spec("npu_rknpu2"), engine="onnxim").predict(wl)
    chk(set(o) == {"latency_us", "bound", "provenance"}, "engine='onnxim' returns the frozen contract")
    if ONNX.exists():
        chk("ONNXim" in o["provenance"] and "NOT silicon" in o["provenance"], "ONNXim build present -> heavy path, tagged simulated NOT silicon")
    else:
        chk(abs(a["latency_us"] - o["latency_us"]) < 1e-9, "build deferred -> faithful analytic fallback (same latency)")
        chk("ANALYTIC fallback" in o["provenance"] and "onnxim" in o["provenance"], "fallback honestly noted in provenance")

    print("\n=== honesty: heavy engines are simulated, never silicon-validated ===")
    chk("silicon" not in r["provenance"].lower() or "not silicon" in r["provenance"].lower(),
        "Ramulator2 never claims silicon")
    chk("simulated" in o["provenance"].lower(), "ONNXim path is simulated")

    print(f"\n{'='*52}\n{'ALL PHASE-1.3 DROP-IN CHECKS PASS' if not fails else f'{len(fails)} FAILURES'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
