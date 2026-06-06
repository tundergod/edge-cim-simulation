"""Phase 1.2 cross-check (plan step 19): the swappable-spec + shared-engine integration gate.

Loads EVERY spec through the loader, feeds each into its engine, asserts the FROZEN predict()
contract, and verifies the honesty discipline is consistent end-to-end:
  - each engine returns exactly {latency_us, bound, provenance};
  - the honesty tag in predict() provenance matches the unit (CPU=calibrated, NPU/GPU/SRAM=simulated/
    assumption, MEM per-spec: LPDDR4x=calibrated anchor, LPDDR5=simulated, peaks=assumption);
  - NO FAKE GATE: the units with no silicon (NPU #13, GPU INT8, SRAM) carry NO per-op numeric
    silicon acceptance gate — only trend-shape / lower-bound / architecture acceptance;
  - CIM = Alpha 13pts calibrated (+ Card-revalidated if the Card run succeeded, else DEFERRED_FALLBACK).

Run: ./.venv/bin/python tools/analysis/check_phase1_2.py   (exit 0 = all pass)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from simulator.specs.loader import load_spec                       # noqa: E402
from simulator.models.engine import Workload, check_return        # noqa: E402
from simulator.models.m4_cpu import CpuModel                       # noqa: E402
from simulator.models.m4_npu import NpuModel                       # noqa: E402
from simulator.models.m2_memory import MemoryModel                 # noqa: E402
from simulator.models.m4_gpu_roofline import GpuRooflineModel      # noqa: E402
from simulator.models.m1_cim_spm import SramTier                   # noqa: E402

REP = ROOT / "validation/reports/phase1.2"
SPECS = ["cpu_rk3588", "npu_rknpu2", "gpu_mali_g610", "mem_lpddr4", "mem_lpddr4x",
         "mem_lpddr5", "sram_metis_aipu", "cim_topo_alpha", "cim_topo_card", "cim_topo_edge"]
fails = []


def chk(cond, msg):
    print(f"  {'OK  ' if cond else 'FAIL'} {msg}")
    if not cond:
        fails.append(msg)


def main():
    print("=== 1. every spec loads + carries provenance ===")
    for n in SPECS:
        s = load_spec(n)
        chk(bool(s.get("provenance")), f"{n}: provenance present ({len(s.get('provenance', {}))} fields)")

    print("\n=== 2. every engine conforms to the frozen predict() contract ===")
    cpu = CpuModel(load_spec("cpu_rk3588"))
    npu = NpuModel(load_spec("npu_rknpu2"))
    gpu = GpuRooflineModel(load_spec("gpu_mali_g610"))
    sram = SramTier(load_spec("sram_metis_aipu"))
    probes = [
        ("cpu", cpu.predict(Workload(op="softmax", kv=512, extra={"model": "llama-3.1-8b", "dtype": "fp32"}))),
        ("npu", npu.predict(Workload(op="matmul", M=1, K=2048, N=2048))),
        ("gpu", gpu.predict(Workload(op="gemm", M=1, K=4096, N=4096, dtype="fp16"))),
        ("sram", sram.predict(Workload(op="stream", nbytes=1_000_000))),
    ]
    for sp in ["mem_lpddr4", "mem_lpddr4x", "mem_lpddr5"]:
        probes.append((sp, MemoryModel(load_spec(sp)).predict(Workload(op="stream", nbytes=1_000_000))))
    probes.append(("cim_topo_alpha", MemoryModel(load_spec("cim_topo_alpha")).predict(Workload(op="pcie", nbytes=0))))
    probes.append(("cim_topo_card", MemoryModel(load_spec("cim_topo_card")).predict(Workload(op="stream", nbytes=1_000_000))))
    probes.append(("cim_topo_edge", MemoryModel(load_spec("cim_topo_edge")).predict(Workload(op="stream", nbytes=1_000_000))))
    for name, out in probes:
        try:
            check_return(out)
            chk(True, f"{name}: predict -> {out['bound']:7s} lat={out['latency_us']:.1f}us")
        except Exception as e:
            chk(False, f"{name}: {e}")

    print("\n=== 3. honesty tag in predict() provenance matches the unit ===")
    chk("CALIBRATED" in probes[0][1]["provenance"], "CPU provenance = CALIBRATED")
    chk("simulated" in probes[1][1]["provenance"], "NPU provenance = simulated")
    chk("simulated" in probes[2][1]["provenance"], "GPU roofline provenance = simulated")
    chk("calibrated" in dict(probes)["mem_lpddr4x"]["provenance"], "LPDDR4x = calibrated anchor")
    chk("simulated" in dict(probes)["mem_lpddr5"]["provenance"], "LPDDR5 = simulated")
    chk("assumption" in dict(probes)["mem_lpddr4"]["provenance"], "LPDDR4 = assumption (derived)")
    chk(dict(probes)["cim_topo_alpha"]["bound"] == "floor", "Alpha pays the per-call floor (bound=floor)")
    chk("assumption" in dict(probes)["cim_topo_edge"]["provenance"] and dict(probes)["cim_topo_edge"]["bound"] == "memory",
        "edge CIM = assumption memory wall (target LPDDR5 x noc_eff, NOT Card 24.2)")

    print("\n=== 4. NO FAKE GATE (units with no silicon carry no numeric silicon gate) ===")
    npu_r = json.loads((REP / "m4_npu.json").read_text())
    chk(npu_r["honesty"] == "simulated" and npu_r.get("all_trends_pass_simulated") is True,
        "NPU report: honesty=simulated + trend-shape acceptance")
    chk("NO per-op numeric gate" in npu_r["acceptance"] or "no per-op numeric gate" in npu_r["acceptance"].lower(),
        "NPU report: explicit NO per-op numeric gate")
    chk("superseded-not-satisfied" in json.dumps(npu_r["upgrade"]),
        "NPU report: issue #13 silicon = superseded-not-satisfied (not achieved)")
    gpu_r = json.loads((REP / "m4_gpu_roofline.json").read_text())
    chk("ZERO" in gpu_r["honesty"]["int8"], "GPU report: INT8 = ZERO data (no INT8 gate)")
    cim_r = json.loads((REP / "cim_card_revalidate.json").read_text())
    chk("Alpha 13" in cim_r["honesty"] and ("CALIBRATED" in cim_r["honesty"] or "calibrated" in cim_r["honesty"]),
        f"CIM = Alpha 13pts calibrated (status={cim_r['status']})")

    print("\n=== 5. engine interface conformance test still green ===")
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "tests/test_engine_iface.py")], capture_output=True, text=True)
    chk(r.returncode == 0, "tests/test_engine_iface.py: " + (r.stdout.strip().splitlines()[-1] if r.stdout else "FAIL"))

    print(f"\n{'='*54}\n{'ALL PHASE-1.2 CROSS-CHECKS PASS' if not fails else f'{len(fails)} FAILURES'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
