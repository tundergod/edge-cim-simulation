"""Phase 1.3 — Ramulator2-vs-analytic LPDDR5 single-stream delta report -> validation/reports/phase1.3/m2_ramulator2.json.

Compares the Ramulator2 v2.1 LPDDR5_6400 streaming result (simulator/engines/ramulator2/lpddr5_eff.json)
against the Phase-1.2 analytic LPDDR5 (mem_lpddr5 spec, eff 0.65). The key finding is a DEVICE-vs-
SYSTEM distinction, not a contradiction: Ramulator2 models the DRAM DEVICE timing (refresh, bank
conflicts) and a single stream reaches ~0.92 of peak; the analytic 0.65 is the SYSTEM-level
efficiency calibrated to the silicon decode wall (24.2 GB/s on LPDDR4x), which folds in controller/
NoC/queueing overheads Ramulator2's device model omits. So Ramulator2 confirms the DRAM device is
NOT the single-stream bottleneck, and VALIDATES ADR-0002's decision to calibrate the system
efficiency from silicon rather than import it from Ramulator2. Analytic 33.3 stays primary; the
Ramulator2 device BW (47.1) is the device-level ceiling. Multi-unit contention = Phase 2.

Run: ./.venv/bin/python tools/analysis/build_mem_ramulator2.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from simulator.specs.loader import load_spec  # noqa: E402

RAM2 = ROOT / "simulator/engines/ramulator2/lpddr5_eff.json"
OUT = ROOT / "validation/reports/phase1.3/m2_ramulator2.json"


def main():
    r = json.loads(RAM2.read_text())
    mem5 = load_spec("mem_lpddr5")
    analytic_eff = mem5["efficiency_sim"]          # 0.65
    analytic_bw = mem5["eff_BW_GBs"]               # 33.3
    peak = mem5["peak_GBs"]                        # 51.2
    ram_eff = r["efficiency"]                      # 0.92 (device, achievable)
    ram_bw = r["eff_BW_GBs"]                       # 47.1

    # per-shape latency delta (a constant ratio, since streaming BW is shape-independent)
    shapes = [("8B-weights/token (decode)", 8_000_000), ("1MB", 1_000_000), ("64KB", 64_000)]
    per_shape = [{"label": lbl, "bytes": b,
                  "analytic_us": round(b / (analytic_bw * 1e9) * 1e6, 2),
                  "ramulator2_us": round(b / (ram_bw * 1e9) * 1e6, 2),
                  "delta_pct": round((analytic_bw - ram_bw) / ram_bw * 100, 1)} for lbl, b in shapes]

    report = {
        "module": "m2_ramulator2",
        "phase": "1.3",
        "engine": "MemoryModel(mem_lpddr5, engine='ramulator2')",
        "source": "Ramulator2 v2.1 (commit %s), LPDDR5_6400 saturated streaming" % r["v2_1_commit"][:10],
        "saturation_peak_efficiency": r["peak_efficiency_saturation"],   # 0.986 -> stream is saturated
        "ramulator2_device": {"efficiency": ram_eff, "eff_BW_GBs": ram_bw,
                              "peak_GBs_single_channel": r["peak_GBs_single_channel"],
                              "scope": "DRAM device timing only (refresh + bank conflicts); refresh=AllBank"},
        "analytic_system": {"efficiency": analytic_eff, "eff_BW_GBs": analytic_bw,
                            "scope": "system-level, calibrated to the silicon decode wall (24.2 GB/s LPDDR4x)"},
        "delta": {"ramulator2_eff_minus_analytic_eff": round(ram_eff - analytic_eff, 3),
                  "ramulator2_BW_over_analytic_BW": round(ram_bw / analytic_bw, 2)},
        "per_shape_latency_delta": per_shape,
        "finding": "Ramulator2 (device-only) single-stream efficiency 0.92 >> analytic system efficiency "
                   "0.65 — NOT a contradiction: the gap is the system overhead (controller/NoC/queueing/"
                   "real-workload) that the analytic 0.65 captures (silicon-calibrated) and Ramulator2's "
                   "device model omits. Confirms the DRAM device is not the single-stream bottleneck, and "
                   "validates ADR-0002 (system efficiency calibrated from silicon, NOT imported from Ramulator2).",
        "honesty": "simulated (Ramulator2 v2.1 device model), NOT silicon; single-stream only. The analytic "
                   "system-level 33.3 stays PRIMARY. Ramulator2's signature value (multi-unit contention) is Phase 2.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(f"m2_ramulator2: Ramulator2 device eff {ram_eff:.2f} ({ram_bw} GB/s) vs analytic system "
          f"eff {analytic_eff} ({analytic_bw} GB/s); peak saturated {r['peak_efficiency_saturation']:.1%} "
          f"-> {OUT}")


if __name__ == "__main__":
    main()
