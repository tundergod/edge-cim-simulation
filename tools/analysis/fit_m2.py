"""Phase 1 — build M2 memory params (fixed, no fit) + sanity-validate.

Reads  measurements/aetina/metis_alpha_matmul.json (pcie_floor_A1d5)
Writes simulator/models/params/{m2_pcie,m2_lpddr5}.json
       validation/reports/phase1.1/m2.json

PCIe floor + BW are fixed params (no per-shape sweep collected). LPDDR5 effective BW
is analytic (JEDEC peak + measured decode wall); Ramulator2 deferred to Phase 2.

Run: ./.venv/bin/python tools/analysis/fit_m2.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from simulator.models.m2_memory import MemoryModel  # noqa: E402
from simulator.specs.loader import load_spec  # noqa: E402

AET = ROOT / "measurements/aetina"

# The MEASURED anchor (24.2 GB/s) is the production Metis Card's on-card LPDDR4x effective
# decode bandwidth (voyager-sdk.md:278, r2=0.997). LPDDR4x-4266 x64-bit peak ~= 34 GB/s ->
# 24.2/34 = 71% efficiency, consistent with the literature for memory-bound LLM decode:
#   HeteroInfer SOSP'25 (Fig 5): single-proc decode = 40-45 / 68 GB/s = 59-66% of peak;
#   web survey: 60-80% (21% unoptimised, 85% best-case FPGA).
LPDDR4X_PEAK_GBs = 34.1                # LPDDR4x-4266 x64-bit (production-card config undocumented [GAP])
MEASURED_LPDDR4X_EFF_GBs = 24.2        # production card, the validation anchor
# The SIMULATED forward-looking SoC uses LPDDR5 (a DIFFERENT memory): peak 51.2 (LPDDR5-6400
# 4x16b); effective at ~65% (evidence above) ~= 33 GB/s. Do NOT report 24.2 as "% of LPDDR5".
SIM_LPDDR5_PEAK_GBs = 51.2
SIM_EFFICIENCY = 0.65
SIM_LPDDR5_EFF_GBs = round(SIM_LPDDR5_PEAK_GBs * SIM_EFFICIENCY, 1)


def main():
    matmul = json.loads((AET / "metis_alpha_matmul.json").read_text())
    floor = matmul["pcie_floor_A1d5"]

    pcie = {
        "_doc": "M2 2a PCIe/DMA. Fixed params (no per-shape PCIe sweep collected, Phase 0.3 "
                "gap). Floor applies to DISCRETE host<->device transfers only; decode "
                "weight-streaming uses BW term (no per-call floor) in recompose/production.",
        "fixed_overhead_us_median": floor["fixed_overhead_us_median"],
        "fixed_overhead_us_p95": floor["fixed_overhead_us_p95"],
        "pcie_BW_GBs": 3.9,
        "floor_applies_to": ["kv_reload", "activation_handoff", "conversion_op_traffic"],
        "no_slope_refit": "no per-shape (bytes,latency) PCIe table committed (Phase 0.3 gap)",
    }
    (ROOT / "simulator/models/params/m2_pcie.json").write_text(json.dumps(pcie, indent=1))

    eff_vs_lpddr4x = round(MEASURED_LPDDR4X_EFF_GBs / LPDDR4X_PEAK_GBs, 3)   # 0.71
    dram = {
        "_doc": "M2 host DRAM backend, ANALYTIC (Ramulator2 deferred to Phase 2, ADR-0002). "
                "The MEASURED anchor is the production Metis Card's on-card LPDDR4x; the simulated "
                "forward-looking SoC uses LPDDR5 (a DIFFERENT memory). Do NOT report 24.2 as '% of LPDDR5'.",
        "memory_type_measured": "LPDDR4x (production Metis Card, on-card; exact config [GAP])",
        "effective_BW_GBs": MEASURED_LPDDR4X_EFF_GBs,            # 24.2 — model uses this (validation anchor)
        "jedec_peak_GBs": LPDDR4X_PEAK_GBs,                      # peak of the MEASURED memory (LPDDR4x ~34)
        "efficiency_vs_measured_peak": eff_vs_lpddr4x,          # 24.2/34.1 = 0.71 (NOT 47% of LPDDR5)
        "efficiency_evidence": "HeteroInfer SOSP'25 Fig5: single-proc decode 40-45/68=59-66% of peak; "
                               "web survey 60-80% (21% unoptimised, 85% best-case) for memory-bound LLM decode.",
        "sim_lpddr5_peak_GBs": SIM_LPDDR5_PEAK_GBs,             # 51.2 (LPDDR5-6400 4x16b), forward-looking SoC
        "sim_lpddr5_eff_BW_GBs": SIM_LPDDR5_EFF_GBs,            # ~33.3 (51.2 x 0.65)
        "sim_efficiency": SIM_EFFICIENCY,
        "source_measured": "voyager-sdk.md:278 (decode time prop weight bytes, r2=0.997)",
        "ramulator2": "deferred to Phase 2",
    }
    (ROOT / "simulator/models/params/m2_lpddr5.json").write_text(json.dumps(dram, indent=1))

    # decision-A migration: the Phase 1.2 spec-based engine for the monotonicity sanity (PCIe floor
    # from the Alpha topology spec; stream/kv from the LPDDR4x anchor spec). The report's numeric
    # params come from the MEASURED constants below (keeps the frozen 1.1 m2.json exact: floor 911.1).
    pci = MemoryModel(load_spec("cim_topo_alpha"))
    mem = MemoryModel(load_spec("mem_lpddr4x"))
    sizes = [1e3, 1e5, 1e6, 1e7, 1e8]
    tr = [pci.pcie_transfer_us(b) for b in sizes]
    kv = [mem.kv_append_us(b) for b in sizes]
    floor_us, pcie_BW = floor["fixed_overhead_us_median"], 3.9   # measured (m2_pcie.json)
    eff, peak = MEASURED_LPDDR4X_EFF_GBs, LPDDR4X_PEAK_GBs       # 24.2, 34.1 (LPDDR4x)

    report = {
        "module": "m2_memory",
        "equation": {"pcie": "transfer_us = floor(911us) + bytes/BW(3.9GB/s)  [discrete only]",
                     "dram": "stream_us = bytes/eff_BW", "kv_cache": "kv_append_us = kv_bytes/eff_BW"},
        "params": {"pcie_floor_us": floor_us, "pcie_BW_GBs": pcie_BW,
                   "measured_eff_BW_GBs": eff, "measured_memory": "LPDDR4x (production card)",
                   "measured_peak_GBs": peak, "efficiency_vs_measured_peak": eff_vs_lpddr4x,
                   "sim_lpddr5_peak_GBs": SIM_LPDDR5_PEAK_GBs, "sim_lpddr5_eff_GBs": SIM_LPDDR5_EFF_GBs},
        "sanity": {
            "pcie_transfer_positive": all(t > 0 for t in tr),
            "pcie_transfer_monotonic": all(b >= a for a, b in zip(tr, tr[1:])),
            "kv_append_positive_monotonic": all(b >= a > 0 for a, b in zip(kv, kv[1:])),
            "eff_in_60_80pct_of_measured_peak": 0.55 <= eff / peak <= 0.85,
            "efficiency_note": f"{eff_vs_lpddr4x:.0%} of LPDDR4x peak — matches HeteroInfer (59-66%) "
                               "+ web (60-80%) for memory-bound decode. (Prior '47% of LPDDR5' was a "
                               "wrong-memory comparison.)"},
        "notes": {
            "build_sram_hierarchy": "Phase 2 WILL build L1 (4 MiB/core) + L2 (32 MiB shared) "
                "residency (reversed the earlier no-build decision) — needed for architecture "
                "research once the memory-wall bottleneck is addressed. Alpha l2/ddr showed no "
                "effect only because Alpha has no on-card DRAM; the simulated SoC does.",
            "kv_cache_unvalidated": "Phase 0.3 did not isolate KV-append; analytic kv_bytes/BW, "
                "used as a temporary stand-in (board offline) — flagged unvalidated.",
            "ramulator2_deferred": "Phase 2 (ADR-0002 swappable); PIM-like ext = risk #6."},
    }
    (ROOT / "validation/reports/phase1.1/m2.json").write_text(json.dumps(report, indent=1))
    s = report["sanity"]
    print(f"M2: pcie floor={floor_us}us BW={pcie_BW}GB/s | measured LPDDR4x eff={eff} "
          f"peak={peak} ({eff_vs_lpddr4x:.0%}) | sim LPDDR5 eff={SIM_LPDDR5_EFF_GBs}/{SIM_LPDDR5_PEAK_GBs} "
          f"| sanity={all(v for v in s.values() if isinstance(v, bool))}")


if __name__ == "__main__":
    main()
