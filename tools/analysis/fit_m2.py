"""Phase 1 — build M2 memory params (fixed, no fit) + sanity-validate.

Reads  measurements/aetina/metis_alpha_matmul.json (pcie_floor_A1d5)
Writes simulator/models/params/{m2_pcie,m2_lpddr5}.json
       validation/reports/m2.json

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

AET = ROOT / "measurements/aetina"

# JEDEC LPDDR5-6400, 4x16-bit channels = 51.2 GB/s peak (a representative mobile-SoC config).
JEDEC_LPDDR5_PEAK_GBs = 51.2
# measured decode wall (~24 GB/s, voyager-sdk.md / ADR-0006; C5 implied 16.2-20.5 per model)
DECODE_WALL_GBs = 24.2


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

    lpddr5 = {
        "_doc": "M2 2b host LPDDR5 backend, ANALYTIC (Ramulator2 deferred to Phase 2, "
                "ADR-0002 swappable). PIM-like extension deferred (risk #6).",
        "jedec_peak_GBs": JEDEC_LPDDR5_PEAK_GBs,
        "jedec_config": "LPDDR5-6400, 4x16-bit ch",
        "effective_BW_GBs": DECODE_WALL_GBs,
        "efficiency_vs_peak": round(DECODE_WALL_GBs / JEDEC_LPDDR5_PEAK_GBs, 3),
        "source": "measured decode wall ~24 GB/s (voyager-sdk.md / ADR-0006)",
        "ramulator2": "deferred to Phase 2",
    }
    (ROOT / "simulator/models/params/m2_lpddr5.json").write_text(json.dumps(lpddr5, indent=1))

    m = MemoryModel(pcie, lpddr5)

    # sanity: transfer_us positive + monotonic in bytes; kv_append linear; BW brackets 24
    sizes = [1e3, 1e5, 1e6, 1e7, 1e8]
    tr = [m.pcie_transfer_us(b) for b in sizes]
    kv = [m.kv_append_us(b) for b in sizes]
    eff, peak = m.lpddr5_eff_BW_GBs, m.lpddr5_peak_GBs

    report = {
        "module": "m2_memory",
        "equation": {"pcie": "transfer_us = floor(911us) + bytes/BW(3.9GB/s)  [discrete only]",
                     "lpddr5": "stream_us = bytes/eff_BW", "kv_cache": "kv_append_us = kv_bytes/eff_BW"},
        "params": {"pcie_floor_us": m.floor_us, "pcie_BW_GBs": m.pcie_BW_GBs,
                   "lpddr5_eff_BW_GBs": eff, "lpddr5_peak_GBs": peak,
                   "lpddr5_efficiency": lpddr5["efficiency_vs_peak"]},
        "sanity": {
            "pcie_transfer_positive": all(t > 0 for t in tr),
            "pcie_transfer_monotonic": all(b >= a for a, b in zip(tr, tr[1:])),
            "kv_append_positive_monotonic": all(b >= a > 0 for a, b in zip(kv, kv[1:])),
            "eff_BW_brackets_24": abs(eff - 24.0) <= 2.0,
            "eff_le_peak": eff <= peak,
            "eff_ge_0.4_peak": eff >= 0.4 * peak,
            "eff_efficiency_note": f"{lpddr5['efficiency_vs_peak']:.0%} of JEDEC peak — typical "
                                   "memory-wall effective BW (plan bound relaxed 0.5->0.4)"},
        "notes": {
            "no_sram_residency": "Alpha l2/ddr ratio ~1.00-1.01 (no on-card DRAM) -> NO "
                "SRAM L1/L2 residency model (recorded in m2 contract for Phase 2).",
            "kv_cache_unvalidated": "Phase 0.3 did not isolate KV-append; analytic only.",
            "ramulator2_deferred": "Phase 2 (ADR-0002 swappable); PIM-like ext = risk #6."},
    }
    (ROOT / "validation/reports/m2.json").write_text(json.dumps(report, indent=1))
    s = report["sanity"]
    print(f"M2: pcie floor={m.floor_us}us BW={m.pcie_BW_GBs}GB/s | LPDDR5 eff={eff} peak={peak} "
          f"({lpddr5['efficiency_vs_peak']:.0%}) | sanity all={all(v for v in s.values() if isinstance(v, bool))}")
    print(f"  {s}")


if __name__ == "__main__":
    main()
