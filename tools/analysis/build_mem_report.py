"""Phase 1.2 — M2 memory report: 3 mem specs vs the 24.2 GB/s LPDDR4x anchor + SRAM what-if.

Loads mem_lpddr4/4x/5 (+ sram_metis_aipu) via the spec loader, drives the analytic
MemoryModel / SramTier engines, and writes validation/reports/phase1.2/m2_memory.json with
per-spec eff_BW + honesty tags. NO new fit: peaks are ASSUMPTION; LPDDR4x 24.2 is the
MEASURED anchor (calibrated); LPDDR5 33.3 is SIMULATED (sim eff 0.65 vs measured 0.71 —
one-line explanation below); SRAM BW/latency are CACTI ASSUMPTION; residency is
architecture-only (8B weights never resident -> spill to DRAM).

Run: ./.venv/bin/python tools/analysis/build_mem_report.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from simulator.specs.loader import load_spec                  # noqa: E402
from simulator.models.engine import Workload                  # noqa: E402
from simulator.models.m2_memory import MemoryModel            # noqa: E402
from simulator.models.m1_cim_spm import SramTier              # noqa: E402

OUT = ROOT / "validation/reports/phase1.2/m2_memory.json"
ANCHOR_EFF_BW = 24.2     # production-card LPDDR4x decode anchor (mem_lpddr4x)
MEM_SPECS = ["mem_lpddr4", "mem_lpddr4x", "mem_lpddr5"]
# honesty tag per spec's eff_BW (calibrated = fit to OUR silicon)
TAG = {"mem_lpddr4": "assumption (eff derived from measured-4x 0.71 efficiency)",
       "mem_lpddr4x": "calibrated (measured production-card decode anchor 24.2 GB/s)",
       "mem_lpddr5": "simulated (sim eff 0.65 < measured 0.71; different memory, discounted)"}


def main():
    # --- 3 memory specs vs the 24.2 anchor -------------------------------
    sizes = [1e3, 1e5, 1e7, 1e9]
    mem_rows = {}
    for name in MEM_SPECS:
        sp = load_spec(name)
        m = MemoryModel(sp)
        lats = [m.predict(Workload(op="stream", nbytes=int(b)))["latency_us"] for b in sizes]
        mem_rows[name] = {
            "memory_type": sp["memory_type"],
            "peak_GBs": sp.get("peak_GBs"),
            "eff_BW_GBs": m.eff_BW_GBs,
            "eff_BW_tag": TAG[name],
            "peak_tag": "assumption (in-repo no data-rate source)",
            "vs_anchor_ratio": round(m.eff_BW_GBs / ANCHOR_EFF_BW, 3),
            "stream_us": {f"{int(b):.0e}B": round(x, 3) for b, x in zip(sizes, lats)},
            "positive": all(x > 0 for x in lats),
            "monotonic_in_bytes": all(b > a for a, b in zip(lats, lats[1:])),
        }

    # --- topology floors (alpha pays 911, card pays 0) -------------------
    alpha = MemoryModel(load_spec("cim_topo_alpha"))
    card = MemoryModel(load_spec("cim_topo_card"))
    topo = {
        "alpha": {"per_call_floor_us": alpha.floor_us, "pcie_BW_GBs": alpha.pcie_BW_GBs,
                  "pcie_transfer_us(0B)": alpha.predict(Workload(op="pcie", nbytes=0))["latency_us"],
                  "tag": "floor measured (911 us per-call DMA, Alpha topology artifact)"},
        "card": {"per_call_floor_us": card.floor_us, "on_card_eff_BW_GBs": card.eff_BW_GBs,
                 "tag": "no per-call floor (on-card streaming); BW = measured 24.2 anchor"},
    }

    # --- SRAM residency what-if (8B weights never resident) --------------
    ssp = load_spec("sram_metis_aipu")
    sram = SramTier(ssp)
    w8b = 8_000_000_000          # ~8 GB INT8 weights
    act = 4 * 1024 * 1024        # 4 MiB activation tile (fits L2)
    sram_whatif = {
        "l2_MiB_shared": sram.l2_MiB,
        "bw_GBs": sram.bw_GBs, "latency_ns": sram.latency_ns,
        "bw_latency_tag": "CACTI assumption (no published Metis SRAM BW/latency)",
        "residency": sram.residency,
        "weights_8B_resident": sram.resident(w8b),
        "weights_8B_resolves_to": "DRAM" if not sram.resident(w8b) else "SRAM",
        "weights_8B_us": round(sram.predict(Workload(op="weight_stream", nbytes=w8b))["latency_us"], 1),
        "act_4MiB_resident": sram.resident(act),
        "act_4MiB_us": round(sram.predict(Workload(op="act_tile", nbytes=act))["latency_us"], 3),
        "what_if": "if the 8B weight set FIT L2 (it cannot, 8 GB >> 32 MiB) the decode wall "
                   "would move from host LPDDR to SRAM BW — residency is the load-bearing "
                   "architecture variable (architecture-only, weights never resident).",
    }

    report = {
        "module": "m2_memory",
        "phase": "1.2",
        "anchor": {"eff_BW_GBs": ANCHOR_EFF_BW, "memory": "LPDDR4x (production Metis Card)",
                   "source": "voyager-sdk.md:278 (decode time prop weight bytes, r2=0.997)",
                   "tag": "measured (calibrated anchor)"},
        "memory_specs": mem_rows,
        "topology": topo,
        "sram_what_if": sram_whatif,
        "sim_eff_explanation": "LPDDR5 sim eff 0.65 vs measured 0.71 (LPDDR4x): LPDDR5 efficiency "
                               "was NOT measured on our silicon (a different memory), so we discount "
                               "below the measured 4x value rather than assume parity.",
        "honesty": {
            "peaks": "assumption (MT/s x 64-bit; in-repo no data-rate source)",
            "lpddr4x_24.2": "measured anchor -> calibrated",
            "lpddr5_33.3": "simulated (0.65 discount)",
            "sram_bw_latency": "CACTI assumption",
            "residency": "architecture-only (8B weights never resident)",
        },
        "ramulator2": "deferred to Phase 1.3 (ADR-0002 swappable; same constructor + contract). "
                      "NOTE LPDDR4(x) is not a first-class Ramulator2 DRAM preset — Phase 1.3 must "
                      "supply/adapt the timing config; LPDDR5 has a preset.",
        "sanity": {
            "three_specs_positive_monotonic": all(
                r["positive"] and r["monotonic_in_bytes"] for r in mem_rows.values()),
            "lpddr4x_eff_BW_is_anchor": mem_rows["mem_lpddr4x"]["eff_BW_GBs"] == ANCHOR_EFF_BW,
            "lpddr5_eff_BW_33.3": mem_rows["mem_lpddr5"]["eff_BW_GBs"] == 33.3,
            "alpha_pays_floor": topo["alpha"]["per_call_floor_us"] == 911,
            "card_pays_no_floor": topo["card"]["per_call_floor_us"] == 0,
            "weights_8B_never_resident": not sram_whatif["weights_8B_resident"],
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    s = report["sanity"]
    print(f"M2 memory report -> {OUT.relative_to(ROOT)}")
    for name, r in mem_rows.items():
        print(f"  {r['memory_type']:8s} eff_BW={r['eff_BW_GBs']:>5}  ({r['eff_BW_tag'].split(' ')[0]})")
    print(f"  alpha floor={topo['alpha']['per_call_floor_us']}us  card floor={topo['card']['per_call_floor_us']}us")
    print(f"  8B weights resident={sram_whatif['weights_8B_resident']} -> {sram_whatif['weights_8B_resolves_to']} tier")
    print(f"  sanity all-pass = {all(s.values())}")


if __name__ == "__main__":
    main()
