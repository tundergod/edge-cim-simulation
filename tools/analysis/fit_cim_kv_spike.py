"""Phase 1.5 — KV-cache isolation SPIKE: can a memory-bound proxy isolate the kv_append DRAM BW?

Reads  measurements/metis_card/cim_card_revalidate_raw.json  (kv_proxy group: K=1 memory-bound conv,
       transfer-size sweep) + validation/reports/phase1.1/m2.json (M2 measured LPDDR4x eff_BW) +
       simulator/models/params/m1_cim.json (the residency-cliff knee = on-chip SRAM capacity).
Writes validation/reports/phase1.5/kv_append_spike.json

decode kv_append is modeled analytically as kv_bytes / eff_BW (M2 streaming BW). To independently
validate that BW on-card we measure a K=1 memory-bound conv proxy (arithmetic intensity ~2) and back
out eff_BW over a TRANSFER-SIZE sweep. RESULT (honest negative): the proxy's working set (output
N*M elements) stays BELOW the ~8M residency-cliff knee (= on-chip SRAM capacity) for every M that
compiles (M<=1024; M>=2048 fails), so the data is SRAM-resident and eff_BW reflects on-chip throughput
— it RISES monotonically with transfer size (9.6 -> 44.4 GB/s) and never converges to a DRAM plateau.
The proxy therefore CANNOT isolate the kv_append DRAM bandwidth (it never reaches the DRAM regime).
Verdict: PROXY_INCONCLUSIVE. kv_append stays analytic on M2's measured DRAM streaming BW; the only
DRAM-bound on-card datapoint is the residency-cliff SPILL FLOOR (~M2-order), reported for reference.

Run: ./.venv/bin/python tools/analysis/fit_cim_kv_spike.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "measurements/metis_card/cim_card_revalidate_raw.json"
M2 = ROOT / "validation/reports/phase1.1/m2.json"
PARAMS = ROOT / "simulator/models/params/m1_cim.json"
OUT = ROOT / "validation/reports/phase1.5/kv_append_spike.json"
CONVERGE_TOL = 0.08   # |last-prev|/prev below this = plateau reached
N_PROXY = 2048        # proxy output width (single tile)


def main():
    raw = json.loads(RAW.read_text())
    m2_eff = json.loads(M2.read_text())["params"]["measured_eff_BW_GBs"]
    P = json.loads(PARAMS.read_text())
    knee = P.get("multitile_knee_kn", 8.16e6)          # ~on-chip SRAM capacity (elements)
    floor_gops = P.get("multitile_floor_gops")          # DRAM-spill throughput (the real DRAM-bound pt)

    pts = sorted(({"M": r["M"], "eff_BW_GBs": round(r["eff_BW_GBs"], 1),
                   "workset_elems": N_PROXY * r["M"],
                   "sram_resident": bool(N_PROXY * r["M"] < knee)}
                  for r in raw.values() if r.get("group") == "kv_proxy" and "eff_BW_GBs" in r),
                 key=lambda x: x["M"])
    if len(pts) < 3:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"module": "kv_append_spike", "status": "NO_DATA",
            "reason": "need >=3 kv_proxy points (run the campaign)."}, indent=1))
        print("kv_spike: NO_DATA"); return

    bw = [p["eff_BW_GBs"] for p in pts]
    rise = (bw[-1] - bw[-2]) / bw[-2]                    # still rising at the largest transfer?
    converged = abs(rise) < CONVERGE_TOL
    all_sram = all(p["sram_resident"] for p in pts)     # never reached the DRAM regime?
    # the residency-cliff spill floor IS a DRAM-bound on-card BW estimate (M=1 GEMV weight stream):
    # floor_gops/2 G-MAC/s x 1 byte/MAC = GB/s.
    spill_dram_bw = round(floor_gops / 2.0, 1) if floor_gops else None

    if all_sram and not converged:
        status = "PROXY_INCONCLUSIVE"
        honesty = ("The K=1 conv proxy is SRAM-STAGING-bound: its working set (output N*M elements) "
                   "stays below the ~%.1fM residency-cliff knee (on-chip SRAM capacity) for every M that "
                   "compiles (M<=%d; M>=%d fails), so eff_BW reflects on-chip throughput and RISES with "
                   "transfer size (%.1f -> %.1f GB/s, +%.0f%% at the last step) instead of converging to a "
                   "DRAM plateau. The proxy never reaches the DRAM regime, so it CANNOT validate the "
                   "kv_append DRAM bandwidth. kv_append stays ANALYTIC on M2's measured streaming BW "
                   "(%.1f GB/s); the earlier single-point 'M=256 ~ M2' match was coincidental."
                   % (knee / 1e6, pts[-1]["M"], pts[-1]["M"] * 2, bw[0], bw[-1], rise * 100, m2_eff))
    elif converged:
        rel = abs(bw[-1] - m2_eff) / m2_eff
        status = "CONFIRMED-CONSISTENT" if rel <= 0.15 else "RECALIBRATE"
        honesty = ("eff_BW plateaued at %.1f GB/s; vs M2 %.1f GB/s rel %.0f%%." % (bw[-1], m2_eff, rel * 100))
    else:
        status = "INCONCLUSIVE"
        honesty = "eff_BW neither plateaued nor cleanly SRAM-bound; inconclusive."

    report = {
        "module": "kv_append_spike",
        "status": status,
        "honesty": honesty,
        "verdict": {"converged": bool(converged), "all_points_sram_resident": bool(all_sram),
                    "last_step_rise_pct": round(rise * 100, 1), "bw_range_GBs": [bw[0], bw[-1]],
                    "sram_knee_M_elems": round(knee / 1e6, 2),
                    "m2_measured_eff_BW_GBs": m2_eff,
                    "spill_floor_dram_BW_GBs": spill_dram_bw,
                    "note_spill_floor": "the residency-cliff spill floor (M=1 GEMV streaming weights from "
                                        "DRAM) is the only DRAM-BOUND on-card BW datapoint; same order as M2."},
        "proxy_points": pts,
        "kv_append_basis": ("kv_append stays analytic = kv_bytes / M2 measured DRAM streaming BW (%.1f "
                            "GB/s, from the decode weight-stream wall). This SPIKE did NOT add independent "
                            "validation: the on-card proxy could not be pushed into the DRAM regime within "
                            "the compile envelope (working set < SRAM knee)." % m2_eff),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(f"kv_spike: {status} — BW {bw[0]}->{bw[-1]} GB/s (rising {rise*100:.0f}%, all SRAM-resident="
          f"{all_sram}); proxy can't reach DRAM regime. kv stays analytic on M2 {m2_eff} GB/s. -> {OUT}")


if __name__ == "__main__":
    main()
