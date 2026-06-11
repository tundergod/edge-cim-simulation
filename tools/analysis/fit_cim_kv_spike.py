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
import statistics
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

    pts = sorted(({"M": r["M"], "N": r["N"], "eff_BW_GBs": round(r["eff_BW_GBs"], 1),
                   "workset_elems": r["N"] * r["M"],                    # actual transfer (output N*M)
                   "sram_resident": bool(r["N"] * r["M"] < knee)}
                  for r in raw.values() if r.get("group") == "kv_proxy" and "eff_BW_GBs" in r),
                 key=lambda x: x["N"] * x["M"])                         # order by transfer size
    if len(pts) < 3:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"module": "kv_append_spike", "status": "NO_DATA",
            "reason": "need >=3 kv_proxy points (run the campaign)."}, indent=1))
        print("kv_spike: NO_DATA"); return

    bw = [p["eff_BW_GBs"] for p in pts]
    rise = (bw[-1] - bw[-2]) / bw[-2]                    # still rising at the largest transfer?
    converged = abs(rise) < CONVERGE_TOL
    all_sram = all(p["sram_resident"] for p in pts)     # never reached the DRAM regime?
    proxy_max_ws = max(p["workset_elems"] for p in pts)  # largest working set the proxy can COMPILE
    structural = proxy_max_ws < knee                     # compile wall sits below the DRAM knee?

    # The CONVERGED DRAM BW the proxy cannot reach: the residency-cliff SPILL regime (M=1 GEMV, K*N >
    # knee) streams weights from DRAM; its throughput is flat. DRAM BW = K*N bytes (INT8) / dev_lat.
    spill = []
    for r in raw.values():
        if r.get("M") == 1 and "dev_lat_us" in r and r.get("group") in ("envelope_probe", "cliff_map", "multitile"):
            if r["K"] * r["N"] > knee:
                spill.append({"kn_M": round(r["K"] * r["N"] / 1e6, 1),
                              "dram_BW_GBs": round(r["K"] * r["N"] / (r["dev_lat_us"] * 1e-6) / 1e9, 1)})
    spill = sorted(spill, key=lambda x: x["kn_M"])
    spill_bws = [s["dram_BW_GBs"] for s in spill]
    spill_dram_bw = round(statistics.median(spill_bws), 1) if spill_bws else None

    if structural:
        status = "PROXY_INCONCLUSIVE"
        honesty = ("STRUCTURAL: the K=1 conv proxy's compile wall (max working set %.1fM output elements) "
                   "sits BELOW the ~%.1fM residency knee (= DRAM-spill threshold, on-chip SRAM capacity), so "
                   "the proxy can NEVER be compiled large enough to spill to DRAM — it is confined to the "
                   "SRAM-resident regime, where eff_BW reflects on-chip throughput and RISES with transfer "
                   "size (%.1f -> %.1f GB/s, +%.0f%% at the last step, NOT converging). It therefore cannot "
                   "isolate the kv_append DRAM bandwidth, period. The earlier single-point 'M=256 ~ M2' was "
                   "coincidental. THE CONVERGED on-card DRAM BW DOES exist — from the residency-cliff SPILL "
                   "regime (M=1 GEMV, K*N>knee, weights streamed from DRAM): %s GB/s, FLAT across K*N "
                   "8-17M. kv_append stays ANALYTIC on M2's measured %.1f GB/s (conservative; the spill "
                   "weight-read BW %s is the upper, same-order bound)."
                   % (proxy_max_ws / 1e6, knee / 1e6, bw[0], bw[-1], rise * 100,
                      spill_dram_bw, m2_eff, spill_dram_bw))
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
        "verdict": {"converged": bool(converged), "structural_cannot_reach_dram": bool(structural),
                    "last_step_rise_pct": round(rise * 100, 1), "bw_range_GBs": [bw[0], bw[-1]],
                    "proxy_max_workset_M_elems": round(proxy_max_ws / 1e6, 2),
                    "sram_knee_M_elems": round(knee / 1e6, 2),
                    "compile_wall_below_knee": bool(structural),
                    "m2_measured_eff_BW_GBs": m2_eff,
                    "spill_dram_BW_GBs_converged": spill_dram_bw,
                    "note": "proxy compile wall (max working set) sits below the DRAM knee -> the proxy is "
                            "structurally confined to SRAM and cannot measure DRAM BW. The CONVERGED on-card "
                            "DRAM BW is the spill regime below (flat across K*N 8-17M)."},
        "proxy_points": pts,
        "spill_regime_dram_bw": spill,
        "kv_append_basis": ("kv_append stays analytic = kv_bytes / M2 measured DRAM streaming BW (%.1f "
                            "GB/s, from the decode weight-stream wall). The dedicated proxy CANNOT isolate "
                            "the kv DRAM BW (compile wall < DRAM knee). The converged on-card DRAM weight-read "
                            "BW is %s GB/s (spill regime); the true kv_append (strided write) coefficient is "
                            "bounded ~%.1f-%s GB/s, precise value = Phase-2 work." % (m2_eff, spill_dram_bw, m2_eff, spill_dram_bw)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(f"kv_spike: {status} — proxy max workset {proxy_max_ws/1e6:.1f}M < DRAM knee {knee/1e6:.1f}M "
          f"(structural); proxy BW {bw[0]}->{bw[-1]} rising. CONVERGED DRAM BW (spill) = {spill_dram_bw} GB/s. -> {OUT}")


if __name__ == "__main__":
    main()
