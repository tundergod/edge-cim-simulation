"""Phase 1.2 — CIM-Card re-validation report (cross-check Card vs Alpha 13 points).

Reads  measurements/metis_card/cim_card_revalidate_raw.json  (produced by
       characterization/metis_card/run_metis_cim_v16.py ON THE CARD; absent until SSH-run)
       simulator/models/params/m1_cim.json  (the Alpha 13 native throughput points)
Writes validation/reports/phase1.2/cim_card_revalidate.json

Both boards run the SAME quad-core AIPU at 800MHz (machines.md) -> dev_gflops are DIRECTLY
comparable, NO clock rescale. axrunmodel dev FPS = isolated compute (dev/system split), not a
synthetic difference.

FALLBACK (the current state): if no Card raw file exists (SSH to metiscard not authorized this
session), this writes the DOCUMENTED FALLBACK report: CIM stays calibrated on the Alpha 13 points
(non-frozen, pending board); Card revalidation is deferred. This is the plan's fallback, surfaced
honestly — not a silent path change.

Run: ./.venv/bin/python tools/analysis/validate_cim_card.py
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "measurements/metis_card/cim_card_revalidate_raw.json"
M1 = ROOT / "simulator/models/params/m1_cim.json"
OUT = ROOT / "validation/reports/phase1.2/cim_card_revalidate.json"


def main():
    alpha = json.loads(M1.read_text())
    alpha_pts = {(p["N"], p["K"]): p["gops"] for p in alpha["native_throughput_points"]}

    if not RAW.exists():
        report = {
            "module": "cim_card_revalidate",
            "status": "DEFERRED_FALLBACK",
            "honesty": "CIM = Alpha 13pts CALIBRATED (non-frozen, pending board); "
                       "Card revalidation DEFERRED.",
            "reason": "SSH to metiscard was not authorized this session (harness denied remote-shell "
                      "to the shared board; user asleep). No Card raw measurements available.",
            "fallback_per_plan": "plans/phase-1.2.md step 18 note + handoff §5: low-level compile "
                                 "absent / MatMul unsupported / board unreachable -> 'Alpha 13pts "
                                 "calibrated (pending board) + Card e2e validates memory-wall', report user.",
            "alpha_13_points_NK_gops": {f"N{n}K{k}": g for (n, k), g in alpha_pts.items()},
            "to_unblock": "grant SSH (add Bash rule for `ssh metiscard`), then: rsync the port to the "
                          "Card, run `python run_metis_cim_v16.py --spike` (feasibility), then full, "
                          "rsync results back, re-run this validator.",
            "both_boards_800MHz": "machines.md: Alpha + Card both clock=800MHz -> dev_gflops directly "
                                  "comparable, no rescale (applies once Card data exists).",
        }
        OUT.write_text(json.dumps(report, indent=1))
        print(f"cim_card_revalidate: DEFERRED_FALLBACK (no Card data; SSH not authorized). "
              f"CIM stays Alpha-13pts calibrated. -> {OUT}")
        return

    # --- Card data present: cross-validate dev_gflops vs Alpha (same AIPU, 800MHz, no rescale) ---
    raw = json.loads(RAW.read_text())
    cross, prefill = [], []
    for tid, r in raw.items():
        if "dev_gflops" not in r:
            continue
        N, K, M = r["N"], r["K"], r["M"]
        if r.get("group") == "alpha13" and (N, K) in alpha_pts:
            a = alpha_pts[(N, K)]
            cross.append({"N": N, "K": K, "card_gops": round(r["dev_gflops"], 1), "alpha_gops": a,
                          "rel_diff": round(abs(r["dev_gflops"] - a) / a, 3)})
        elif r.get("group") == "prefill":
            prefill.append({"M": M, "K": K, "N": N, "card_gops": round(r["dev_gflops"], 1),
                            "dev_lat_us": round(r["dev_lat_us"], 1),
                            "tiled_extrapolated": bool(r.get("tiled_extrapolated"))})

    # --- tolerance gate: CARD_REVALIDATED only when the Card actually TRACKS Alpha within the decode
    # fit tolerance (matches fit_m1_cim's native gate). Otherwise the status must say it DISAGREES. ---
    TOL_MEDIAN, TOL_P95 = 0.10, 0.20
    diffs = [c["rel_diff"] for c in cross]
    median = round(statistics.median(diffs), 3) if diffs else None
    p95 = round(sorted(diffs)[int(0.95 * (len(diffs) - 1))], 3) if diffs else None
    if not cross:
        status = "NO_CROSS_POINTS"
    elif median <= TOL_MEDIAN and p95 <= TOL_P95:
        status = "CARD_REVALIDATED"
    else:
        status = "CARD_DISAGREES"
    honesty = {
        "CARD_REVALIDATED": "CIM = Alpha 13pts calibrated + Card-revalidated (same AIPU, 800MHz, no rescale).",
        "CARD_DISAGREES": "CIM = Alpha 13pts calibrated; Card cross-val median %s / p95 %s EXCEEDS tolerance "
                          "(%s/%s) -> Card DISAGREES, decode fit NOT confirmed." % (median, p95, TOL_MEDIAN, TOL_P95),
        "NO_CROSS_POINTS": "CIM = Alpha 13pts calibrated; no Card cross points (cannot revalidate).",
    }[status]
    report = {
        "module": "cim_card_revalidate",
        "status": status,
        "tolerance": {"median_rel_diff": TOL_MEDIAN, "p95_rel_diff": TOL_P95},
        "honesty": honesty,
        "compile_path": next((r.get("compile_path") for r in raw.values() if r.get("compile_path")), "?"),
        "cross_validation_alpha13": cross,
        "consistency": {"n": len(cross), "median_rel_diff": median, "p95_rel_diff": p95},
        "prefill_compute_bound_NEW": prefill,
        "note": "axrunmodel dev FPS = isolated compute (dev/system split). prefill rows with "
                "tiled_extrapolated=true are n_tiles x ONE measured 2048x2048 tile (the full GEMM was "
                "NOT compiled directly; only the canonical tile was measured) -> card_gops is extrapolated.",
    }
    OUT.write_text(json.dumps(report, indent=1))
    md = report["consistency"]["median_rel_diff"]
    print(f"cim_card_revalidate: {report['status']} — {len(cross)} cross pts, "
          f"median |diff|={md}, {len(prefill)} prefill pts. compile={report['compile_path']} -> {OUT}")


if __name__ == "__main__":
    main()
