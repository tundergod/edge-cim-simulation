"""Phase 1.2 — fit the CIM PREFILL M-amortization + cross-validate vs vendor TTFT.

Reads  measurements/metis_card/cim_card_revalidate_raw.json  (Card tile sweep, M in {1,64,128,256})
       measurements/metis_card/twopillar_prediction_fitted.json  (vendor 8B prefill TTFT)
Writes simulator/models/params/m1_cim.json   (ADDS prefill_tile_* keys; preserves the decode G_eff fit)
       validation/reports/phase1.2/cim_prefill_fit.json

The decode (M=1) G_eff(N,K) fit captures the WEIGHT-LOAD-bound regime only; extrapolating it
linearly in M to prefill over-predicts ~80x (decode GEMV ~204 GOP/s is hit at M=1, the 2048x2048
weight tile loaded for a SINGLE column). At larger M the same load amortizes over M columns, so the
per-tile latency is AFFINE in M:

    tile_lat(M) = a + b*M     (a = weight-load floor us; b = per-column compute us)

fit ONLY on the genuine PREFILL points M in {64,128,256} (full 2048x2048 tiles, the compilable
range; M>256 fails on v1.6 axcompile's SRAM L1/L2 wall). The M=1 decode point is a DIFFERENT regime
(direct single-column compile, modeled by G_eff) and is reported separately, NOT mixed into the
prefill line. A GEMM = (K*N / W^2) * tile_lat(M) (FRACTIONAL tile area, not ceil). M>256 or partial-
width tiles are analytic extrapolation (CimTileModel.prefill_extrapolated). main() REFUSES to write
params if the fit is degenerate (max rel-err > 0.05), and self-computes the decode cross-val median
(no dependency on validate_cim_card.py's output). NOTE: fit_m1_cim.py now PRESERVES these keys.

Run: ./.venv/bin/python tools/analysis/fit_cim_prefill.py
"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from simulator.models.m1_cim_tile import CimTileModel  # noqa: E402

RAW = ROOT / "measurements/metis_card/cim_card_revalidate_raw.json"
PARAMS = ROOT / "simulator/models/params/m1_cim.json"
REPORT = ROOT / "validation/reports/phase1.2/cim_prefill_fit.json"
TILE_KN = 2048 * 2048   # canonical tile the M-amortization is measured at


def tile_points():
    """(M, tile_lat_us) at the canonical 2048x2048 tile: M=1 from alpha13 square, M>1 from
    each prefill task's cached tile latency (de-duplicated per M)."""
    raw = json.loads(RAW.read_text())
    pts = {}
    sq = raw.get("alpha13|native|M1|K2048|N2048")
    if sq and "dev_lat_us" in sq:
        pts[1] = sq["dev_lat_us"]
    for r in raw.values():
        if r.get("group") == "prefill" and "tile_dev_lat_us" in r:
            pts.setdefault(r["M"], r["tile_dev_lat_us"])   # same tile per M (cache); first wins
    return sorted(pts.items())


def main():
    pts = tile_points()                                       # [(1,..),(64,..),(128,..),(256,..)]
    decode_pt = next((p for p in pts if p[0] == 1), None)     # M=1 decode anchor (reported, NOT fit)
    prefill = [p for p in pts if p[0] > 1]                    # M>=64 = the genuine prefill fit points
    if len(prefill) < 2:
        sys.exit(f"fit_cim_prefill: need >=2 prefill points (M>1); got {len(prefill)}")
    M = np.array([m for m, _ in prefill], float)
    lat = np.array([t for _, t in prefill], float)
    b, a = np.polyfit(M, lat, 1)                              # lat = a + b*M (polyfit -> [slope, intercept])
    resid = np.abs((a + b * M) - lat) / lat
    max_rel_err = float(np.max(resid))
    asymptote_tops = 2 * TILE_KN / (b * 1e6)                  # M->inf throughput TOPS (b in us/col)

    # #27(3): a degenerate fit must NOT silently become the production params.
    if max_rel_err > 0.05:
        sys.exit(f"fit_cim_prefill: prefill fit max rel-err {max_rel_err:.3f} > 0.05 -> refusing to "
                 f"write m1_cim.json (investigate the Card prefill points {list(zip(M.tolist(), lat.tolist()))})")

    # --- ADD prefill keys to m1_cim.json (preserve the decode G_eff fit) ---
    params = json.loads(PARAMS.read_text())
    params.pop("prefill_M_measured", None)   # superseded by prefill_M_fit (drop stale key from older runs)
    params["prefill_tile_a_us"] = round(float(a), 3)
    params["prefill_tile_b_us"] = round(float(b), 4)
    params["prefill_tile_kn"] = TILE_KN
    params["prefill_M_fit"] = [int(m) for m in M]             # the M>=64 prefill points the line is fit on
    params["prefill_M_max"] = 256
    params["prefill_M_decode_anchor"] = ({"M": 1, "tile_lat_us": round(decode_pt[1], 2),
        "gops_measured": round(2 * TILE_KN / decode_pt[1] / 1e3, 1),
        "note": "measured decode (M=1) tile on its measured basis; NOT in the prefill fit (different "
                "regime: direct single-column compile, modeled by G_eff)"} if decode_pt else None)
    params["_prefill_doc"] = ("prefill (M>1): tile_lat(M)=a+b*M fit on full 2048x2048 tiles at "
                              "M in {64,128,256}; GEMM = (K*N/W^2)*tile_lat (fractional area, not ceil). "
                              "M>256 / partial-width tiles extrapolated (CimTileModel.prefill_extrapolated).")
    # E16 (SELF-CONTAINED, no dep on validate_cim_card.py's output -> no run-order hazard): the decode
    # G_eff fit is Card-CONFIRMED when the cross-val median is within the decode tolerance. Compute it
    # HERE from the raw alpha13 dev_gflops vs the Alpha native pts (same formula as validate_cim_card).
    alpha_pts = {(p["N"], p["K"]): p["gops"] for p in params["native_throughput_points"]}
    raw = json.loads(RAW.read_text())
    diffs = sorted(abs(r["dev_gflops"] - alpha_pts[(r["N"], r["K"])]) / alpha_pts[(r["N"], r["K"])]
                   for r in raw.values() if r.get("group") == "alpha13" and "dev_gflops" in r
                   and (r["N"], r["K"]) in alpha_pts)
    if not diffs:
        sys.exit("fit_cim_prefill: no alpha13 cross points in the raw file -> cannot assess the decode "
                 "revalidation; run the Card measurement first")
    MIN_CROSS_N = 8                                          # a spike (n=1) must not self-certify (#36)
    med = round(statistics.median(diffs), 3)
    p95 = round(float(np.percentile(diffs, 95)), 3)          # linear-interp quantile, not nearest-rank-down (#37)
    decision = ("INSUFFICIENT-CROSS-POINTS: only %d/%d alpha13 pts (< %d) -> decode revalidation unconfirmed"
                % (len(diffs), 13, MIN_CROSS_N) if len(diffs) < MIN_CROSS_N
                else "CONFIRMED-ON-CARD: kept Alpha fit (un-frozen)" if med <= 0.10
                else "RE-FIT NEEDED: Card cross-val exceeds the 0.10 decode tolerance")
    params["decode_card_revalidation"] = {
        "median_rel_diff": med, "p95_rel_diff": p95, "n": len(diffs), "decision": decision,
        "note": "median |rel_diff| over the alpha13 pts (computed here; matches validate_cim_card). "
                "<= 0.10 (with n >= %d) -> the Alpha G_eff is Card-confirmed, freeze lifted." % MIN_CROSS_N}
    PARAMS.write_text(json.dumps(params, indent=1))
    m = CimTileModel(params)

    # --- validate the model reproduces the measured full prefill GEMMs ---
    gemm_rows = []
    for tid, r in raw.items():
        if r.get("group") != "prefill":
            continue
        p = m.dev_lat_us(r["M"], r["K"], r["N"])
        gemm_rows.append({"M": r["M"], "K": r["K"], "N": r["N"], "tiles": r["tiles"],
                          "extrapolated": m.prefill_extrapolated(r["M"], r["K"], r["N"]),
                          "meas_us": round(r["dev_lat_us"], 1), "pred_us": round(p, 1),
                          "rel_err": round(abs(p - r["dev_lat_us"]) / r["dev_lat_us"], 3)})

    # --- monotonicity: tile throughput rises with M. M=1 on the MEASURED basis, M>=64 from the fit ---
    thru = []
    if decode_pt:
        thru.append((1, round(2 * TILE_KN / decode_pt[1] / 1e3, 1)))          # measured M=1, NOT fitted
    thru += [(int(mm), round(float(2 * TILE_KN * mm / (a + b * mm) / 1e3), 1)) for mm in M]
    monotone = all(thru[i][1] < thru[i + 1][1] for i in range(len(thru) - 1))

    report = {
        "module": "cim_prefill_fit",
        "honesty": "Prefill GEMM M-amortization MEASURED on the Card (1x1-conv proxy, dev FPS) at the "
                   "FULL-tile prefill points M in {64,128,256}; M>256 fails to compile (v1.6 SRAM wall) "
                   "-> extrapolation. The M=1 decode point is reported separately (different regime), "
                   "NOT in the prefill fit. Decode (M=1) G_eff fit unchanged.",
        "tile": {"kn": TILE_KN, "shape": "2048x2048"},
        "affine_fit_tile_lat_us": {"a_weight_load_us": round(float(a), 3), "b_per_col_us": round(float(b), 4),
                                   "asymptote_TOPS": round(float(asymptote_tops), 1),
                                   "fit_basis": "M in {64,128,256} (prefill only; M=1 decode excluded)"},
        "decode_anchor_M1_measured": params["prefill_M_decode_anchor"],
        "fit_points": [{"M": int(mm), "meas_us": round(tt, 2), "pred_us": round(float(a + b * mm), 2),
                        "rel_err": round(float(rr), 3)} for mm, tt, rr in zip(M, lat, resid)],
        "fit_quality": {"median_rel_err": round(float(np.median(resid)), 3), "max_rel_err": round(max_rel_err, 3),
                        "pass_max_le_0.05": bool(max_rel_err <= 0.05),
                        "enforced": "main() sys.exit if max_rel_err > 0.05 -> params not written"},
        "full_gemm_meas_vs_pred": gemm_rows,
        "tile_throughput_monotone_in_M": {"GOP_s_by_M": dict(thru), "monotone_increasing": bool(monotone),
                                          "note": "M=1 on the measured decode basis; M>=64 from the fit"},
        "decode_card_revalidation": params["decode_card_revalidation"],
        "ttft_cross_val_note": "8B prefill TTFT cross-validation (linear-M refuted vs M-amortized) is in "
                               "recompose_e2e.py (P14) -> twopillar_prediction_fitted.json.",
        "extrapolation_note": "M>256 OR partial-width tiles (K or N not a multiple of 2048) extrapolate "
                              "beyond the full-tile measured range; CimTileModel.prefill_extrapolated flags them.",
    }
    REPORT.write_text(json.dumps(report, indent=1))

    print(f"prefill fit (M>=64 only): tile_lat(M) = {a:.2f} + {b:.4f}*M us (asymptote {asymptote_tops:.0f} TOPS) "
          f"| max rel-err {max_rel_err:.3f} (PASS, enforced)")
    if decode_pt:
        print(f"  M=1 decode anchor (separate, measured): {decode_pt[1]:.1f}us = {2*TILE_KN/decode_pt[1]/1e3:.1f} GOP/s")
    print(f"  full-GEMM meas-vs-pred max rel-err: {max(r['rel_err'] for r in gemm_rows):.3f}")
    print(f"  decode revalidation: median |rel_diff|={med} -> {params['decode_card_revalidation']['decision'].split(':')[0]}")


if __name__ == "__main__":
    main()
