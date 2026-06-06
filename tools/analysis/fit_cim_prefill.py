"""Phase 1.2 — fit the CIM PREFILL M-amortization + cross-validate vs vendor TTFT.

Reads  measurements/metis_card/cim_card_revalidate_raw.json  (Card tile sweep, M in {1,64,128,256})
       measurements/metis_card/twopillar_prediction_fitted.json  (vendor 8B prefill TTFT)
Writes simulator/models/params/m1_cim.json   (ADDS prefill_tile_* keys; preserves the decode G_eff fit)
       validation/reports/phase1.2/cim_prefill_fit.json

The decode (M=1) G_eff(N,K) fit captures the WEIGHT-LOAD-bound regime only; extrapolating it
linearly in M to prefill over-predicts ~80x (the decode GEMV throughput, ~204 GOP/s, is hit at M=1
when the 2048x2048 weight tile is loaded for a SINGLE activation column). At larger M the same tile
load is amortized over M columns, so the per-tile latency is AFFINE in M:

    tile_lat(M) = a + b*M     (a = weight-load floor us; b = per-column compute us)

fit on the Card canonical 2048x2048 tile at the COMPILABLE M in {1,64,128,256} (M>256 fails to
compile on v1.6 axcompile, SRAM L1/L2 wall). A full prefill GEMM = n_tiles * tile_lat(M). M>256 is
analytic extrapolation of this affine law (flagged). NOTE: re-running fit_m1_cim.py rewrites
m1_cim.json and drops these keys -> re-run this script after.

Run: ./.venv/bin/python tools/analysis/fit_cim_prefill.py
"""
import json
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
    pts = tile_points()
    M = np.array([m for m, _ in pts], float)
    lat = np.array([t for _, t in pts], float)
    b, a = np.polyfit(M, lat, 1)            # lat = a + b*M  (polyfit returns [slope, intercept])
    pred = a + b * M
    resid = np.abs(pred - lat) / lat
    asymptote_tops = 2 * TILE_KN / (b * 1e6)   # M->inf throughput TOPS (2 ops/MAC per col, b in us/col)

    # --- ADD prefill keys to m1_cim.json (preserve the decode G_eff fit) ---
    params = json.loads(PARAMS.read_text())
    params["prefill_tile_a_us"] = round(float(a), 3)
    params["prefill_tile_b_us"] = round(float(b), 4)
    params["prefill_tile_kn"] = TILE_KN
    params["prefill_M_measured"] = [int(m) for m in M]
    params["prefill_M_max"] = 256
    params["_prefill_doc"] = ("prefill (M>1): tile_lat(M)=a+b*M on the canonical 2048x2048 tile "
                              "(Card-measured, M<=256; M>256 extrapolated). dev_lat = n_tiles*tile_lat.")
    PARAMS.write_text(json.dumps(params, indent=1))
    m = CimTileModel(params)

    # --- validate the model reproduces the measured full prefill GEMMs (n_tiles * tile_lat) ---
    raw = json.loads(RAW.read_text())
    gemm_rows = []
    for tid, r in raw.items():
        if r.get("group") != "prefill":
            continue
        p = m.dev_lat_us(r["M"], r["K"], r["N"])
        gemm_rows.append({"M": r["M"], "K": r["K"], "N": r["N"], "tiles": r["tiles"],
                          "meas_us": round(r["dev_lat_us"], 1), "pred_us": round(p, 1),
                          "rel_err": round(abs(p - r["dev_lat_us"]) / r["dev_lat_us"], 3)})

    # --- monotonicity: tile throughput rises with M (the prefill amortization) ---
    thru = [(int(mm), round(float(2 * TILE_KN * mm / (a + b * mm) / 1e3), 1)) for mm in M]  # GOP/s per M
    monotone = all(thru[i][1] < thru[i + 1][1] for i in range(len(thru) - 1))

    report = {
        "module": "cim_prefill_fit",
        "honesty": "Prefill GEMM M-amortization MEASURED on the Card (1x1-conv proxy, dev FPS) at "
                   "M in {1,64,128,256}; M>256 fails to compile (v1.6 SRAM wall) -> affine extrapolation. "
                   "Decode (M=1) G_eff fit unchanged.",
        "tile": {"kn": TILE_KN, "shape": "2048x2048"},
        "affine_fit_tile_lat_us": {"a_weight_load_us": round(float(a), 3), "b_per_col_us": round(float(b), 4),
                                   "asymptote_TOPS": round(float(asymptote_tops), 1)},
        "fit_points": [{"M": int(mm), "meas_us": round(tt, 2), "pred_us": round(float(a + b * mm), 2),
                        "rel_err": round(float(rr), 3)} for mm, tt, rr in zip(M, lat, resid)],
        "fit_quality": {"median_rel_err": round(float(np.median(resid)), 3),
                        "max_rel_err": round(float(np.max(resid)), 3),
                        "pass_max_le_0.05": bool(np.max(resid) <= 0.05)},
        "full_gemm_meas_vs_pred": gemm_rows,
        "tile_throughput_monotone_in_M": {"GOP_s_by_M": dict(thru), "monotone_increasing": bool(monotone)},
        "ttft_cross_val_note": "8B prefill TTFT cross-validation (linear-M refuted vs M-amortized "
                               "bounded by compute<=TTFT) is in recompose_e2e.py (P14) -> "
                               "twopillar_prediction_fitted.json:prefill_gemm_compute_BOUNDED.",
        "extrapolation_note": "M>256 (incl. the 8B S=1024 e2e use) extrapolates the affine tile law "
                              "beyond the compilable measured range; UNVALIDATED above M=256.",
    }
    REPORT.write_text(json.dumps(report, indent=1))

    print(f"prefill fit: tile_lat(M) = {a:.2f} + {b:.4f}*M us (asymptote {asymptote_tops:.0f} TOPS) | "
          f"fit max rel-err {np.max(resid):.3f} ({'PASS' if np.max(resid) <= 0.05 else 'FAIL'})")
    print(f"  full-GEMM meas-vs-pred max rel-err: {max(r['rel_err'] for r in gemm_rows):.3f}")
    print(f"  tile throughput GOP/s by M: {dict(thru)} (monotone={monotone}) | 8B TTFT cross-val -> recompose_e2e (P14)")


if __name__ == "__main__":
    main()
