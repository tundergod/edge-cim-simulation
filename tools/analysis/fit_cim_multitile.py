"""Phase 1.5 — fit the native multi-tile RESIDENCY-CLIFF model + validate held-out.

Reads  measurements/metis_card/cim_card_revalidate_raw.json  (Card native M=1 multi-tile sweep:
       alpha13 / envelope_probe / cliff_map / multitile groups)
Writes simulator/models/params/m1_cim.json   (ADDS multitile_* + native_envelope_kn; preserves the
       decode G_eff + prefill keys)
       validation/reports/phase1.5/cim_multitile.json

The OLD model tile-summed every multi-tile GEMM at the single-tile throughput, which (a) OVER-predicts
below the residency knee (the compiler fuses tiles, faster than the sum) and (b) UNDER-predicts
~-65% above it (it has no cliff) — +31% abs median error overall. The Card native sweep (M=1, some
dim > W=2048) shows: throughput
rises smoothly to ~264 GOP/s up to a knee at K*N ~8M (weights resident in on-chip SRAM), then collapses
~3.5x to a ~70 GOP/s memory-bound floor (weights spill to DRAM). Two-regime model:

    resident (K*N <= knee):  lat = a_r + b_r * (K*N)        # weights in SRAM, affine in K*N
    spill    (K*N >  knee):  lat = 2*K*N / (G_floor * 1e9)  # DRAM-bound floor throughput

The knee is auto-detected as the midpoint of the throughput cliff (the adjacent-K*N pair where GOP/s
drops > 2x). The fit is held-out validated (fit on half the resident points, predict the other half);
main() REFUSES to write params if the resident held-out median exceeds 0.10.

Run: ./.venv/bin/python tools/analysis/fit_cim_multitile.py
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
REPORT = ROOT / "validation/reports/phase1.5/cim_multitile.json"
W = 2048  # combined tile width (n_cores*512); "multi-tile" = some dim > W
GROUPS = ("alpha13", "envelope_probe", "cliff_map", "multitile", "k_staircase")


def native_multitile_points():
    """Native M=1 multi-tile points (some dim > W): (K, N, K*N, dev_lat_us, dev_gflops), K*N-sorted,
    de-duplicated per (K,N) (first wins)."""
    raw = json.loads(RAW.read_text())
    seen = {}
    for r in raw.values():
        if r.get("M") != 1 or r.get("group") not in GROUPS or "dev_lat_us" not in r:
            continue
        K, N = r["K"], r["N"]
        if K <= W and N <= W:           # single tile -> the G_eff path, not this model
            continue
        seen.setdefault((K, N), (K * N, r["dev_lat_us"], r["dev_gflops"]))
    return sorted([(K, N, kn, lat, g) for (K, N), (kn, lat, g) in seen.items()], key=lambda x: x[2])


def detect_knee(pts):
    """Knee K*N = midpoint of the adjacent-K*N pair whose throughput drops > 2x (the cliff)."""
    for i in range(1, len(pts)):
        if pts[i][4] < pts[i - 1][4] / 2.0:        # GOP/s halves -> cliff
            return (pts[i - 1][2] + pts[i][2]) / 2.0
    return pts[-1][2]                              # no cliff seen -> all resident


def main():
    pts = native_multitile_points()
    if len(pts) < 6:
        sys.exit(f"fit_cim_multitile: need >=6 native multi-tile pts; got {len(pts)} (run the Card campaign)")
    knee = detect_knee(pts)
    resident = [p for p in pts if p[2] <= knee]
    spill = [p for p in pts if p[2] > knee]
    if len(resident) < 4 or len(spill) < 1:
        sys.exit(f"fit_cim_multitile: degenerate split (resident={len(resident)} spill={len(spill)}) "
                 f"at knee={knee/1e6:.2f}M — inspect the native sweep")

    # --- resident: lat = a_r + b_r * K*N (affine), fit on ALL resident pts ---
    KN = np.array([p[2] for p in resident], float)
    L = np.array([p[3] for p in resident], float)
    b_r, a_r = np.polyfit(KN, L, 1)
    res_err = np.abs((a_r + b_r * KN) - L) / L

    # --- held-out: split on UNIQUE K*N (the resident model is f(K*N), and symmetric K<->N pairs share
    # a K*N — splitting raw indices would leak those values into both sets). Hold out every-other
    # distinct K*N so test points are genuinely unseen. ---
    uniq = sorted(set(KN.tolist()))
    te_kn = set(uniq[1::2])
    te_mask = np.array([k in te_kn for k in KN])
    tr_mask = ~te_mask
    tr, te = np.where(tr_mask)[0], np.where(te_mask)[0]
    b2, a2 = np.polyfit(KN[tr], L[tr], 1)
    ho = np.abs((a2 + b2 * KN[te]) - L[te]) / L[te]
    ho_median = float(np.median(ho))
    if ho_median > 0.10:
        sys.exit(f"fit_cim_multitile: resident held-out median {ho_median:.3f} > 0.10 -> refusing to write "
                 f"(the affine-in-K*N resident assumption does not hold; inspect {[(p[0],p[1]) for p in resident]})")

    # --- spill floor: G_floor = mean measured throughput above the knee ---
    g_floor = float(np.mean([p[4] for p in spill]))
    native_envelope_kn = int(max(p[2] for p in pts))

    # --- write params (ADD multitile keys; preserve G_eff + prefill) ---
    params = json.loads(PARAMS.read_text())
    params["multitile_knee_kn"] = int(round(knee))
    params["multitile_resident_a_us"] = round(float(a_r), 3)
    params["multitile_resident_b_us"] = float(f"{b_r:.4e}")        # us per K*N param (~5.8e-6)
    params["multitile_floor_gops"] = round(g_floor, 1)
    params["native_envelope_kn"] = native_envelope_kn
    params["_multitile_doc"] = ("native M=1 multi-tile RESIDENCY CLIFF (Card, Phase 1.5): resident "
        "(K*N<=knee) lat=a+b*K*N (SRAM-resident); spill (K*N>knee) lat=2*K*N/(floor_gops) (DRAM-bound). "
        "Replaced the tile-sum (over below knee / ~-65% under above; +31% abs median overall). "
        "K*N>native_envelope_kn = floor-extrapolated.")
    PARAMS.write_text(json.dumps(params, indent=1))
    m = CimTileModel(params)

    # --- old tile-sum vs new cliff model on EVERY native multi-tile pt (the improvement) ---
    rows, old_err, new_err = [], [], []
    for K, N, kn, lat, g in pts:
        old = m._decode_lat_us(K, N)               # tile-sum (the pre-cliff behavior)
        new = m._decode_multitile_lat_us(K, N)     # the fitted cliff model
        oe, ne = abs(old - lat) / lat, abs(new - lat) / lat
        old_err.append(oe); new_err.append(ne)
        rows.append({"K": K, "N": N, "kn_M": round(kn / 1e6, 2), "regime": "resident" if kn <= knee else "spill",
                     "meas_us": round(lat, 1), "old_tilesum_us": round(old, 1), "new_cliff_us": round(new, 1),
                     "old_relerr": round(oe, 3), "new_relerr": round(ne, 3)})

    report = {
        "module": "cim_multitile",
        "honesty": "native M=1 multi-tile (some dim > 2048) measured DIRECTLY on the Card (axcompile "
                   "compiles multi-tile natively up to K*N~16.8M; the old SAFE_KN=4.19M 'wall' was a "
                   "conservative assumption). The +36% Alpha tile-sum point is SUPERSEDED by this "
                   "Card-native characterization. K*N > native_envelope_kn is floor-extrapolated.",
        "model": {"knee_kn": params["multitile_knee_kn"], "knee_M_params": round(knee / 1e6, 2),
                  "resident_a_us": params["multitile_resident_a_us"], "resident_b_us_per_param": params["multitile_resident_b_us"],
                  "spill_floor_gops": params["multitile_floor_gops"], "native_envelope_kn": native_envelope_kn,
                  "physical": "knee ~ on-chip SRAM weight capacity; below = compute-bound (resident), "
                              "above = memory-bound (DRAM spill, ~M2 streaming BW)."},
        "resident_fit": {"n": len(resident), "median_relerr": round(float(np.median(res_err)), 3),
                         "max_relerr": round(float(res_err.max()), 3)},
        "resident_holdout": {"n_fit": len(tr), "n_pred": len(te), "median_relerr": round(ho_median, 3),
                             "max_relerr": round(float(ho.max()), 3),
                             "gate": "median <= 0.10 (enforced; main() sys.exit otherwise)"},
        "spill_fit": {"n": len(spill), "floor_gops": params["multitile_floor_gops"],
                      "max_relerr": round(max(abs(2 * p[2] / (g_floor * 1e9) * 1e6 - p[3]) / p[3] for p in spill), 3)},
        "old_vs_new": {"n": len(pts),
                       "old_tilesum_median": round(float(np.median(old_err)), 3), "old_tilesum_max": round(float(max(old_err)), 3),
                       "new_cliff_median": round(float(np.median(new_err)), 3), "new_cliff_max": round(float(max(new_err)), 3)},
        "per_point": rows,
        "phase2_note": "real decode FFN/lm_head GEMVs (K*N >= 16M) live in the SPILL regime; the old "
                       "tile-sum under-predicted them ~3x. The per-op M1 latency the Phase-2 event "
                       "engine consumes is now cliff-correct, not just the aggregate recompose BW.",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1))

    print(f"multitile cliff: knee={knee/1e6:.2f}M params | resident lat={a_r:.2f}+{b_r:.3e}*K*N us "
          f"(holdout median {ho_median*100:.1f}%) | spill floor={g_floor:.1f} GOP/s | envelope={native_envelope_kn/1e6:.1f}M")
    print(f"  OLD tile-sum vs Card: median={np.median(old_err)*100:.0f}% max={max(old_err)*100:.0f}%")
    print(f"  NEW cliff    vs Card: median={np.median(new_err)*100:.1f}% max={max(new_err)*100:.1f}%")


if __name__ == "__main__":
    main()
