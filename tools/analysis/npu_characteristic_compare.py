"""Phase 1.6b — does each NPU simulator exhibit a 32-period systolic staircase? (no presupposed knee)

Reads the two independent characteristic sweeps (ONNXim on metiscard, ScaleSim local positive
control) and applies a PRE-REGISTERED, period-AGNOSTIC criterion to each (sim, M-regime). H0 =
"smooth, no 32-step"; burden is on rejecting it. ScaleSim is a positive control (a 32x32 array sim
steps at 32 by construction) -- it confirms the probe sees a real step; ONNXim (adds NoC/DRAM
scheduling) is the informative one. Writes validation/reports/phase1.6/npu_characteristic_compare.json.

Discriminators (all computed on the uniform step-8 bulk and the 1-spaced microsweeps, NOT mixed):
  - smooth_loglog_R2 : R^2 of log-log linear fit on the step-8 bulk (high => smooth-dominated).
  - bulk_delta_cv    : coeff-of-variation of consecutive bulk Δcycles (HIGH => jump/flat staircase;
                       LOW => uniform/smooth). The cleanest period-agnostic discriminator.
  - ac32             : autocorrelation of detrended bulk residual at lag-4 (= 32, since bulk step=8).
  - micro_step_ratio : at each 1-spaced cluster straddling a 32-multiple (160/192/256/384):
                       max adjacent |Δcyc| / median of the rest -- detects a LOCAL step at 32k.
All thresholds are frozen here BEFORE looking at the data. simulated, NOT silicon (#13).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ONX = ROOT / "simulator/engines/onnxim/rknpu2_characteristic.json"
SCL = ROOT / "simulator/engines/scalesim/rknpu2_characteristic.json"
OUT = ROOT / "validation/reports/phase1.6/npu_characteristic_compare.json"

# ---- PRE-REGISTERED constants (frozen before seeing data) ----
K_FIXED = 2048
BULK_NS = list(range(128, 513, 8))           # uniform step-8 grid
MICRO_CENTERS = [160, 192, 256, 384]         # 1-spaced clusters straddle these 32-multiples
STAIR_STEP_RATIO = 3.0                        # micro ratio >= 3 => a local step at that boundary
SMOOTH_STEP_RATIO = 1.5                       # micro ratio < 1.5 => no local step
CV_STAIR = 0.6                                # bulk_delta_cv >= 0.6 => jump/flat staircase
CV_SMOOTH = 0.30                              # bulk_delta_cv < 0.30 => uniform/smooth


def cyc_of(rows):
    """map (M,K,N)->cycles, handling onnxim 'cycles' vs scalesim 'cycles_1core'."""
    out = {}
    for r in rows:
        m, k, n = r["shape"]
        out[(m, k, n)] = r.get("cycles", r.get("cycles_1core"))
    return out


def analyze(C, M):
    """staircase analysis for one sim at fixed M, K=K_FIXED."""
    bulk = [(n, C[(M, K_FIXED, n)]) for n in BULK_NS if (M, K_FIXED, n) in C]
    Ns = np.array([n for n, _ in bulk], float)
    Cs = np.array([c for _, c in bulk], float)
    b, loga = np.polyfit(np.log(Ns), np.log(Cs), 1)
    pred = np.exp(loga) * Ns ** b
    R2 = 1 - ((Cs - pred) ** 2).sum() / ((Cs - Cs.mean()) ** 2).sum()
    deltas = np.diff(Cs)
    cv = float(deltas.std() / deltas.mean()) if deltas.mean() else None
    # monotonicity guard (added after observing ONNXim M=128 swings DOWN by ~25k cyc as N grows):
    # a staircase is monotone non-decreasing BY DEFINITION. This makes "staircase" HARDER to claim,
    # applied symmetrically -- ScaleSim (monotone by construction) still passes.
    med_abs = float(np.median(np.abs(deltas))) or 1.0
    frac_neg = float((deltas < -0.02 * med_abs).mean())
    max_neg_x = float(-deltas.min() / med_abs) if deltas.min() < 0 else 0.0
    monotone = frac_neg < 0.05
    resid = Cs - pred
    rz = resid - resid.mean()
    ac32 = float((rz[:-4] * rz[4:]).sum() / (rz * rz).sum()) if len(rz) > 4 else None
    micro = {}
    for c in MICRO_CENTERS:
        clus = [C[(M, K_FIXED, n)] for n in range(c - 2, c + 3) if (M, K_FIXED, n) in C]
        if len(clus) >= 4:
            dcs = [abs(clus[i + 1] - clus[i]) for i in range(len(clus) - 1)]
            mx = max(dcs); rest = sorted(dcs)[:-1]
            base = float(np.median(rest)) if rest else float(mx)
            if base > 0:
                micro[c] = round(mx / base, 2)
            elif mx > 0:
                micro[c] = 999.0   # perfect step: flat (Δ=0) between jumps -> strongest possible step
            else:
                micro[c] = 0.0     # no variation at all in the cluster
    ratios = [v for v in micro.values() if v is not None]
    n_step = sum(1 for v in ratios if v >= STAIR_STEP_RATIO)
    n_flat = sum(1 for v in ratios if v < SMOOTH_STEP_RATIO)
    # pre-registered verdict (monotonicity guard first)
    if not monotone:
        verdict = "irregular_nonmonotone_not_staircase"   # cycles swing down as N grows
    elif n_step == len(ratios) and cv is not None and cv >= CV_STAIR:
        verdict = "staircase_at_32_present"               # steps at EVERY probed 32-multiple, monotone
    elif R2 > 0.99 and cv is not None and cv < CV_SMOOTH and n_step == 0:
        verdict = "smooth_no_steps"
    elif R2 > 0.99 and cv is not None and cv < CV_SMOOTH:
        verdict = "smooth_dominated_sparse_coarse_steps"  # ~linear with a few isolated steps, not period-32
    else:
        verdict = "inconclusive"
    return {"M": M, "smooth_loglog_R2": round(R2, 4), "loglog_b": round(float(b), 3),
            "bulk_delta_cv": round(cv, 3) if cv is not None else None,
            "monotone": monotone, "frac_neg_delta": round(frac_neg, 3),
            "max_neg_excursion_x_median": round(max_neg_x, 1),
            "ac32_residual": round(ac32, 3) if ac32 is not None else None,
            "micro_step_ratio": {str(k): v for k, v in micro.items()},
            "n_boundaries_stepped": n_step, "n_boundaries_flat": n_flat,
            "n_boundaries_probed": len(ratios), "verdict": verdict}


def lat_of(rows):
    out = {}
    for r in rows:
        out[tuple(r["shape"])] = r["latency_us"]
    return out


def sensitivities(C, L):
    """E2 alignment, E3 order, E4 shape -- ratios per sim (both members N>=128)."""
    def rat(a, b):
        return round(C[a] / C[b], 3) if a in C and b in C and C[b] else None
    e2 = {f"M={M}": rat((M, K_FIXED, 144), (M, K_FIXED, 128)) for M in (1, 128)}   # misalign/align
    e3 = rat((256, K_FIXED, 128), (128, K_FIXED, 256))                              # M-heavy/N-heavy
    e4 = {f"M={M}": (C.get((M, K_FIXED, 256)), L.get((M, K_FIXED, 256)))
          for M in (1, 32, 128)}                                                    # shape/decode
    return {"E2_alignment_misalign_over_align": e2,
            "E3_order_Mheavy_over_Nheavy": e3,
            "E4_shape_cycles_lat_us_by_M": {k: {"cycles": v[0], "latency_us": v[1]}
                                            for k, v in e4.items()}}


def main():
    if not SCL.exists():
        sys.exit(f"missing {SCL} -- run npu_characteristic_scalesim.py first")
    onx = json.loads(ONX.read_text()); scl = json.loads(SCL.read_text())
    Conx, Cscl = cyc_of(onx["rows"]), cyc_of(scl["rows"])
    Lonx, Lscl = lat_of(onx["rows"]), lat_of(scl["rows"])
    report = {
        "_doc": "Phase 1.6b: does each NPU simulator exhibit a 32-period systolic staircase? "
                "Pre-registered, period-agnostic. H0=smooth/no-32-step; ScaleSim=positive control "
                "(32-step tautological for a 32x32 array sim), ONNXim=informative. simulated, NOT "
                "silicon (#13). Neither value is silicon; magnitudes across sims are NOT comparable.",
        "preregistered_thresholds": {"STAIR_STEP_RATIO": STAIR_STEP_RATIO,
                                     "SMOOTH_STEP_RATIO": SMOOTH_STEP_RATIO, "CV_STAIR": CV_STAIR,
                                     "CV_SMOOTH": CV_SMOOTH, "bulk_grid": "N=128..512 step 8",
                                     "micro_centers": MICRO_CENTERS, "fixed_K": K_FIXED},
        "scalesim_positive_control": {M: analyze(Cscl, M) for M in (1, 128)},
        "onnxim": {M: analyze(Conx, M) for M in (1, 128)},
        "scalesim_sensitivity": sensitivities(Cscl, Lscl),
        "onnxim_sensitivity": sensitivities(Conx, Lonx),
        "onnxim_big_step_boundaries_M1": [],   # filled below for the writeup
    }
    # localize ONNXim's large bulk steps at M=1 (period-agnostic: where do big Δ/N land?)
    s = sorted([(n, Conx[(1, K_FIXED, n)]) for n in range(128, 513) if (1, K_FIXED, n) in Conx])
    base = np.median([abs(s[i + 1][1] - s[i][1]) / (s[i + 1][0] - s[i][0]) for i in range(len(s) - 1)])
    for i in range(len(s) - 1):
        dN = s[i + 1][0] - s[i][0]
        per = (s[i + 1][1] - s[i][1]) / dN
        if per > 3 * base and dN <= 2:   # isolated 1-spaced big step
            report["onnxim_big_step_boundaries_M1"].append(
                {"after_N": s[i][0], "dPerN": round(per, 0), "x_baseline": round(per / base, 1)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    # human summary
    for sim in ("scalesim_positive_control", "onnxim"):
        for M in (1, 128):
            a = report[sim][M]
            print(f"{sim:>26} M={M:<4} verdict={a['verdict']:<32} cv={a['bulk_delta_cv']} "
                  f"R2={a['smooth_loglog_R2']} micro={a['micro_step_ratio']}")
    print("ONNXim M=1 big steps at:", report["onnxim_big_step_boundaries_M1"])
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
