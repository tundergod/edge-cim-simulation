"""Phase 1 — fit the M1 CIM tile timing model + validate (decode-calibrated).

Reads  measurements/aetina/metis_alpha_matmul.json (by_group)
Writes simulator/models/params/m1_cim.json   (equation params + G_eff lookup)
       validation/reports/m1.json             (held-out errors + narrow-N residual)

The M1 equation has NO per-model free parameters: its constants (effective-throughput
curve G_eff(N), canonical tile latency T_tile, 2048-tile size) are CIM physics, fit on
the 8B channel-64 staircase + aspect microbenchmark. Validation = does the single
equation predict every proj_decode/lmhead shape (1b/3b/8b/qwen)?  The 8b+qwen
projections are held out (not used to set any constant). The wide-K narrow-N kv-proj
is reported as a separate residual (plan verify e).

Run: ./.venv/bin/python tools/analysis/fit_m1_cim.py
"""
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from simulator.models.m1_cim_tile import CimTileModel  # noqa: E402

AET = ROOT / "measurements/aetina"
PARAMS = ROOT / "simulator/models/params/m1_cim.json"
REPORT = ROOT / "validation/reports/m1.json"
TILE = 2048


def _co(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not serializable: {type(o)}")


def rel(pred, meas):
    return abs(pred - meas) / meas


def fit_hill(Ns, Gs):
    """G = Gmax*N/(N+Nhalf)  via linear fit of 1/G = 1/Gmax + (Nhalf/Gmax)/N."""
    x = 1.0 / np.array(Ns, float)
    y = 1.0 / np.array(Gs, float)
    b, a = np.polyfit(x, y, 1)        # y = a + b*x
    Gmax = 1.0 / a
    Nhalf = b * Gmax
    return Gmax, Nhalf


def main():
    d = json.loads((AET / "metis_alpha_matmul.json").read_text())
    bg = d["by_group"]

    # --- G_eff(N) from the 8B channel-64 staircase (single-tile points, N<=2048) ---
    stair = [(r["N"], r["dev_gflops"], r["dev_lat_us"]) for r in bg["staircase64"]
             if r.get("tiles", 1) == 1]
    stair.sort()
    Ns = [n for n, _, _ in stair]
    Gs = [g for _, g, _ in stair]
    lookup = {str(n): g for n, g, _ in stair}
    Gmax, Nhalf = fit_hill(Ns, Gs)
    # Hill fit error on the staircase = the INDEPENDENT M1 compute-fit metric
    # (the multi-tile proj_decode values are themselves canonical_tile*n_tiles
    #  extrapolations, so reproducing them is not an independent check).
    hill_err = [rel(Gmax * n / (n + Nhalf), g) for n, g in zip(Ns, Gs)]
    use_lookup = statistics.median(hill_err) > 0.10
    compute_fit_gate = {
        "set": "8B channel-64 staircase G_eff(N), 8 pts (independent fit target)",
        "median": round(float(statistics.median(hill_err)), 3),
        "p95": round(float(np.percentile(hill_err, 95)), 3),
        "max": round(float(max(hill_err)), 3),
        "pass_median_le_0.10": bool(statistics.median(hill_err) <= 0.10),
        "pass_p95_le_0.20": bool(float(np.percentile(hill_err, 95)) <= 0.20)}
    T_tile = next(r["dev_lat_us"] for r in bg["staircase64"] if r["N"] == 2048)

    params = {
        "_doc": "M1 CIM tile timing (Phase 1). G_eff(N) saturating throughput from 8B "
                "channel-64 staircase; T_tile = full 2048x2048 tile dev latency. "
                "Decode-calibrated; prefill unvalidated.",
        "G_eff_Gmax_gflops": round(Gmax, 2),
        "G_eff_Nhalf": round(Nhalf, 1),
        "G_eff_lookup": {k: round(v, 2) for k, v in lookup.items()},
        "use_lookup": use_lookup,
        "G_eff_hill_median_relerr": round(statistics.median(hill_err), 3),
        "crossbar_tile": TILE,
        "canonical_tile_us": round(T_tile, 2),
        "device_envelope_params": 6_000_000,
    }
    PARAMS.write_text(json.dumps(params, indent=1, default=_co))
    m = CimTileModel(params)

    # --- validate against proj_decode (all models; M=1 decode) ---
    wellfilled, narrow = [], []
    for r in bg["proj_decode"]:
        if r.get("dev_lat_us") is None:
            continue
        pred = m.dev_lat_us(r["M"], r["K"], r["N"])
        row = {"model": r["model"], "family": r["family"], "K": r["K"], "N": r["N"],
               "meas_us": round(r["dev_lat_us"], 2), "pred_us": round(pred, 2),
               "rel_err": round(rel(pred, r["dev_lat_us"]), 3)}
        (narrow if r["N"] < TILE else wellfilled).append(row)

    heldout = [x for x in wellfilled if x["model"] in ("llama-3.1-8b", "qwen2.5-7b")]
    errs = [x["rel_err"] for x in heldout]
    gate = {"n": len(errs), "median": round(statistics.median(errs), 3),
            "p95": round(np.percentile(errs, 95), 3), "max": round(max(errs), 3),
            "pass_median_le_0.10": statistics.median(errs) <= 0.10,
            "pass_p95_le_0.20": float(np.percentile(errs, 95)) <= 0.20}

    # staircase held-out: fit Hill on N<=1536, predict N in {2048}
    fit_pts = [(n, g) for n, g in zip(Ns, Gs) if n <= 1536]
    gm2, nh2 = fit_hill([n for n, _ in fit_pts], [g for _, g in fit_pts])
    g_pred_2048 = gm2 * 2048 / (2048 + nh2)
    stair_heldout = {"N": 2048, "G_pred": round(g_pred_2048, 1),
                     "G_meas": round(dict(zip(Ns, Gs))[2048], 1),
                     "rel_err": round(rel(g_pred_2048, dict(zip(Ns, Gs))[2048]), 3)}

    # sanity: monotonic dev_lat in N at fixed K (the staircase); risers at 64-multiples
    lat_by_N = [(r["N"], r["dev_lat_us"]) for r in bg["staircase64"]]
    lat_by_N.sort()
    monotonic = all(b[1] >= a[1] for a, b in zip(lat_by_N, lat_by_N[1:]))
    risers_64 = all(n % 64 == 0 for n, _ in lat_by_N)

    # aspect: equal-MAC spread (descriptive)
    asp = [(r["K"], r["N"], r["dev_gflops"]) for r in bg["aspect"]]
    aspect_spread = round((max(g for *_, g in asp) - min(g for *_, g in asp))
                          / statistics.mean(g for *_, g in asp), 3)

    report = {
        "module": "m1_cim_tile",
        "equation": "dev_lat_us = (N<2048) 2*M*K*N/G_eff(N)  else  M*n_tiles*T_tile",
        "params": {"G_eff": "lookup" if use_lookup else f"Hill Gmax={params['G_eff_Gmax_gflops']},Nhalf={params['G_eff_Nhalf']}",
                   "T_tile_us": params["canonical_tile_us"], "tile": TILE,
                   "G_eff_hill_median_relerr": params["G_eff_hill_median_relerr"],
                   "use_lookup_fallback": use_lookup},
        "compute_fit_gate_G_eff_staircase": compute_fit_gate,
        "tiling_reproduction_note": "multi-tile proj_decode measured dev_lat ARE "
            "canonical_tile*n_tiles extrapolations (by_group flag tiled_extrapolated); the model "
            "reproduces them by construction (errors ~0), so this is NOT an independent "
            "check. Independent M1 validation = G_eff staircase fit (above) + staircase "
            "held-out + native single-tile + narrow-N residual.",
        "tiling_reproduction_8b_qwen_wellfilled": gate,
        "wellfilled_all_models": wellfilled,
        "narrow_kv_residual_verify_e": {
            "note": "wide-K narrow-N underfill; G_eff(N) over-predicts throughput->latency; "
                    "single 8B point, unfittable -> reported, NOT in gate.",
            "rows": narrow},
        "staircase_heldout_N2048": stair_heldout,
        "sanity": {"monotonic_dev_lat_in_N": monotonic, "risers_all_64_multiple": risers_64,
                   "aspect_equal_mac_spread": aspect_spread},
        "nonequation_regions": {
            "lm_head_N_128k_152k": "analytic n_tiles*T_tile, no measurement (canonical N=4096 tile)",
            "prefill_M_ge_512": "device-alloc fail, analytic, unvalidated",
            "qwen_non2048": "predicted on padded tiles (no restore); ~1.24x is a GFLOP/s reporting bias only"},
    }
    REPORT.write_text(json.dumps(report, indent=1, default=_co))

    print(f"M1: G_eff Hill median relerr={params['G_eff_hill_median_relerr']} -> "
          f"use_lookup={use_lookup}; T_tile={params['canonical_tile_us']}us")
    print(f"  held-out (8b+qwen well-filled) n={gate['n']} median={gate['median']} "
          f"p95={gate['p95']} max={gate['max']} "
          f"PASS={gate['pass_median_le_0.10'] and gate['pass_p95_le_0.20']}")
    print(f"  staircase held-out N=2048: rel_err={stair_heldout['rel_err']}")
    print(f"  narrow kv residuals: " +
          ", ".join(f"{r['model'].split('-')[-1]} N{r['N']}={r['rel_err']:+.0%}" for r in narrow))
    print(f"  sanity monotonic={monotonic} risers64={risers_64} aspect_spread={aspect_spread}")


if __name__ == "__main__":
    main()
