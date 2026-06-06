"""Phase 1 — fit the M1 CIM tile timing model + validate (decode-calibrated).

Reads  measurements/aetina/metis_alpha_matmul.json + _raw.json (native flags)
Writes simulator/models/params/m1_cim.json   (2D G_eff params, n_cores=4)
       validation/reports/phase1.1/m1.json             (native fit errors + native/generated split)

Architecture (papers/metis-silicon/metis-aipu-isscc2024.md): quad-core, 512x512 per core.
G_eff(N,K) (GOP/s, INT8) rises with output width N AND input depth K -> fit a 2D closed form
on NATIVE single-tile points (K*N <= 4.19M). Above that = unvalidated extrapolation.

Run: ./.venv/bin/python tools/analysis/fit_m1_cim.py
"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from simulator.models.m1_cim_tile import CimTileModel  # noqa: E402

AET = ROOT / "measurements/aetina"
PARAMS = ROOT / "simulator/models/params/m1_cim.json"
REPORT = ROOT / "validation/reports/phase1.1/m1.json"
NATIVE_MAX_KN = 2048 * 2048   # 4.19M params = largest natively measured single tile


def rel(p, m):
    return abs(p - m) / m


# projection-GEMV groups (weight-stationary matmul). attn_floor is the attention conv-proxy
# (fixed-overhead-dominated, K=head_dim) -> a different regime, excluded from the throughput fit.
PROJ_GROUPS = {"staircase64", "staircase_off64", "proj_decode", "aspect", "l2_ddr"}


def native_points(single_tile_only=True):
    """Native (M=1) projection throughput points: (N, K, gops, dev_lat_us, group).
    single_tile_only -> N <= one combined tile (4*512=2048); the one native multi-tile point
    (N=4096,K=1024) is returned only when single_tile_only=False (for tiling validation)."""
    raw = json.loads((AET / "metis_alpha_matmul_raw.json").read_text())
    seen = {}
    for r in raw.values():
        if r.get("tiled_extrapolated") or r.get("M") != 1 or r.get("group") not in PROJ_GROUPS:
            continue
        if not (r.get("dev_gflops") and r.get("K") and r.get("N") and r.get("dev_lat_us")):
            continue
        if r["K"] * r["N"] > NATIVE_MAX_KN:
            continue
        if single_tile_only and r["N"] > 2048:
            continue
        seen[(r["N"], r["K"])] = (r["dev_gflops"], r["dev_lat_us"], r.get("group"))
    return [(N, K, g, lat, grp) for (N, K), (g, lat, grp) in sorted(seen.items())]


def fit_2d(pts):
    """G = Gmax * N/(N+Na) * K/(K+Kb), grid search for lowest median rel error (GOP/s)."""
    N = np.array([p[0] for p in pts], float)
    K = np.array([p[1] for p in pts], float)
    G = np.array([p[2] for p in pts], float)
    best = None
    for Gmax in np.linspace(230, 360, 80):
        for Na in np.linspace(150, 1400, 80):
            for Kb in np.linspace(50, 3500, 80):
                pred = Gmax * (N / (N + Na)) * (K / (K + Kb))
                e = float(np.median(np.abs(pred - G) / G))
                if best is None or e < best[0]:
                    best = (e, Gmax, Na, Kb)
    return best[1], best[2], best[3]


def main():
    pts = native_points()
    Gmax, Na, Kb = fit_2d(pts)
    params = {
        "_doc": "M1 CIM tile (Phase 1). Quad-core, 512x512/core (ISSCC 2024). G_eff(N,K) 2D "
                "throughput (GOP/s, INT8) fit on native single-tile pts (K*N<=4.19M). n_cores free.",
        "n_cores": 4, "core_width": 512,
        "G_eff_Gmax_gops": round(float(Gmax), 2), "G_eff_Na": round(float(Na), 1),
        "G_eff_Kb": round(float(Kb), 1),
        "native_max_kn": NATIVE_MAX_KN, "alloc_envelope_params": 6_000_000,
        "native_throughput_points": [{"N": N, "K": K, "gops": round(g, 1)} for N, K, g, _, _ in pts],
    }
    # Preserve prefill / Card-revalidation keys that fit_cim_prefill.py adds later: a standalone decode
    # re-fit must NOT silently drop the prefill model (issue #28). Merge, don't clobber.
    if PARAMS.exists():
        old = json.loads(PARAMS.read_text())
        for k, v in old.items():
            if k.startswith("prefill_") or k in ("decode_card_revalidation", "_prefill_doc"):
                params[k] = v
    PARAMS.write_text(json.dumps(params, indent=1))
    m = CimTileModel(params)

    # --- throughput fit error on NATIVE points (the honest gate) ---
    g_err = [rel(m.g_eff(N, K), g) for N, K, g, _, _ in pts]
    # --- latency error on NATIVE points (meas vs pred) ---
    lat_rows, lat_err = [], []
    for N, K, g, lat, grp in pts:
        pred = m.dev_lat_us(1, K, N)
        lat_rows.append({"N": N, "K": K, "meas_us": round(lat, 1), "pred_us": round(pred, 1),
                         "rel_err": round(rel(pred, lat), 3), "group": grp})
        lat_err.append(rel(pred, lat))

    # --- tiling validation on the ONE native multi-tile point (N=4096,K=1024) ---
    multi = [p for p in native_points(single_tile_only=False) if p[0] > 2048]
    tiling_check = []
    for N, K, g, lat, grp in multi:
        pred = m.dev_lat_us(1, K, N)
        tiling_check.append({"N": N, "K": K, "meas_us": round(lat, 1), "pred_us": round(pred, 1),
                             "rel_err": round(rel(pred, lat), 3),
                             "note": "the only native multi-tile point; tests continued-rise tiling"})

    # --- native vs generated split of proj_decode (the A1.4 honesty fix) ---
    raw = json.loads((AET / "metis_alpha_matmul_raw.json").read_text())
    proj = []
    for r in raw.values():
        if r.get("group") == "proj_decode" and r.get("dev_lat_us"):
            proj.append({"model": r["model"], "family": r["family"], "K": r["K"], "N": r["N"],
                         "meas_us": round(r["dev_lat_us"], 1),
                         "native": not r.get("tiled_extrapolated")})
    n_native = sum(1 for x in proj if x["native"])

    gate = {"set": "native single-tile throughput G_eff(N,K), %d pts (all natively measured)" % len(pts),
            "median": round(float(statistics.median(g_err)), 3),
            "p95": round(float(np.percentile(g_err, 95)), 3),
            "max": round(float(max(g_err)), 3),
            "pass_median_le_0.10": bool(statistics.median(g_err) <= 0.10),
            "pass_p95_le_0.20": bool(float(np.percentile(g_err, 95)) <= 0.20)}

    report = {
        "module": "m1_cim_tile",
        "architecture": "quad-core, 512x512 INT8 D-IMC per core; n_cores free (=4 Metis); GOP/s not FLOP/s",
        "equation": "dev_lat = sum over N-tiles(width<=n_cores*512) of 2*M*K*n/G_eff(n,K); "
                    "G_eff(N,K)=Gmax*N/(N+Na)*K/(K+Kb)",
        "params": {"Gmax_gops": params["G_eff_Gmax_gops"], "Na": params["G_eff_Na"],
                   "Kb": params["G_eff_Kb"], "n_cores": 4, "core_width": 512,
                   "native_max_kn_M": round(NATIVE_MAX_KN / 1e6, 2)},
        "throughput_fit_gate_native": gate,
        "native_latency_meas_vs_pred": lat_rows,
        "tiling_validation_native_multitile": tiling_check,
        "proj_decode_native_vs_generated": {
            "n_native": n_native, "n_generated": len(proj) - n_native,
            "note": "only the native rows are measurements; generated rows are this model's "
                    "tile-sum output (no native measurement exists, K*N > 4.19M envelope).",
            "rows": proj},
        "extrapolation_note": "K*N > 4.19M (multi-tile) is UNVALIDATED: no native data above one "
            "tile; latency uses a continued-rising tile-sum (partial last tile adds its own size, "
            "not a full tile). Board offline -> cannot measure (issues #2/#11/#17).",
        "regime_note": "decode/memory-bound only (~%d GOP/s eff, ~0.1%% of the 209600 GOP/s "
            "compute peak); compute-ceiling not modeled (issue #16)." % round(max(p[2] for p in pts)),
        "unit_note": "throughput is INT8 GOP/s (issue #18); the raw JSON field name 'dev_gflops' is legacy.",
    }
    REPORT.write_text(json.dumps(report, indent=1))

    print(f"M1: 2D G_eff(N,K) GOP/s: Gmax={params['G_eff_Gmax_gops']} Na={params['G_eff_Na']} "
          f"Kb={params['G_eff_Kb']} | native pts={len(pts)}")
    print(f"  throughput fit (native): median={gate['median']} p95={gate['p95']} max={gate['max']} "
          f"PASS={gate['pass_median_le_0.10'] and gate['pass_p95_le_0.20']}")
    print(f"  native latency rel-err: median={statistics.median(lat_err):.3f} max={max(lat_err):.3f}")
    print(f"  proj_decode: {n_native} native, {len(proj)-n_native} generated (tile-sum, no measurement)")


if __name__ == "__main__":
    main()
