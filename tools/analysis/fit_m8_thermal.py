"""Phase 1.7 — fit the M8 lumped-RC thermal model + the perf-vs-temperature result.

Reads the on-board heat-campaign capture (sustained 4-core 2048x2048 matmul bursts, each a clean
(t, dev_fps, max-core temp_C) point from axrunmodel) and emits:
  - validation/reports/phase1.7/thermal.json   (gate numbers the report renders)
  - simulator/models/params/m8_thermal.json    (RC params for the Phase-2 ThermalModel)

RC heating model (single lumped max-core temperature): T(t) = T_inf - (T_inf - T0)*exp(-t/tau).
Perf-vs-temp: linear fit dev_fps ~ temp_C; compared against the constant-temperature (plateau)
dev_fps noise floor -> "throttle / no-throttle within the achievable range" verdict.

Run: ./.venv/bin/python tools/analysis/fit_m8_thermal.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "measurements/metis_card/thermal_heat_20260612.json"
REP = ROOT / "validation/reports/phase1.7/thermal.json"
PARAM = ROOT / "simulator/models/params/m8_thermal.json"
THRESH = {"pvt_warning": 95, "hw_throttle": 105, "freq_downscale": 110, "sw_throttle": 200}


def main():
    d = json.loads(SRC.read_text())
    b = [x for x in d["bursts"] if x.get("temp_C") and x.get("dev_fps")]
    t = np.array([x["t_elapsed"] for x in b], float)
    T = np.array([x["temp_C"] for x in b], float)
    F = np.array([x["dev_fps"] for x in b], float)
    cores, M = d["meta"]["cores"], d["meta"]["M"]
    T0, Tinf_obs = T[0], T.max()

    # --- RC heating fit: T(t) = Tinf - (Tinf-T0)*exp(-t/tau) ; numpy grid search (no scipy) ---
    def rc(tt, Tinf, tau):
        return Tinf - (Tinf - T0) * np.exp(-tt / tau)
    best = None
    for Tinf_g in np.arange(Tinf_obs, Tinf_obs + 2.01, 0.1):
        for tau_g in np.arange(15, 300, 1):
            rmse = np.sqrt(np.mean((T - rc(t, Tinf_g, tau_g)) ** 2))
            if best is None or rmse < best[0]:
                best = (rmse, Tinf_g, tau_g)
    rc_rmse, Tinf, tau = float(best[0]), float(best[1]), float(best[2])

    # --- perf-vs-temp: slope vs the constant-temperature (plateau) noise floor ---
    slope, icpt = np.polyfit(T, F, 1)
    plateau = F[T == T.max()]
    noise_std = float(np.std(plateau, ddof=1))
    fps_cov_pct = float(100 * np.std(F, ddof=1) / np.mean(F))
    flat = abs(slope) < 2 * noise_std            # |slope per 1C| within 2-sigma of constant-temp noise

    rep = {
        "module": "m8_thermal", "phase": "1.7", "platform": "metis_card",
        "load": {"kind": "synthetic 1x1-conv==2048x2048 matmul (non-LLM)",
                 "cores": cores, "M": M, "tool": "axrunmodel --seconds --aipu-cores",
                 "n_bursts": len(b), "sustained_s": float(t[-1])},
        "temperature": {"source": "axrunmodel temp:<C>C (max core)", "resolution_C": 1,
                        "start_C": float(T0), "plateau_obs_C": float(Tinf_obs),
                        "span_C": float(Tinf_obs - T0), "host_load_max": max(x["host_load"] for x in b)},
        "rc_fit": {"model": "T(t)=Tinf-(Tinf-T0)*exp(-t/tau)", "T0_C": float(T0),
                   "Tinf_C": round(float(Tinf), 1), "tau_s": round(float(tau), 1),
                   "rmse_C": round(rc_rmse, 2),
                   "note": "1C temp resolution + small span (board runs cool) -> tau is approximate"},
        "perf_vs_temp": {"fps_mean": round(float(F.mean()), 1), "fps_cov_pct": round(fps_cov_pct, 3),
                         "slope_fps_per_C": round(float(slope), 2),
                         "noise_std_fps": round(noise_std, 1),
                         "throttle_observed": (not flat),
                         "verdict": ("flat: |slope| within constant-temp noise -> no throttle in "
                                     "the achievable range" if flat else "resolvable slope")},
        "throttle_thresholds_C": THRESH,
        "throttle_reached": False,
        "headroom_to_downscale_C": round(THRESH["freq_downscale"] - float(Tinf_obs), 1),
        "sanity": {"temp_positive": bool((T > 0).all()),
                   "monotonic_nondecreasing": bool((np.diff(T) >= 0).all()),
                   "below_pvt_warning_95C": bool(Tinf_obs < 95)},
        "honesty": ("no power telemetry -> no absolute R_th(°C/W); single lumped max-core temp (5-sensor "
                    "slog intermittent); cooldown not captured (collector idle); perf-vs-temp only over the "
                    "achievable 41-44°C envelope, 110°C downscale threshold never reached; shared die "
                    "(verified effectively single-user at capture, host_load logged)."),
    }
    REP.parent.mkdir(parents=True, exist_ok=True)
    REP.write_text(json.dumps(rep, indent=1))

    param = {"_doc": "M8 lumped-RC thermal (Phase 1.7, Metis Card, max-core temp). "
                     "dT/dt = ((Tinf-T_amb)*duty - (T-T_amb))/tau ; open-loop post-hoc layer (v1, no "
                     "throttle feedback). Relative: no power telemetry -> no absolute R_th(°C/W).",
             "T_amb_C": float(T0), "Tinf_full4core_C": round(float(Tinf), 1),
             "tau_s": round(float(tau), 1), "cores": cores, "kn": "2048x2048", "M": M,
             "throttle_thresholds_C": THRESH, "perf_temp_independent": bool(flat)}
    PARAM.write_text(json.dumps(param, indent=1))
    print(f"wrote {REP.relative_to(ROOT)} + {PARAM.relative_to(ROOT)}")
    print(f"  Tinf={Tinf:.1f}C tau={tau:.0f}s rmse={rc_rmse:.2f}C | perf slope={slope:.1f} "
          f"vs noise {noise_std:.1f} -> {'FLAT/no-throttle' if flat else 'resolvable'}")


if __name__ == "__main__":
    main()
