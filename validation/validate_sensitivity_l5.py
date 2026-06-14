"""Sensitivity / conclusion-robustness (Phase 2.3, ADR-0006 L5) — energy as a BAND, not a point.

Split by the path on which each knob is actually TWO-SIDED (resources.py:35
aggregate = min(k*eff_BW, knee*icn); the AllCim serial path is k=1, so knee/icn only de-rate
BELOW eff and are inert on the +20% side — sweeping them there would be half-by-construction):

 (a) AllCim serial tok/s sensitivity = +/-20% on the eff_BW ANCHOR 24.2 (NOT on bw_efficiency
     directly). eff = peak(34.1) * bw_efficiency, so set bw_efficiency = (24.2 * {0.8,1.0,1.2})/34.1
     -> memory_eff_BW_GBs endpoints 19.36 / 24.2 / 29.04. This is genuinely two-sided (the default
     knee tracks eff). The CIM-centric conclusion (decode is BANDWIDTH-BOUND -> tok/s rises with
     eff_BW; model-size ordering 1B>3B>8B holds) must be invariant across the band.
 (b) Contention-path knee/icn sensitivity (k>1, SIMULATED — no concurrent silicon, #52): +/-20%
     of knee_GBs / interconnect_efficiency where they ARE two-sided, on the SharedBandwidth
     aggregate. Reported as a band, EXCLUDED from the AllCim serial robustness claim.
 (c) Energy band = +/-20% POST-PROCESS on the runner's returned energy_per_token_J (M7 has no
     config scale hook; this is NOT a runner sweep).

Writes validation/reports/phase2/sensitivity.json
Run:   ./.venv/bin/python validation/validate_sensitivity_l5.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.runtime.config import SimConfig  # noqa: E402
from simulator.runtime.runner import run  # noqa: E402
from simulator.runtime.resources import SharedBandwidth  # noqa: E402
from simulator.specs.loader import load_spec  # noqa: E402

OUT = ROOT / "validation/reports/phase2"
MODELS = ["llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b"]
ANCHOR = 24.2                                     # measured on-card LPDDR4x decode anchor
PEAK = float(load_spec("mem_lpddr4x")["peak_GBs"])   # 34.1
FACTORS = [0.8, 1.0, 1.2]                         # +/-20% on the eff_BW anchor


def _cfg(model, bw_eff):
    return SimConfig.from_dict({
        "workload": {"model": model, "context": 1024},
        "platform": {"topology": "cim_topo_card", "memory_spec": "mem_lpddr4x",
                     "bw_efficiency": bw_eff},
        "scheduler": {"policy": "all_cim"},
        "tunables": {"pipeline": False},
    })


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # (a) AllCim serial eff_BW sensitivity (two-sided)
    serial = {}
    for m in MODELS:
        serial[m] = {}
        for f in FACTORS:
            bw_eff = (ANCHOR * f) / PEAK
            r = run(_cfg(m, bw_eff))
            serial[m][f] = {"eff_BW_GBs": round(r["memory_eff_BW_GBs"], 2),
                            "tok_s": r["tok_s"], "energy_per_token_J": r["energy_per_token_J"]}

    # endpoints land exactly on +/-20% of 24.2
    eff_endpoints_ok = all(
        abs(serial[m][0.8]["eff_BW_GBs"] - 19.36) < 0.05
        and abs(serial[m][1.0]["eff_BW_GBs"] - 24.2) < 0.05
        and abs(serial[m][1.2]["eff_BW_GBs"] - 29.04) < 0.05 for m in MODELS)
    # bw_efficiency is genuinely two-sided: tok/s strictly rises with eff_BW (not inert)
    tok_moves_both_ways = all(serial[m][0.8]["tok_s"] < serial[m][1.0]["tok_s"]
                              < serial[m][1.2]["tok_s"] for m in MODELS)
    # CONCLUSION ROBUSTNESS: bandwidth-bound (tok/s monotone in eff_BW) + model ordering 1B>3B>8B at every point
    ordering_ok = all(serial["llama-3.2-1b"][f]["tok_s"] > serial["llama-3.2-3b"][f]["tok_s"]
                      > serial["llama-3.1-8b"][f]["tok_s"] for f in FACTORS)
    conclusion_robust = bool(tok_moves_both_ways and ordering_ok)

    # (b) contention-path knee/icn band (k>1, SIMULATED, excluded from serial robustness)
    K = 4
    base = SharedBandwidth(ANCHOR, knee_GBs=ANCHOR, interconnect_efficiency=1.0).aggregate_GBs(K)
    knee_band = [round(SharedBandwidth(ANCHOR, knee_GBs=ANCHOR * f).aggregate_GBs(K), 2) for f in (0.8, 1.2)]
    icn_band = [round(SharedBandwidth(ANCHOR, knee_GBs=ANCHOR, interconnect_efficiency=f).aggregate_GBs(K), 2)
                for f in (0.8, 1.2)]
    knee_two_sided = knee_band[0] < base < knee_band[1] and icn_band[0] < base < icn_band[1]

    # (c) energy band = +/-20% post-process on the baseline per-token energy
    energy = {}
    for m in MODELS:
        e = serial[m][1.0]["energy_per_token_J"]
        energy[m] = {"point_J": e, "band_J": [round(e * 0.8, 6), round(e * 1.2, 6)]}

    out = {
        "module": "sensitivity_l5", "phase": "2.3",
        "honesty": "AllCim serial robustness uses ONLY bw_efficiency (two-sided on the k=1 path); "
                   "knee/icn are reported on the k>1 contention path (SIMULATED, #52) and EXCLUDED "
                   "from the serial claim; energy is a +/-20% band, never a point.",
        "anchor_eff_BW_GBs": ANCHOR, "peak_BW_GBs": PEAK,
        "serial_eff_BW_sweep": serial,
        "eff_BW_endpoints_19_36__24_2__29_04": bool(eff_endpoints_ok),
        "bw_efficiency_two_sided_tok_moves": bool(tok_moves_both_ways),
        "model_ordering_1b_gt_3b_gt_8b_all_points": bool(ordering_ok),
        "conclusion_robust": conclusion_robust,
        "conclusion": "decode is BANDWIDTH-BOUND: tok/s rises monotonically with eff_BW across the "
                      "+/-20% band and the model-size ordering holds — the memory-wall term dominates, "
                      "robust to +/-20% BW uncertainty.",
        "contention_path_band_SIMULATED": {
            "k": K, "baseline_aggregate_GBs": round(base, 2),
            "knee_pm20_aggregate_GBs": knee_band, "icn_pm20_aggregate_GBs": icn_band,
            "two_sided": bool(knee_two_sided),
            "note": "k>1 contention knee/icn (SIMULATED, no concurrent silicon #52); NOT part of the "
                    "AllCim serial robustness claim.",
        },
        "energy_band_pm20": energy,
        "pass_all": conclusion_robust and bool(eff_endpoints_ok),
    }
    (OUT / "sensitivity.json").write_text(json.dumps(out, indent=1))

    print("sensitivity (AllCim serial, +/-20% on eff_BW anchor 24.2):")
    for m in MODELS:
        print(f"  {m}: " + " ".join(f"{serial[m][f]['eff_BW_GBs']}GB/s->{serial[m][f]['tok_s']:.2f}t/s"
                                    for f in FACTORS))
    print(f"  eff endpoints 19.36/24.2/29.04: {eff_endpoints_ok}; tok two-sided: {tok_moves_both_ways}")
    print(f"  conclusion_robust (bandwidth-bound + ordering): {conclusion_robust}")
    print(f"  contention band (k=4, SIMULATED): knee+-20%={knee_band} icn+-20%={icn_band} (excluded from serial)")
    return 0 if out["pass_all"] else 1


if __name__ == "__main__":
    sys.exit(main())
