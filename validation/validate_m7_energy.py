"""Phase 1 — build M7 energy params (spec, ADR-0005) + validate (no silicon ground truth).

Writes simulator/models/params/m7_energy.json, validation/reports/m7.json

No power telemetry -> validate by sanity + +/-20% coefficient sensitivity:
  (1) energy positive; (2) monotonic with activity; (3) order-of-magnitude vs a spec-derived
  bound; (4) the qualitative conclusion (decode is memory-dominated, CIM compute is cheap)
  does NOT flip under +/-20% on any coefficient.

Run: ./.venv/bin/python validation/validate_m7_energy.py
"""
import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.models.m7_energy import EnergyModel  # noqa: E402

# spec constants (ADR-0005)
PARAMS = {
    "_doc": "M7 spec-based energy (ADR-0005). Estimated, not measured. +/-20% sensitivity.",
    "cim_tops_w": 15.0,            # vendor Metis INT8
    "lpddr5_pj_per_bit": 4.0,      # JEDEC LPDDR5 access (assumption; cited)
    "pcie_pj_per_bit": 5.0,        # PCIe spec (assumption)
    "a76_core_w": 0.75,            # ARM A76 datasheet active (per core)
    "cpu_cores": 4,
}

# 8B Llama decode per-token activity (closed form, INT8 weight bytes streamed once/token)
H, F, kv, V, L = 4096, 14336, 1024, 128256, 32
WEIGHT_BYTES = (H * H + 2 * (H * kv) + H * H + 2 * (H * F) + F * H) * L + V * H  # bytes
WEIGHT_FLOPS = 2 * WEIGHT_BYTES               # 1 INT8 param -> 1 MAC -> 2 flops
CPU_SUPPORT_US = 5000.0                        # ~per-token A76 support (rmsnorm/rope/swiglu/softmax/argmax x L), fp16 upper bound


def per_token_breakdown(m):
    return {
        "cim_proj_mJ": m.cim_J(WEIGHT_FLOPS) * 1e3,
        "dram_stream_mJ": m.dram_J(WEIGHT_BYTES) * 1e3,
        "cpu_support_mJ": m.cpu_J(CPU_SUPPORT_US) * 1e3,
    }


def main():
    (ROOT / "simulator/models/params/m7_energy.json").write_text(json.dumps(PARAMS, indent=1))
    m = EnergyModel(PARAMS)

    base = per_token_breakdown(m)
    total = sum(base.values())

    # (2) monotonic with activity
    mono = (m.cim_J(2e9) < m.cim_J(4e9) and m.dram_J(1e6) < m.dram_J(2e6)
            and m.cpu_J(100) < m.cpu_J(200))

    # (3) INDEPENDENT order-of-magnitude check: spec per-token energy / measured decode time
    #     must imply a plausible SoC average power (not a tautology vs the same coefficient).
    DECODE_TOK_S_8B = 2.70                         # vendor measured (1c)
    decode_time_s = 1.0 / DECODE_TOK_S_8B
    avg_power_W = (total / 1e3) / decode_time_s     # total mJ -> J, over the decode step
    om_ok = 0.1 <= avg_power_W <= 20.0              # plausible mobile-SoC envelope

    # (4) +/-20% sensitivity: does "memory dominates" survive every corner?
    flips = []
    for sc in product([0.8, 1.2], repeat=4):
        scale = dict(zip(["cim", "dram", "pcie", "cpu"], sc))
        b = per_token_breakdown(EnergyModel(PARAMS, scale))
        if not (b["dram_stream_mJ"] > b["cim_proj_mJ"]):   # memory should dominate CIM compute
            flips.append(scale)
    no_flip = not flips

    report = {
        "module": "m7_energy",
        "model": "spec-based (ADR-0005): CIM 15 TOPS/W, DRAM/PCIe pJ/bit, CPU A76 W*cores*t",
        "params": {k: v for k, v in PARAMS.items() if not k.startswith("_")},
        "per_token_8b_decode_mJ": {k: round(v, 3) for k, v in base.items()},
        "per_token_total_mJ": round(total, 3),
        "dominant_term": max(base, key=base.get),
        "sanity": {
            "all_positive": all(v > 0 for v in base.values()),
            "monotonic_with_activity": mono,
            "implied_avg_power_W_plausible": om_ok,
            "implied_avg_power_W": round(avg_power_W, 3),
            "memory_dominates_robust_to_pm20pct": no_flip},
        "sensitivity_pm20pct": {"corners_tested": 16, "conclusion_flips": len(flips)},
        "limitation": "energy ESTIMATED not measured (no telemetry, ADR-0005); conclusions "
                      "robust to +/-20%. CPU support time is a coarse per-token estimate.",
    }
    (ROOT / "validation/reports/m7.json").write_text(json.dumps(report, indent=1))
    s = report["sanity"]
    bool_checks = ["all_positive", "monotonic_with_activity", "implied_avg_power_W_plausible",
                   "memory_dominates_robust_to_pm20pct"]
    print(f"M7: per-token 8B decode = {report['per_token_8b_decode_mJ']} mJ "
          f"(total {report['per_token_total_mJ']}); dominant={report['dominant_term']}")
    print(f"  sanity: positive={s['all_positive']} monotonic={s['monotonic_with_activity']} "
          f"avg_power={s['implied_avg_power_W']}W(plausible={s['implied_avg_power_W_plausible']}) "
          f"no_flip_pm20={s['memory_dominates_robust_to_pm20pct']}")
    print(f"  PASS={all(s[k] for k in bool_checks)}")


if __name__ == "__main__":
    main()
