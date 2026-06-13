"""Contention-model check (Phase 2.1) — SIMULATED, NOT a numeric gate.

The shared-bandwidth knee has NO concurrent-unit silicon (Aetina offline, issue
#52), so this is a SHAPE check, not a reproduced silicon value (m3.yaml:
contention_trend, tag simulated). Two anchors:
 1. the SharedBandwidth aggregate must rise with concurrent streams then saturate
    at the knee (the memory-wall shape), and collapse to a linear sum with
    contention off;
 2. the only real on-silicon contention signal is the Metis-Card 4c/1c decode
    scaling (measured 1.130/1.096/1.081, near-flat -> "more cores barely help");
    the knee is set so the model reproduces that ~1.1x near-flat level.
A constant knee does NOT capture the mild size-dependence of the ratio
(1.13->1.08) — reported honestly, not gated.

Ramulator2 multi-stream cross-check is a FUTURE item (single-stream LPDDR5 eff
only ships today); see issue #52.

Reads  measurements/metis_card/vendor_llm_int8.json
Writes validation/reports/phase2/contention.json
Run:   ./.venv/bin/python validation/validate_contention.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.runtime.resources import SharedBandwidth  # noqa: E402

MC = ROOT / "measurements/metis_card"
OUT = ROOT / "validation/reports/phase2"
EFF_BW = 24.2                                  # measured on-card LPDDR4x single-stream anchor


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vendor = json.loads((MC / "vendor_llm_int8.json").read_text())
    meas_ratio = {m: round(vendor[f"{m}/4c"]["tok_s_median"] / vendor[f"{m}/1c"]["tok_s_median"], 3)
                  for m in ("llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b")}
    mean_ratio = sum(meas_ratio.values()) / len(meas_ratio)

    # knee set so 4-stream aggregate reproduces the ~1.1x near-flat 4c/1c level
    knee = EFF_BW * mean_ratio
    bw = SharedBandwidth(EFF_BW, knee_GBs=knee)
    sweep = {k: round(bw.aggregate_GBs(k), 2) for k in range(1, 9)}
    sweep_off = {k: round(bw.aggregate_GBs(k, contention=False), 2) for k in range(1, 9)}
    model_ratio_4c = round(bw.aggregate_GBs(4) / bw.aggregate_GBs(1), 3)

    # shape assertions (trend, not a measured gate)
    rising_then_flat = sweep[1] < sweep[2] and sweep[2] == sweep[8]
    off_linear = abs(sweep_off[4] - 4 * EFF_BW) < 1e-6

    out = {
        "module": "contention", "phase": "2.1", "tag": "simulated",
        "honesty": "SIMULATED knee — no concurrent-unit silicon (Aetina offline, issue #52). "
                   "Shape check, not a reproduced silicon value; NOT a numeric gate.",
        "eff_BW_GBs_single_stream": EFF_BW,
        "knee_GBs": round(knee, 2),
        "aggregate_sweep_GBs": sweep,
        "aggregate_sweep_contention_off_GBs": sweep_off,
        "rising_then_saturating": bool(rising_then_flat),
        "contention_off_is_linear": bool(off_linear),
        "card_4c_1c_measured": meas_ratio,
        "card_4c_1c_mean": round(mean_ratio, 3),
        "model_4c_1c_ratio": model_ratio_4c,
        "size_dependence_note": "constant knee gives a constant 4c/1c ratio; the measured mild "
                                "size-dependence (1.130->1.081) is NOT captured by one knee — noted, simulated.",
        "ramulator2_multistream": "FUTURE cross-check (only single-stream LPDDR5 eff ships today); issue #52.",
        "trend_consistent": bool(rising_then_flat and off_linear),
    }
    (OUT / "contention.json").write_text(json.dumps(out, indent=1))
    print(f"contention (SIMULATED, no gate): eff_BW={EFF_BW} knee={knee:.1f} GB/s")
    print(f"  aggregate sweep GB/s: {sweep}")
    print(f"  model 4c/1c={model_ratio_4c} vs measured {meas_ratio} (mean {mean_ratio:.3f})")
    print(f"  rising-then-saturating={rising_then_flat}  contention-off-linear={off_linear}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
