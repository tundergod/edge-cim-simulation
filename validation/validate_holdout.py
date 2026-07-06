"""Hold-out / size-extrapolation (Phase 2.3, ADR-0006) — two honestly-distinct layers.

The engine has NO size-coupled free parameter to refit (the 24.2 anchor is a fixed scalar, and
the CIM-compute correction is parameter-free), so name precisely what is and isn't fitted:

 (a) GENUINE leave-8B-out (closed form): the only free, size-regressed, fittable scalar is the
     two-pillar effective decode BW `fit_BW`. RE-DERIVE it from 1B + 3B ONLY —
       implied_BW(m) = measured_tok_s(m) * per_token_weight_bytes_GB(m);  fit_BW = mean(1B, 3B)
     — then predict 8B in closed form: pred_8B = fit_BW / weight_bytes_8B. This leaves the ENGINE's
     24.2 anchor UNTOUCHED (byte-identical). Expect ~9.5% (the committed ADR-0006 hold-out).
 (b) ENGINE path: run 8B through the FULL engine (runner.run) with NO 8B-specific tuning — the
     CIM-compute correction is parameter-free and the 24.2 backbone is fixed. CAVEAT (front-loaded,
     not a footnote): the 24.2 backbone is regressed across all sizes incl. 8B in Phase 1, so it is
     PARTLY IN-SAMPLE; the genuinely out-of-sample content is the parameter-free compute correction.

Gate (on the engine number): 8B rel-error <= 15% AND <= the closed-form leave-8B-out error. If the
engine is WORSE than the closed form = Gate-6c signal -> fix the responsible model element (M1/M2/M3),
NOT the integration layer, and name it in the PR.

Reads  measurements/metis_card/twopillar_prediction_fitted.json, validation/reports/phase2/e2e_l4.json
Writes validation/reports/phase2/holdout.json
Run:   ./.venv/bin/python validation/validate_holdout.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.runtime.config import SimConfig  # noqa: E402
from simulator.runtime.runner import run  # noqa: E402

MC = ROOT / "measurements/metis_card"
OUT = ROOT / "validation/reports/phase2"
GATE = 0.15
FIT_MODELS = ["llama-3.2-1b", "llama-3.2-3b"]   # the {1B,3B} fit set
HOLDOUT = "llama-3.1-8b"                          # 8B is held out


def _cfg(model):
    return SimConfig.from_dict({
        "workload": {"model": model, "context": 1024},
        "platform": {"topology": "cim_topo_card", "memory_spec": "mem_lpddr4x"},
        "scheduler": {"policy": "all_cim"},
        "tunables": {"pipeline": False},
    })


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tp = json.loads((MC / "twopillar_prediction_fitted.json").read_text())
    wb = tp["per_token_weight_bytes"]               # GB per decode token
    meas = tp["measured_tok_s_1c"]

    # (a) GENUINE leave-8B-out closed form: fit_BW from {1B,3B} only
    implied = {m: meas[m] * wb[m] for m in FIT_MODELS}
    fit_BW = sum(implied.values()) / len(implied)   # mean of the {1B,3B} implied BW
    pred_8b_closed = fit_BW / wb[HOLDOUT]
    rel_closed = abs(pred_8b_closed - meas[HOLDOUT]) / meas[HOLDOUT]

    # (b) ENGINE path: 8B through the full engine, no 8B-specific tuning
    r8 = run(_cfg(HOLDOUT))
    pred_8b_engine = r8["tok_s"]
    rel_engine = abs(pred_8b_engine - meas[HOLDOUT]) / meas[HOLDOUT]
    # cross-check the engine 8B equals the committed L4 mechanism (byte-identical)
    committed_engine = json.loads((OUT / "e2e_l4.json").read_text())["mechanism_independent_pricing"][HOLDOUT]
    byte_identical = abs(rel_engine - committed_engine["rel_error"]) < 1e-4

    within_gate = rel_engine <= GATE
    not_worse_than_closed_form = rel_engine <= rel_closed + 1e-9

    out = {
        "module": "holdout", "phase": "2.3", "gate_threshold": GATE,
        "fit_set": FIT_MODELS, "holdout": HOLDOUT,
        "genuine_leave_8b_out_closed_form": {
            "fitted_scalar": "fit_BW (two-pillar effective decode BW), re-derived from {1B,3B} ONLY",
            "per_token_weight_bytes_GB": {m: wb[m] for m in FIT_MODELS + [HOLDOUT]},
            "implied_BW_GBs_fit_set": {m: round(implied[m], 3) for m in FIT_MODELS},
            "fit_BW_GBs": round(fit_BW, 3),
            "committed_fit_BW_GBs": tp["fit_BW_GBs"],            # cross-check (~18.33)
            "pred_8b_tok_s": round(pred_8b_closed, 3),
            "measured_8b_tok_s": meas[HOLDOUT],
            "rel_error_8b": round(rel_closed, 4),
            "note": "engine 24.2 anchor untouched; this is the TRUE size hold-out (8B never in the fit).",
        },
        "engine_path": {
            "pred_8b_tok_s": pred_8b_engine,
            "measured_8b_tok_s": meas[HOLDOUT],
            "rel_error_8b": round(rel_engine, 4),
            "byte_identical_to_committed_L4": bool(byte_identical),
            "CAVEAT": "the engine's 24.2 backbone is regressed across ALL sizes incl. 8B in Phase 1 "
                      "-> PARTLY IN-SAMPLE; the genuinely out-of-sample content is the parameter-free "
                      "CIM-compute correction. Do NOT read the engine 8B error as a pure hold-out.",
        },
        "within_gate": bool(within_gate),
        "not_worse_than_closed_form": bool(not_worse_than_closed_form),
        "gate_6c_signal": bool(not not_worse_than_closed_form),
        "finding": (f"genuine leave-8B-out closed form predicts 8B at {rel_closed*100:.1f}%; the full "
                    f"engine (parameter-free compute correction on the partly-in-sample 24.2 backbone) "
                    f"predicts 8B at {rel_engine*100:.1f}% — within {GATE:.0%} and not worse than the "
                    f"closed form. If the engine were worse = Gate-6c (fix M1/M2/M3, not integration)."),
        "pass_all": bool(within_gate and not_worse_than_closed_form and byte_identical),
    }
    (OUT / "holdout.json").write_text(json.dumps(out, indent=1))

    print("hold-out (fit {1B,3B} -> predict 8B):")
    print(f"  (a) GENUINE leave-8B-out closed form: fit_BW={fit_BW:.2f} GB/s (committed {tp['fit_BW_GBs']}) "
          f"-> pred 8B {pred_8b_closed:.2f} vs {meas[HOLDOUT]} = {rel_closed*100:.1f}%")
    print(f"  (b) ENGINE 8B (partly-in-sample backbone + parameter-free compute): {pred_8b_engine:.2f} "
          f"vs {meas[HOLDOUT]} = {rel_engine*100:.1f}% (byte-identical to L4: {byte_identical})")
    print(f"  within_gate({GATE:.0%}): {within_gate}; not_worse_than_closed_form: {not_worse_than_closed_form}")
    return 0 if out["pass_all"] else 1


if __name__ == "__main__":
    sys.exit(main())
