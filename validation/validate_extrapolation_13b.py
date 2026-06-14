"""13B/32GB-class size extrapolation (Phase 2.3, ADR-0006) — Qwen2.5-14B, ENGINE path.

NOT a supported config and NOT validated: Qwen2.5-14B (~14B, one doubling beyond the 8B measured
envelope) stands in for the ADR-0006 "~13B/32GB" stretch point. There is NO 14B silicon. The point
is produced as a MECHANISTIC-MODEL EXTRAPOLATION (engine path): an ADR-0007 FakeTensor/meta trace
(device-independent — no weights/GPU/Card) feeds the SAME M5/M3 engine the 1B/3B/8B runs use.

This is the dedicated CI gate for the most-misread artifact of the phase. It asserts the full
chain holds AND that every extrapolation dimension is labelled:
  - op inventory + value-flow fixture exist; op_profile.Model self-validates; structural oracle passes
  - engine runner.run output is sane (positive tok/s, energy band)
  - THREE extrapolation flags: model-size (>8B), capacity (>16 GB SKU), and unit_shape
    (per-op CIM shapes beyond the M1 native measured envelope -> cim_shape_extrapolated_count)
  - PINNED capacity formula -> footprint margins vs 16/32 GiB + the overflow crossover context (#58)

Writes validation/reports/phase2/extrapolation_13b.json
Run:   ./.venv/bin/python validation/validate_extrapolation_13b.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "trace_export"))
from simulator.runtime.config import SimConfig  # noqa: E402
from simulator.runtime.runner import run  # noqa: E402
from simulator.runtime.platform import Platform  # noqa: E402
from simulator.runtime.scheduler import SCHEDULERS  # noqa: E402
from simulator.runtime.workload import build_token_dag, structural_check  # noqa: E402
import op_profile  # noqa: E402

OUT = ROOT / "validation/reports/phase2"
KEY = "qwen2.5-14b"
CONTEXT = 1024
SKU_GB = 32          # hypothetical >16 GiB SKU so the fail-loud feasibility gate passes
ENVELOPE_GB = 16     # production card


def _footprint_GB(weights_GB, cfg, context):
    """PINNED resident-footprint formula (validator-side; the runner gate counts weights ONLY,
    runner.py:89). All terms from the traced config; act = worst-case PREFILL peak (conservative)."""
    kv_GB = 2 * cfg["n_layers"] * cfg["kv_heads"] * cfg["head_dim"] * context * 1 / 1e9   # INT8 KV
    act_GB = 1 * context * cfg["hidden"] * 2 / 1e9                                         # fp16 prefill peak
    return weights_GB + kv_GB + act_GB, kv_GB, act_GB


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inv_path = ROOT / f"measurements/op_inventory/{KEY}.json"
    fix_path = ROOT / f"traces/fixture/{KEY}.json.gz"
    if not inv_path.exists() or not fix_path.exists():
        raise FileNotFoundError(f"13B-class extrapolation needs BOTH {inv_path} and {fix_path} "
                                f"(ADR-0007 meta-trace + value-flow fixture). Generate them first.")
    cfg_dims = json.loads(inv_path.read_text())["config"]

    # op_profile.Model construction self-validates the inventory; structural oracle checks the fixture DAG
    op_profile.Model(KEY)
    struct = {phase: structural_check(KEY, phase, L) for phase, L in (("decode", 300), ("prefill", 700))}
    struct_ok = all(ok for ok, _ in struct.values())

    # engine extrapolation on a 32 GB SKU
    cfg = SimConfig.from_dict({
        "workload": {"model": KEY, "context": CONTEXT},
        "platform": {"topology": "cim_topo_card", "memory_spec": "mem_lpddr4x", "memory_capacity_GB": SKU_GB},
        "scheduler": {"policy": "all_cim"},
        "tunables": {"pipeline": False},
    })
    r = run(cfg)
    engine_sane = r["tok_s"] > 0 and r["energy_band_J"][0] > 0

    # extrapolation dimension 3: per-op CIM shapes beyond the M1 native measured envelope
    plat = Platform(KEY, topology="cim_topo_card", memory_spec="mem_lpddr4x")
    dec = SCHEDULERS["all_cim"].assign(build_token_dag(KEY, "decode", CONTEXT // 2), cfg)
    cim_matmuls = [n for n in dec.nodes if n.category == "matmul" and n.unit == "cim"]
    cim_shape_extrap = sum(1 for n in cim_matmuls
                           if "EXTRAPOLATED beyond native envelope" in plat.price(n).get("compute_provenance", ""))

    # pinned capacity formula -> margins + overflow crossover context
    weights_GB = r["model_footprint_GB"]                       # unique INT8 weights (runner footprint)
    footprint_GB, kv_GB, act_GB = _footprint_GB(weights_GB, cfg_dims, CONTEXT)
    margin_16 = round(ENVELOPE_GB - footprint_GB, 3)
    margin_32 = round(SKU_GB - footprint_GB, 3)
    # context at which footprint crosses 16 GiB: weights + C*(per-ctx kv+act) = 16
    per_ctx = (2 * cfg_dims["n_layers"] * cfg_dims["kv_heads"] * cfg_dims["head_dim"] * 1
               + cfg_dims["hidden"] * 2) / 1e9
    overflow_ctx_16 = int((ENVELOPE_GB - weights_GB) / per_ctx) if per_ctx > 0 else None

    flags = {
        "model_size_extrapolated": True,      # 14B > 8B measured envelope
        "capacity_extrapolated": SKU_GB > ENVELOPE_GB,
        "unit_shape_extrapolated": cim_shape_extrap > 0,
    }
    out = {
        "module": "extrapolation_13b", "phase": "2.3",
        "model": KEY, "hf_repo": "Qwen/Qwen2.5-14B", "params_approx": "~14B",
        "label": "engine extrapolation (mechanistic-model extrapolation via ADR-0007 meta-trace); "
                 "NOT validated — no 14B silicon. Stands in for the ADR-0006 ~13B/32GB stretch point.",
        "config": cfg_dims,
        "artifacts": {"op_inventory": str(inv_path.relative_to(ROOT)),
                      "value_flow_fixture": str(fix_path.relative_to(ROOT))},
        "op_profile_self_validation": True,    # would have raised above otherwise
        "structural_oracle": {p: {"ok": ok, "detail": d} for p, (ok, d) in struct.items()},
        "engine": {"tok_s": r["tok_s"], "decode_token_us": r["decode_token_us"],
                   "eff_BW_GBs": round(r["memory_eff_BW_GBs"], 3),
                   "energy_band_J": r["energy_band_J"], "sane": bool(engine_sane)},
        "extrapolation_flags": flags,
        "cim_shape_extrapolated_count": cim_shape_extrap,
        "cim_matmul_count": len(cim_matmuls),
        "cim_shape_note": "not merely 'a bigger model': %d of %d per-decode CIM matmuls have shapes "
                          "BEYOND the M1 native measured envelope (floor extrapolation, m1_cim_tile."
                          "is_extrapolated)." % (cim_shape_extrap, len(cim_matmuls)),
        "capacity": {
            "context": CONTEXT,
            "weights_GB": round(weights_GB, 3), "kv_GB": round(kv_GB, 4), "act_GB": round(act_GB, 4),
            "resident_footprint_GB": round(footprint_GB, 3),
            "footprint_16GiB_margin_GB": margin_16, "footprint_32GiB_margin_GB": margin_32,
            "capacity_risk_16GiB": bool(margin_16 < 0),
            "overflow_context_16GiB": overflow_ctx_16,
            "note": ("at ctx %d, 14B fits the 16 GiB card with %.2f GB headroom, but overflows it at "
                     "ctx ~%s -> spill to host LPDDR over PCIe (~3.9 GB/s) -> decode collapse, NOT "
                     "modelled in v1 (capacity = fail-loud feasibility gate; #58)."
                     % (CONTEXT, margin_16, overflow_ctx_16)),
        },
        "pass_all": bool(struct_ok and engine_sane and all(flags.values())),
    }
    (OUT / "extrapolation_13b.json").write_text(json.dumps(out, indent=1))

    print(f"13B-class extrapolation = {KEY} (~14B, engine extrapolation, NOT validated):")
    print(f"  engine: {r['tok_s']:.3f} tok/s @ {r['memory_eff_BW_GBs']:.1f} GB/s (decode {r['decode_token_us']:.0f} us)")
    print(f"  structural oracle: {struct_ok}; op_profile self-val: OK")
    print(f"  extrapolation flags: {flags}  cim_shape_extrapolated={cim_shape_extrap}/{len(cim_matmuls)}")
    print(f"  capacity@ctx{CONTEXT}: footprint={footprint_GB:.2f} GB (16GiB margin {margin_16:.2f}, "
          f"32GiB margin {margin_32:.2f}); overflows 16 GiB at ctx ~{overflow_ctx_16}")
    return 0 if out["pass_all"] else 1


if __name__ == "__main__":
    sys.exit(main())
