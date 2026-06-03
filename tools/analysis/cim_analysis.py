"""Phase 0.3 analysis (Mac) — structure CIM raw results + C4 composed attention + C5 two-pillar.

Reads:
  measurements/aetina/metis_alpha_matmul_raw.json   (raw run_metis_cim results, synced from board)
  measurements/op_profile/*.json                    (Phase 0.2 op counts)
  measurements/metis_card/vendor_llm_int8.json      (B, L4 anchor)
Writes:
  measurements/aetina/metis_alpha_matmul.json       (structured: by group + PCIe floor A1d.5)
  measurements/aetina/cim_attention_composed.json   (C4)
  measurements/metis_card/twopillar_prediction.json (C5 hold-out)

Run: ./.venv/bin/python tools/analysis/cim_analysis.py
"""
import json, statistics
from pathlib import Path

AET = Path("measurements/aetina")
MC = Path("measurements/metis_card")
OP = Path("measurements/op_profile")

MODELS = {
    "llama-3.2-1b": dict(H=2048, F=8192, kv=512,  V=128256, hd=64,  heads=32, kvh=8, L=16),
    "llama-3.2-3b": dict(H=3072, F=8192, kv=1024, V=128256, hd=128, heads=24, kvh=8, L=28),
    "llama-3.1-8b": dict(H=4096, F=14336, kv=1024, V=128256, hd=128, heads=32, kvh=8, L=32),
    "qwen2.5-7b":   dict(H=3584, F=18944, kv=512,  V=152064, hd=128, heads=28, kvh=4, L=28),
}


def weight_bytes(c):  # INT8 = 1 byte/param; per-layer projections x L + embed + lm_head
    H, F, kv, V, L = c["H"], c["F"], c["kv"], c["V"], c["L"]
    per_layer = H * H + 2 * (H * kv) + H * H + 2 * (H * F) + F * H  # q,k,v,o,gate,up,down
    return per_layer * L + V * H + H * V


def structure_cim(raw):
    """Group raw CIM results + derive the per-call PCIe/DMA fixed overhead (A1d.5)."""
    by_group = {}
    for tid, r in raw.items():
        by_group.setdefault(r["group"], []).append(r)
    # A1d.5: per-call fixed overhead = system_lat - dev_lat over single-tile (untiled) shapes
    floors = [r["system_lat_us"] - r["dev_lat_us"] for r in raw.values()
              if "system_lat_us" in r and not r.get("tiled_extrapolated") and r.get("tiles", 1) == 1]
    pcie = {}
    if floors:
        pcie = {"fixed_overhead_us_median": round(statistics.median(floors), 1),
                "fixed_overhead_us_p95": round(sorted(floors)[int(0.95 * len(floors))], 1),
                "n_shapes": len(floors),
                "note": "per-call host<->device floor = system_lat - dev_lat; the gap dominates "
                        "small decode GEMVs (dev fast, system floor-bound)."}
    return {"by_group": by_group, "pcie_floor_A1d5": pcie}


def cim_proj_latency(struct, model, family):
    """dev latency (us) of a projection family from the structured CIM data (single-tile or tiled)."""
    for r in struct["by_group"].get("proj_decode", []):
        if r["model"] == model and r["family"] == family and "dev_lat_us" in r:
            return r["dev_lat_us"]
    return None


def c4_composed_attention(struct, pcie):
    """C4: composed CIM attention = conv-proxy floor + KV-reload (Alpha-topology penalty)."""
    c = MODELS["llama-3.1-8b"]
    fixed = (pcie.get("fixed_overhead_us_median", 0) or 0)
    # conv-proxy floor: QK^T + S.V dev latency (single head) at on-grid kv anchors
    floor = {}
    for r in struct["by_group"].get("attn_floor", []):
        if "dev_lat_us" in r:
            floor.setdefault(r["N"] if r["family"].startswith("qkT") else r["K"], {})[r["family"]] = r["dev_lat_us"]
    BW_pcie_GBs = 3.9  # A2 PCIe Gen3x4 (Alpha host link)
    out = {"note": "Alpha-topology penalty estimate (NOT production absolute). kv_bytes=2*kv*kvh*hd INT8; "
                   "reload = kv_bytes/BW + n_dma*fixed_overhead; n_dma = L (KV-reload per step).",
           "BW_pcie_GBs": BW_pcie_GBs, "fixed_overhead_us": fixed, "rows": []}
    for kv in sorted(floor):
        f = sum(floor[kv].values())  # QK^T + SV single-head dev floor
        kv_bytes = 2 * kv * c["kvh"] * c["hd"]
        reload_us = kv_bytes / (BW_pcie_GBs * 1e9) * 1e6 + c["L"] * fixed
        out["rows"].append({"kv": kv, "floor_us": round(f, 2), "kv_bytes": kv_bytes,
                            "reload_us": round(reload_us, 1), "composed_us": round(f + reload_us, 1),
                            "reload_share": round(reload_us / (f + reload_us), 3)})
    return out


def c5_two_pillar(vendor):
    """C5: ADR-0006 hold-out. Fit effective decode BW from 1b+3b, predict 8b tok/s."""
    wb = {m: weight_bytes(c) for m, c in MODELS.items()}
    meas = {m: vendor[f"{m}/1c"]["tok_s_median"] for m in ["llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b"]}
    # BW from 1b+3b: time/token = weight_bytes / BW  ->  BW = weight_bytes * tok_s
    fit = [wb[m] * meas[m] for m in ["llama-3.2-1b", "llama-3.2-3b"]]
    BW = statistics.mean(fit)
    pred_8b = BW / wb["llama-3.1-8b"]
    err = abs(pred_8b - meas["llama-3.1-8b"]) / meas["llama-3.1-8b"]
    return {"method": "ADR-0006 hold-out: fit decode BW on 1b+3b, predict 8b (size-invariance test)",
            "weight_bytes": wb, "measured_tok_s_1c": meas,
            "fit_BW_GBs": round(BW / 1e9, 2),
            "implied_BW_per_model_GBs": {m: round(wb[m] * meas[m] / 1e9, 2) for m in meas},
            "pred_8b_tok_s": round(pred_8b, 2), "measured_8b_tok_s": meas["llama-3.1-8b"],
            "rel_error": round(err, 3), "within_25pct": bool(err <= 0.25),
            "sanity_floor_note": "tok_s ~= BW / weight_bytes; decode is weight-streaming (r2~0.997 per voyager-sdk §9)."}


def main():
    raw_p = AET / "metis_alpha_matmul_raw.json"
    vendor = json.loads((MC / "vendor_llm_int8.json").read_text())
    struct, pcie = {"by_group": {}}, {}
    if raw_p.exists():
        raw = json.loads(raw_p.read_text())
        s = structure_cim(raw)
        struct, pcie = s, s["pcie_floor_A1d5"]
        (AET / "metis_alpha_matmul.json").write_text(json.dumps(s, indent=1))
        print(f"structured {len(raw)} CIM results; PCIe fixed overhead = {pcie.get('fixed_overhead_us_median')} us")
        c4 = c4_composed_attention(struct, pcie)
        (AET / "cim_attention_composed.json").write_text(json.dumps(c4, indent=1))
        print("C4 composed attention (8B):", [(r["kv"], r["composed_us"], f"reload {int(r['reload_share']*100)}%") for r in c4["rows"]])
    else:
        print("(no CIM raw yet — run after CIM completes; doing C5 only)")
    c5 = c5_two_pillar(vendor)
    (MC / "twopillar_prediction.json").write_text(json.dumps(c5, indent=1))
    print(f"\nC5 two-pillar hold-out: fit BW={c5['fit_BW_GBs']} GB/s (1b+3b) -> "
          f"pred 8b={c5['pred_8b_tok_s']} vs meas={c5['measured_8b_tok_s']} tok/s "
          f"({c5['rel_error']*100:.0f}% err, within25%={c5['within_25pct']})")
    print("  implied BW per model:", c5["implied_BW_per_model_GBs"])


if __name__ == "__main__":
    main()
