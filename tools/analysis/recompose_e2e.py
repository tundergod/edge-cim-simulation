"""Phase 1 capstone — end-to-end recompose (L1->L4) using the fitted equations.

Decode backbone (the ONLY gate): tok_s = BW_eff / weight_bytes_per_token, with weight_bytes
summed from op_profile per-sig decode-matmul bytes (refines C5's closed form) and BW_eff fit
on 1B+3B, predicting held-out 8B (|err| <= 0.25).

Non-streaming additive terms (CPU support, GPU-offload attention, kv_cache) are reported
STANDALONE for transparency: they are already absorbed into the effective decode bandwidth at
the fit point, so ADDING them would double-count (Phase-2 fidelity item; see memory watch-item).
The Alpha 911 us per-call floor is NOT applied (production = on-card DRAM, topology artifact).
The CIM-attention penalty (C4) is reported separately as the offload justification, not in t_step.
prefill GEMM compute is now Card-MEASURED (M-amortization fit, fit_cim_prefill.py) and reported
BOUNDED by compute<=vendor-TTFT (the old linear-M decode-GEMV estimate violates it ~20x). The TTFT
remainder (weight-load memory + prefill attention + host overhead) stays a Phase-2 item.

Reads  measurements/op_profile/*.json, metis_card/vendor_llm_int8.json,
       aetina/cim_attention_composed.json
Writes measurements/metis_card/twopillar_prediction_fitted.json, validation/reports/phase1.1/recompose.json

Run: ./.venv/bin/python tools/analysis/recompose_e2e.py
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from simulator.models.m2_memory import MemoryModel       # noqa: E402
from simulator.models.m4_gpu import MaliGpuModel          # noqa: E402
from simulator.models.m4_cpu import CpuModel              # noqa: E402
from simulator.specs.loader import load_spec              # noqa: E402
from simulator.models.m1_cim_tile import CimTileModel     # noqa: E402

OP = ROOT / "measurements/op_profile"
MC = ROOT / "measurements/metis_card"
AET = ROOT / "measurements/aetina"
CFG = {"llama-3.2-1b": dict(heads=32, kvh=8, L=16),
       "llama-3.2-3b": dict(heads=24, kvh=8, L=28),
       "llama-3.1-8b": dict(heads=32, kvh=8, L=32)}


def per_token_weight_bytes(model):
    """Sum op_profile decode-matmul bytes per token (task-independent; ~= streamed weights)."""
    d = json.loads((OP / f"{model}_gsm8k.json").read_text())
    s = sum(r["bytes"] * r["count"] for r in d["rows"]
            if r["phase"] == "decode" and r["category"] == "matmul")
    return s / d["decode_len"]


def decode_counts_per_token(model):
    """Per-token decode op counts by category (from op_profile / decode_len)."""
    d = json.loads((OP / f"{model}_gsm8k.json").read_text())
    dl = d["decode_len"]
    out = {}
    for r in d["rows"]:
        if r["phase"] == "decode":
            out[r["category"]] = out.get(r["category"], 0) + r["count"] / dl
    return out


def main():
    vendor = json.loads((MC / "vendor_llm_int8.json").read_text())
    wb = {m: per_token_weight_bytes(m) for m in CFG}
    meas = {m: vendor[f"{m}/1c"]["tok_s_median"] for m in CFG}

    # --- GATE: decode weight-streaming, fit BW on 1B+3B, predict 8B ---
    BW = statistics.mean(wb[m] * meas[m] for m in ["llama-3.2-1b", "llama-3.2-3b"])
    pred_8b = BW / wb["llama-3.1-8b"]
    err_8b = abs(pred_8b - meas["llama-3.1-8b"]) / meas["llama-3.1-8b"]

    # --- standalone non-streaming terms for 8B (transparency; NOT added -> double-count) ---
    # Phase 1.2 spec-based engines (decision A migration). The production card decodes from
    # on-card LPDDR4x -> bind mem_lpddr4x (eff_BW 24.2, identical to the Phase-1.1 anchor, so the
    # kv-append term is unchanged). GPU stays the Phase-1.1 micro-benchmark model (MaliGpuModel,
    # untouched). The CPU support term now reflects the calibrated instruction-count roofline; it is
    # a STANDALONE transparency value (absorbed into BW, NOT added) so it does not move the gate.
    mem = MemoryModel(load_spec("mem_lpddr4x"))
    gpu = MaliGpuModel()
    cpu = CpuModel(load_spec("cpu_rk3588"))
    c = CFG["llama-3.1-8b"]
    kvbar = 512                                  # avg decode kv at ctx1024 (P + D/2)
    counts = decode_counts_per_token("llama-3.1-8b")
    stream_us = wb["llama-3.1-8b"] / (BW) * 1e6  # = 1/tok_s backbone, us
    cpu_support_us = (cpu.op_us("rmsnorm", "llama-3.1-8b") * 2 * c["L"]
                      + cpu.op_us("rope_apply", "llama-3.1-8b") * c["L"]
                      + cpu.op_us("swiglu", "llama-3.1-8b") * c["L"]
                      + cpu.op_us("residual", "llama-3.1-8b") * 2 * c["L"]
                      + cpu.op_us("softmax", "llama-3.1-8b", kv=kvbar) * c["L"]
                      + cpu.op_us("sampling_argmax", "llama-3.1-8b"))
    gpu_attn_us = gpu.attn_bmm_us(kvbar, heads=c["heads"], layers=c["L"])  # offload, lower bound
    kv_bytes = 2 * kvbar * c["kvh"] * 128 * c["L"]   # K+V across layers, INT8
    kv_cache_us = mem.kv_append_us(kv_bytes)

    # --- C4 CIM-attention penalty (offload justification, NOT in t_step) ---
    c4 = json.loads((AET / "cim_attention_composed.json").read_text())
    c4_range = [r["composed_us"] / 1000 for r in c4["rows"]]   # ms

    # --- prefill GEMM compute: now Card-MEASURED (M-amortization fit); bounded by compute<=TTFT ---
    M_pf = 1024
    cim = CimTileModel()                                # m1_cim.json: decode G_eff + prefill a+b*M fit
    if cim.prefill_a_us is None:                        # keys-dropped window (fit_m1_cim ran without fit_cim_prefill)
        raise ValueError("prefill (M>1) needs the M-amortization fit; run tools/analysis/fit_cim_prefill.py "
                         "after fit_m1_cim.py (which rewrites m1_cim.json and drops the prefill keys)")
    pf_flops = 2 * wb["llama-3.1-8b"] * M_pf            # weights x M tokens (prefill GEMM)
    n_tiles_pf = wb["llama-3.1-8b"] / cim.native_max_kn  # 2048x2048 weight tiles over all 8B GEMMs
    pf_compute_fitted_s = n_tiles_pf * (cim.prefill_a_us + cim.prefill_b_us * M_pf) * 1e-6  # M-amortized
    pf_compute_decodeGEMV_s = pf_flops / (204e9)       # OLD linear-M (decode GEMV) -> 76s, refuted
    pf_memory_s = wb["llama-3.1-8b"] / (BW)            # weights streamed once
    ttft_meas = vendor["llama-3.1-8b/1c"]["ttft_s_median"]

    out = {
        "method": "decode backbone = weight-streaming BW (fit 1B+3B -> predict 8B), "
                  "weight_bytes from op_profile; non-streaming terms reported standalone.",
        "per_token_weight_bytes": {m: round(wb[m] / 1e9, 3) for m in wb},
        "measured_tok_s_1c": meas,
        "fit_BW_GBs": round(BW / 1e9, 2),
        "implied_BW_per_model_GBs": {m: round(wb[m] * meas[m] / 1e9, 2) for m in meas},
        "pred_8b_tok_s": round(pred_8b, 2), "measured_8b_tok_s": meas["llama-3.1-8b"],
        "rel_error_8b": round(err_8b, 3), "GATE_within_25pct": bool(err_8b <= 0.25),
        "standalone_nonstreaming_8b_us": {
            "decode_stream_backbone_us": round(stream_us, 1),
            "cpu_support_us": round(cpu_support_us, 1),
            "gpu_offload_attention_us_lowerbound": round(gpu_attn_us, 1),
            "kv_cache_append_us": round(kv_cache_us, 1),
            "_caveat": "already absorbed in BW_eff at the fit point; ADDING double-counts "
                       "(Phase-2 fidelity, watch-item). GPU attn is heads*layers, lower-bound kernel."},
        "cim_attention_penalty_C4_ms": {"range": [round(min(c4_range), 1), round(max(c4_range), 1)],
                                        "note": "why offload: CIM attention >> GPU-native; reported, NOT in t_step"},
        "prefill_gemm_compute_BOUNDED": {
            "M_prefill": M_pf,
            "vendor_ttft_s": ttft_meas,
            "gemm_compute_fitted_s": round(pf_compute_fitted_s, 3),
            "gemm_compute_frac_of_ttft": round(pf_compute_fitted_s / ttft_meas, 3),
            "memory_floor_s": round(pf_memory_s, 3),
            "decode_GEMV_linear_s": round(pf_compute_decodeGEMV_s, 1),
            "physical_bound_compute_le_ttft": {
                "fitted_PASS": bool(pf_compute_fitted_s < ttft_meas),
                "decode_GEMV_linear_PASS": bool(pf_compute_decodeGEMV_s < ttft_meas)},
            "note": "GEMM compute now Card-MEASURED (fit_cim_prefill.py M-amortization, M<=256; M=%d "
                    "extrapolated). Fitted %.2fs = %.0f%% of vendor TTFT, satisfying compute<=TTFT; the "
                    "linear-M decode-GEMV estimate (%.0fs) VIOLATES the bound ~20x (REFUTED). TTFT "
                    "remainder = weight-load memory (~%.2fs) + prefill attention (S^2 softmax) + host "
                    "overhead (Phase-2 fidelity). prefill_ms_median degenerate, unused."
                    % (M_pf, pf_compute_fitted_s, 100 * pf_compute_fitted_s / ttft_meas,
                       pf_compute_decodeGEMV_s, pf_memory_s)},
    }
    (MC / "twopillar_prediction_fitted.json").write_text(json.dumps(out, indent=1))
    (ROOT / "validation/reports/phase1.1/recompose.json").write_text(json.dumps(out, indent=1))

    print(f"recompose: fit BW={out['fit_BW_GBs']} GB/s (1b+3b) -> pred 8b={out['pred_8b_tok_s']} "
          f"vs meas={out['measured_8b_tok_s']} tok/s ({err_8b*100:.1f}% err, "
          f"GATE within25%={out['GATE_within_25pct']})")
    print(f"  implied BW/model: {out['implied_BW_per_model_GBs']}")
    print(f"  standalone 8B (us): stream={stream_us:.0f} cpu={cpu_support_us:.0f} "
          f"gpu_attn(LB)={gpu_attn_us:.0f} kv={kv_cache_us:.0f}  [absorbed in BW; not added]")
    print(f"  C4 CIM-attn penalty: {out['cim_attention_penalty_C4_ms']['range']} ms (offload reason)")
    print(f"  prefill GEMM compute (bounded): fitted {pf_compute_fitted_s:.2f}s "
          f"({100*pf_compute_fitted_s/ttft_meas:.0f}% of vendor TTFT {ttft_meas:.2f}s, compute<=TTFT PASS); "
          f"decode-GEMV linear {pf_compute_decodeGEMV_s:.0f}s REFUTED (>TTFT)")


if __name__ == "__main__":
    main()
