"""Runner — wire SimConfig -> M5 (workload) -> M6 (scheduler) -> M3 (engine) -> M7
(energy) and emit metrics (Phase 2.1).

Decode tok/s is the gated quantity: build one steady-state decode token DAG at a
representative kv (= context//2; LLMServingSim-style per-iteration reuse, not a
full-generation expansion), price it through the event engine, tok/s = 1e6 /
token_us. TTFT is REPORTED, not gated (prefill path is analytic/unvalidated, D9).
Energy is an estimate reported as a +/-20% band (M7, no power telemetry).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "tools" / "trace_export"))
import op_profile  # noqa: E402

from simulator.runtime.workload import build_token_dag  # noqa: E402
from simulator.runtime.scheduler import all_cim_assign  # noqa: E402
from simulator.runtime.platform import Platform  # noqa: E402
from simulator.runtime.events import run_dag  # noqa: E402

_SCHEDULERS = {"all_cim": all_cim_assign}


def _energy_per_token_J(dag, plat):
    return sum(plat.energy_J(n, plat.compute_us(n)) for n in dag.nodes)


def run(cfg):
    """Run one SimConfig -> metrics dict."""
    if cfg.scheduler not in _SCHEDULERS:
        raise ValueError(f"unknown scheduler '{cfg.scheduler}' (have {sorted(_SCHEDULERS)})")
    assign = _SCHEDULERS[cfg.scheduler]

    # fail-loud on knobs that are accepted by SimConfig but NOT yet wired in 2.1
    # (so a user can't run a silently-inert experiment). These land in later waves.
    if cfg.topology != "cim_topo_card":
        raise NotImplementedError(
            f"topology '{cfg.topology}': the numeric topology effect (host-MMIO PCIe floor, "
            f"on-card vs edge) is Wave 2.3 (validate_topology_ab). 2.1 wires cim_topo_card only.")
    nonanalytic = {u: e for u, e in cfg.engine.items() if e != "analytic"}
    if nonanalytic:
        raise NotImplementedError(
            f"engine backend(s) {nonanalytic}: only 'analytic' is wired into the runtime in 2.1 "
            f"(onnxim/scalesim/ramulator2 heavy backends are a later wave).")
    if cfg.scheduler == "all_cim" and not (cfg.units.get("cim") and cfg.units.get("cpu")):
        raise ValueError("all_cim scheduler requires units cim+cpu enabled "
                         "(gpu/npu toggles have no effect under all_cim — heterogeneous "
                         "placement is Wave 2.2).")
    plat = Platform(cfg.model, memory_spec=cfg.memory_spec, knee_GBs=cfg.knee_GBs,
                    interconnect_efficiency=cfg.interconnect_efficiency,
                    bw_efficiency=cfg.bw_efficiency)
    model_obj = op_profile.Model(cfg.model)
    concurrency = not cfg.concurrency_off
    contention = not cfg.contention_off
    price_compute = not cfg.compute_off

    # capacity is a FEASIBILITY gate, NOT a throughput knob: decode tok/s is set by bandwidth,
    # not capacity (so it is deliberately not a Platform timing input). Fail-loud if the model's
    # resident INT8 weight footprint does not fit -> an impossible config can't be "calibrated".
    # Capacity-dependent BEHAVIOR (residency cliffs, spill) is a later wave.
    wrows = model_obj.profile(cfg.prefill_len, 1)   # one decode token; count already includes x layers
    weight_bytes = sum(r["bytes"] * r["count"] for r in wrows
                       if r["phase"] == "decode" and r["category"] == "matmul")
    footprint_GB = weight_bytes / 1e9
    if cfg.memory_capacity_GB < footprint_GB:
        raise ValueError(
            f"infeasible config: {cfg.model} needs ~{footprint_GB:.2f} GB resident (INT8 weights) "
            f"but memory_capacity_GB={cfg.memory_capacity_GB}. Capacity is a feasibility gate in 2.1 "
            f"(throughput is bandwidth-bound, not capacity-bound); capacity-dependent behavior is a later wave.")

    # decode tok/s = per-token latency AVERAGED over the decode kv range (LLMServingSim-style
    # per-iteration reuse): sample a few kv positions across [P, min(context, P+D)] and average,
    # so decode_len / context actually affect the result (kv-cache + attention traffic grow with kv).
    P, D = cfg.prefill_len, max(1, cfg.decode_len)
    kv_hi = max(1, min(cfg.context, P + D))
    kv_lo = max(1, min(P, kv_hi))
    kv_pts = sorted({kv_lo, (kv_lo + kv_hi) // 2 or 1, kv_hi})
    tok_us = [run_dag(assign(build_token_dag(cfg.model, "decode", k, _model_obj=model_obj)),
                      plat, plat.bw, concurrency=concurrency, contention=contention,
                      price_compute=price_compute) for k in kv_pts]
    t_dec_us = sum(tok_us) / len(tok_us)
    tok_s = 1e6 / t_dec_us if t_dec_us > 0 else 0.0
    total_generation_s = D * t_dec_us / 1e6
    dec = assign(build_token_dag(cfg.model, "decode", kv_pts[len(kv_pts) // 2], _model_obj=model_obj))

    # prefill: one forward at P (TTFT reported only, not gated)
    pre = assign(build_token_dag(cfg.model, "prefill", cfg.prefill_len, _model_obj=model_obj))
    t_pre_us = run_dag(pre, plat, plat.bw, concurrency=concurrency, contention=contention,
                       price_compute=price_compute)

    e_tok = _energy_per_token_J(dec, plat)
    return {
        "model": cfg.model,
        "scheduler": cfg.scheduler,
        "context": cfg.context,
        "kv_eval_points": kv_pts,
        "prefill_len": cfg.prefill_len,
        "decode_len": cfg.decode_len,
        "decode_token_us": round(t_dec_us, 3),
        "tok_s": round(tok_s, 4),
        "total_generation_s": round(total_generation_s, 4),
        "ttft_s_reported_not_gated": round(t_pre_us / 1e6, 4),
        "energy_per_token_J": e_tok,
        "energy_band_J": [e_tok * 0.8, e_tok * 1.2],
        "memory_eff_BW_GBs": plat.bw.eff_BW,
        "model_footprint_GB": round(footprint_GB, 3),
        "memory_capacity_GB": cfg.memory_capacity_GB,
        "capacity_note": "feasibility gate only; capacity does NOT affect decode tok/s in 2.1 (bandwidth-bound)",
        "ablations": {"concurrency_off": cfg.concurrency_off, "contention_off": cfg.contention_off,
                      "compute_off": cfg.compute_off},
        "provenance": cfg.provenance,
        "calibrated_anchor": cfg.is_calibrated_anchor(),
    }
