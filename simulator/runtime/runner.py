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
    plat = Platform(cfg.model, memory_spec=cfg.memory_spec, knee_GBs=cfg.knee_GBs,
                    interconnect_efficiency=cfg.interconnect_efficiency,
                    bw_efficiency=cfg.bw_efficiency)
    model_obj = op_profile.Model(cfg.model)
    concurrency = not cfg.concurrency_off
    contention = not cfg.contention_off
    price_compute = not cfg.compute_off

    # decode: representative steady-state kv at mid-generation
    kv = max(1, cfg.context // 2)
    dec = assign(build_token_dag(cfg.model, "decode", kv, _model_obj=model_obj))
    t_dec_us = run_dag(dec, plat, plat.bw, concurrency=concurrency, contention=contention,
                       price_compute=price_compute)
    tok_s = 1e6 / t_dec_us if t_dec_us > 0 else 0.0

    # prefill: one forward at P (TTFT reported only, not gated)
    pre = assign(build_token_dag(cfg.model, "prefill", cfg.prefill_len, _model_obj=model_obj))
    t_pre_us = run_dag(pre, plat, plat.bw, concurrency=concurrency, contention=contention,
                       price_compute=price_compute)

    e_tok = _energy_per_token_J(dec, plat)
    return {
        "model": cfg.model,
        "scheduler": cfg.scheduler,
        "context": cfg.context,
        "kv_eval": kv,
        "prefill_len": cfg.prefill_len,
        "decode_len": cfg.decode_len,
        "decode_token_us": round(t_dec_us, 3),
        "tok_s": round(tok_s, 4),
        "ttft_s_reported_not_gated": round(t_pre_us / 1e6, 4),
        "energy_per_token_J": e_tok,
        "energy_band_J": [e_tok * 0.8, e_tok * 1.2],
        "memory_eff_BW_GBs": plat.bw.eff_BW,
        "ablations": {"concurrency_off": cfg.concurrency_off, "contention_off": cfg.contention_off,
                      "compute_off": cfg.compute_off},
        "provenance": cfg.provenance,
        "calibrated_anchor": cfg.is_calibrated_anchor(),
    }
