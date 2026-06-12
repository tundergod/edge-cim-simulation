"""Single source of truth for report/findings metric numbers (build artifact helper).

Every number that appears in a Phase-1 report table or the findings gate-summary is read
HERE from its committed `validation/reports/*.json` and formatted to the display string the
docs render. Chapters/findings carry `{{key}}` placeholders (see build_phase1_report.py /
build_findings.py); the build substitutes them and FAILS on any unresolved `{{...}}`. So a
number cannot be hand-mistyped in prose: it flows from the JSON, and if the JSON changes the
docs change. The only hand-written things here are (a) the JSON path and (b) the formatter —
both guarded by tests/test_report_metrics.py (every path must resolve).

Run `python tools/report/_metrics.py` to print every key=value (eyeball vs the rendered docs).
Import (path-based, no package): `sys.path.insert(0, <this dir>); import _metrics`.

Key naming: `<area>.<field>`; `*_pct` = a percentage NUMBER (the literal `%` stays in the doc
text, e.g. `median **{{cim.decode_median_pct}}%**`). Units (GB/s, µs, tok/s) likewise stay as
literal text next to the placeholder.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "validation/reports"
PARAMS = ROOT / "simulator/models/params"
SPECS = ROOT / "simulator/specs"


def _load(rel):
    return json.loads((REP / rel).read_text())


def _load_param(name):
    return json.loads((PARAMS / f"{name}.json").read_text())


def _load_spec(name):
    return json.loads((SPECS / f"{name}.json").read_text())


def _load_meas(rel):
    return json.loads((ROOT / "measurements" / rel).read_text())


# ---- formatters ---------------------------------------------------------------
def _p1(x):   # fraction -> 1-dp percent number   0.027 -> "2.7"
    return f"{x * 100:.1f}"


def _p0(x):   # fraction -> integer percent number 0.356 -> "36"
    return f"{round(x * 100)}"


def _f(x, d):  # fixed decimals                     2.7 -> "2.70" (d=2)
    return f"{x:.{d}f}"


def _i(x):    # rounded integer                     911.1 -> "911"
    return f"{int(round(x))}"


def load():
    """Return {key: display_string}. Raises (KeyError/FileNotFoundError) if a source path
    no longer resolves — that is the test's signal that a doc number lost its backing."""
    m1 = _load("phase1.1/m1.json")
    m2 = _load("phase1.1/m2.json")
    m4cpu11 = _load("phase1.1/m4_cpu.json")
    m4gpu11 = _load("phase1.1/m4_gpu.json")
    m7 = _load("phase1.1/m7.json")
    rc = _load("phase1.1/recompose.json")
    card = _load("phase1.2/cim_card_revalidate.json")
    pref = _load("phase1.2/cim_prefill_fit.json")
    mt = _load("phase1.5/cim_multitile.json")
    kv = _load("phase1.5/kv_append_spike.json")
    m1p = _load_param("m1_cim")
    m4cpu12 = _load("phase1.2/m4_cpu.json")
    gpurf = _load("phase1.2/m4_gpu_roofline.json")
    ram2 = _load("phase1.3/m2_ramulator2.json")
    onnxim = _load("phase1.3/m4_npu_onnxim.json")
    m2pcie = _load_param("m2_pcie")
    cpuic = _load_param("m4_cpu_instrcount")
    cpurk = _load_spec("cpu_rk3588")
    m5 = _load("phase1.1/m5.json")
    sweep = _load_meas("op_inventory/sweep_matrix.json")
    oh = cpuic["overhead_op_us"]
    a76 = cpurk["clusters"]["a76"]
    cbw = cpurk["cache_bw_GBs"]
    pm5 = m5["per_model"]

    g1 = m1["throughput_fit_gate_native"]
    return {
        # CIM (M1)
        "cim.decode_median_pct": _p1(g1["median"]),                                   # 2.7
        "cim.decode_p95_pct":    _p1(g1["p95"]),                                      # 14.9
        "cim.decode_max_pct":    _p1(g1["max"]),                                      # 17.6
        "cim.card_median_pct":   _p1(card["consistency"]["median_rel_diff"]),         # 4.8
        "cim.card_p95_pct":      _p1(card["consistency"]["p95_rel_diff"]),            # 9.7
        "cim.prefill_median_pct": _p1(pref["fit_quality"]["median_rel_err"]),         # 1.2
        "cim.prefill_fit_max_pct": _p1(pref["fit_quality"]["max_rel_err"]),           # 3.1 (affine tile fit max)
        "cim.prefill_holdout_pct": _p1(pref["holdout"]["median_rel_err"]),            # 0.9
        "cim.prefill_m_max":     _i(m1p["prefill_M_max"]),                            # 508
        "cim.prefill_affine_a":  _f(pref["affine_fit_tile_lat_us"]["a_weight_load_us"], 1),  # 40.8
        "cim.prefill_affine_b":  _f(pref["affine_fit_tile_lat_us"]["b_per_col_us"], 3),      # 0.094
        "cim.prefill_asymptote_tops": _f(pref["affine_fit_tile_lat_us"]["asymptote_TOPS"], 1),  # 89.2
        # CIM G_eff 2D fit params + decode anchor (m1_cim.json) — exposed so the page table is JSON-sourced
        "cim.geff_gmax":   _f(m1p["G_eff_Gmax_gops"], 2),                             # 333.67
        "cim.geff_na":     _f(m1p["G_eff_Na"], 1),                                    # 577.2
        "cim.geff_kb":     _f(m1p["G_eff_Kb"], 1),                                    # 574.1
        "cim.geff_ncores": _i(m1p["n_cores"]),                                        # 4
        "cim.geff_width":  _i(m1p["n_cores"] * m1p["core_width"]),                    # 2048
        "cim.decode_anchor_lat":  _f(m1p["prefill_M_decode_anchor"]["tile_lat_us"], 2),   # 41.83
        "cim.decode_anchor_gops": _f(m1p["prefill_M_decode_anchor"]["gops_measured"], 1), # 200.5
        # CIM multi-tile residency-cliff (Phase 1.5, Card-native): old tile-sum -> new cliff model
        "cim.multitile_old_median_pct": _p0(mt["old_vs_new"]["old_tilesum_median"]),  # 31
        "cim.multitile_old_max_pct":    _p0(mt["old_vs_new"]["old_tilesum_max"]),     # 100
        "cim.multitile_new_median_pct": _p1(mt["old_vs_new"]["new_cliff_median"]),    # 2.8
        "cim.multitile_new_max_pct":    _p1(mt["old_vs_new"]["new_cliff_max"]),       # 12.5
        "cim.multitile_holdout_pct":    _p1(mt["resident_holdout"]["median_relerr"]), # 3.3
        "cim.cliff_knee_m":   _f(mt["model"]["knee_M_params"], 1),                    # 8.2
        "cim.cliff_floor_gops": _i(mt["model"]["spill_floor_gops"]),                  # 70
        "cim.native_envelope_m": _i(mt["model"]["native_envelope_kn"] / 1e6),         # 17
        # KV-cache isolation SPIKE (Phase 1.5) — PROXY_INCONCLUSIVE (proxy structurally SRAM-bound)
        "kv.spike_bw_lo":    _f(kv["verdict"]["bw_range_GBs"][0], 1),                 # 9.6
        "kv.spike_bw_hi":    _f(kv["verdict"]["bw_range_GBs"][1], 1),                 # 44.4
        "kv.spike_m2_bw":    _f(kv["verdict"]["m2_measured_eff_BW_GBs"], 1),          # 24.2
        "kv.spike_dram_bw":  _f(kv["verdict"]["spill_dram_BW_GBs_converged"], 1),     # 35.5 (converged spill)
        "kv.spike_proxy_max": _f(kv["verdict"]["proxy_max_workset_M_elems"], 1),      # 2.1 (M-elems)
        "kv.spike_knee":     _f(kv["verdict"]["sram_knee_M_elems"], 1),               # 8.2 (M-elems)
        # Memory (M2)
        "mem.lpddr4x_eff":     _f(m2["params"]["measured_eff_BW_GBs"], 1),            # 24.2
        "mem.lpddr4x_peak":    _i(m2["params"]["measured_peak_GBs"]),                 # 34
        "mem.lpddr4x_peak_full": _f(m2["params"]["measured_peak_GBs"], 1),           # 34.1
        "mem.lpddr4x_eff_pct": _i(m2["params"]["efficiency_vs_measured_peak"] * 100), # 71
        "mem.lpddr5_eff":      _f(m2["params"]["sim_lpddr5_eff_GBs"], 1),             # 33.3
        "mem.lpddr5_peak":     _f(m2["params"]["sim_lpddr5_peak_GBs"], 1),            # 51.2
        "mem.lpddr5_eff_pct":  _i(m2["params"]["sim_lpddr5_eff_GBs"] / m2["params"]["sim_lpddr5_peak_GBs"] * 100),  # 65
        "mem.pcie_floor_us":   _i(m2["params"]["pcie_floor_us"]),                     # 911
        "mem.pcie_floor_full": _f(m2pcie["fixed_overhead_us_median"], 1),            # 911.1
        "mem.pcie_p95":        _f(m2pcie["fixed_overhead_us_p95"], 1),               # 1111.7
        "mem.pcie_bw":         _f(m2["params"]["pcie_BW_GBs"], 1),                    # 3.9
        "mem.ram2_device_eff": _f(ram2["ramulator2_device"]["efficiency"], 2),       # 0.92
        "mem.ram2_device_bw":  _f(ram2["ramulator2_device"]["eff_BW_GBs"], 1),       # 47.1
        "mem.ram2_system_eff": _f(ram2["analytic_system"]["efficiency"], 2),         # 0.65
        # CPU (M4)
        "cpu.softmax_median_pct": _p1(m4cpu11["softmax_fit_gate"]["median"]),         # 0.3
        "cpu.softmax_p95_pct":    _p1(m4cpu11["softmax_fit_gate"]["p95"]),            # 1.8
        "cpu.softmax_max_pct":    _p1(m4cpu11["softmax_fit_gate"]["max"]),            # 3.4
        "cpu.resid_median_pct":   _f(m4cpu12["overall_residual_pct"]["median"], 2),   # 1.18
        "cpu.resid_p95_pct":      _f(m4cpu12["overall_residual_pct"]["p95"], 2),      # 7.31
        "cpu.resid_max_pct":      _f(m4cpu12["overall_residual_pct"]["max"], 2),      # 13.12
        # CPU instruction-count roofline params (m4_cpu_instrcount.json + cpu_rk3588.json spec)
        "cpu.eta_c":   _f(cpuic["eta_c"], 4),                                         # 0.1521
        "cpu.eta_bw":  _f(cpuic["eta_bw"], 1),                                        # 0.6
        "cpu.overhead_residual": _f(oh["residual"], 3),                              # 0.789
        "cpu.overhead_rmsnorm":  _f(oh["rmsnorm"], 3),                               # 16.815
        "cpu.overhead_rope":     _f(oh["rope_apply"], 3),                            # 22.525
        "cpu.overhead_sampling": _f(oh["sampling_argmax"], 3),                       # 7.357
        "cpu.overhead_softmax":  _f(oh["softmax"], 3),                               # 15.497
        "cpu.overhead_swiglu":   _f(oh["swiglu"], 1),                                # 0.0
        "cpu.a76_lane_gops": _f(cpurk["neon"]["fp32_lanes"] * a76["ipc"] * a76["freq_ghz"], 1),  # 18.4
        "cpu.l1_bw":   _f(cbw["l1d_per_core"], 1),                                    # 73.6
        "cpu.l2_bw":   _f(cbw["l2_per_core"], 1),                                     # 36.8
        "cpu.l3_bw":   _f(cbw["l3_shared_per_core"], 1),                              # 18.4
        # M5 workload — op inventory coverage (m5.json + sweep_matrix.json)
        "m5.sweep_total":  _i(sweep["total"]),                                        # 580
        "m5.n_categories": _i(len(sweep["counts"])),                                  # 9
        "m5.cnt_matmul":    _i(sweep["counts"]["matmul"]),                            # 105
        "m5.cnt_attention": _i(sweep["counts"]["attention"]),                         # 95
        "m5.cnt_rope":      _i(sweep["counts"]["rope"]),                              # 190
        "m5.cnt_kv_cache":  _i(sweep["counts"]["kv_cache"]),                          # 9
        "m5.n_distinct_1b":   _i(pm5["llama-3.2-1b"]["n_distinct_ops"]),              # 38
        "m5.n_distinct_3b":   _i(pm5["llama-3.2-3b"]["n_distinct_ops"]),              # 38
        "m5.n_distinct_8b":   _i(pm5["llama-3.1-8b"]["n_distinct_ops"]),              # 38
        "m5.n_distinct_qwen": _i(pm5["qwen2.5-7b"]["n_distinct_ops"]),                # 39
        "m5.n_tasks":         _i(pm5["llama-3.1-8b"]["n_tasks"]),                     # 4
        "m5.orphans_1b":   _i(len(pm5["llama-3.2-1b"]["orphan_ops"])),                # 0
        "m5.orphans_3b":   _i(len(pm5["llama-3.2-3b"]["orphan_ops"])),                # 0
        "m5.orphans_8b":   _i(len(pm5["llama-3.1-8b"]["orphan_ops"])),                # 0
        "m5.orphans_qwen": _i(len(pm5["qwen2.5-7b"]["orphan_ops"])),                  # 0
        "m5.orphans_total": _i(sum(len(p["orphan_ops"]) for p in pm5.values())),      # 0
        # GPU (M4)
        "gpu.attn_median_pct":    _p1(m4gpu11["attn_offload_gate"]["median_relerr"]), # 0.6
        "gpu.attn_p95_pct":       _p1(m4gpu11["attn_offload_gate"]["p95_relerr"]),    # 1.1
        "gpu.attn_max_pct":       _p1(m4gpu11["attn_offload_gate"]["max_relerr"]),    # 1.1
        "gpu.roofline_median_pct": _p1(gpurf["error_vs_1p1_measured"]["median_abs_relerr"]),  # 2.7
        "gpu.roofline_p95_pct":    _p1(gpurf["error_vs_1p1_measured"]["p95_abs_relerr"]),      # 36.1
        "gpu.roofline_max_pct":    _p1(gpurf["error_vs_1p1_measured"]["max_abs_relerr"]),      # 65.8
        # NPU heavy-sim cross-check (M4)
        "npu.onnxim_median_delta_pct": _f(onnxim["median_abs_delta_pct"], 1),         # 317.9
        "npu.onnxim_max_delta_pct":    _f(onnxim["max_abs_delta_pct"], 1),            # 493.4
        # Recompose (e2e)
        "recompose.err_8b_pct": _p1(rc["rel_error_8b"]),                              # 9.5
        "recompose.pred_8b":    _f(rc["pred_8b_tok_s"], 2),                           # 2.44
        "recompose.meas_8b":    _f(rc["measured_8b_tok_s"], 2),                       # 2.70
        "recompose.meas_8b_1":  _f(rc["measured_8b_tok_s"], 1),                       # 2.7
        "recompose.fit_bw":     _f(rc["fit_BW_GBs"], 2),                              # 18.33
        # Energy (M7)
        "energy.cim_mj":   _f(m7["per_token_8b_decode_mJ"]["cim_proj_mJ"], 3),       # 1.001
        "energy.dram_mj":  _f(m7["per_token_8b_decode_mJ"]["dram_stream_mJ"], 3),    # 240.149
        "energy.cpu_mj":   _f(m7["per_token_8b_decode_mJ"]["cpu_support_mJ"], 1),    # 15.0
        "energy.total_mj": _f(m7["per_token_total_mJ"], 2),                          # 256.15
        "energy.corners": _i(m7["sensitivity_pm20pct"]["corners_tested"]),           # 16
        "energy.flips":   _i(m7["sensitivity_pm20pct"]["conclusion_flips"]),         # 0
    }


def substitute(text, metrics=None):
    """Replace every {{key}} in text using metrics (load() by default). Raise on any
    unresolved {{...}} so the build fails loudly rather than shipping a literal placeholder."""
    import re
    m = metrics if metrics is not None else load()
    missing = []

    def repl(mo):
        k = mo.group(1).strip()
        if k not in m:
            missing.append(k)
            return mo.group(0)
        return m[k]

    out = re.sub(r"\{\{([^}]+)\}\}", repl, text)
    if missing:
        raise KeyError(f"unresolved report metric placeholders: {sorted(set(missing))}")
    return out


if __name__ == "__main__":
    for k, v in load().items():
        print(f"{k} = {v}")
