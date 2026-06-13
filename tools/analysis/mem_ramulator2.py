"""Phase 1.3 — Ramulator2 (v2.1) LPDDR5 single-stream BW cross-check -> simulator/engines/ramulator2/lpddr5_eff.json.

Drives Ramulator 2.1's own latency-throughput harness IN-PROCESS (no CLI/YAML — v2.1 is
Python-bindings-only) and reads its computed `total_throughput_MBps`. Measures the saturated
streaming bandwidth of LPDDR5-6400 twice: refresh OFF (peak, must hit >=95% of the channel
theoretical peak = saturation proof) and refresh ON / AllBank (achievable). The achievable
efficiency (achievable / max_theoretical) is the transferable quantity; we scale it onto our
64-bit spec peak (51.2 GB/s) to get eff_BW for the engine='ramulator2' backend, and compare it to
the analytic 0.65 efficiency assumption (mem_lpddr5 spec).

A single streaming number (NOT per-KV): saturated streaming BW is a channel property, independent
of KV length (KV changes weight bytes = the workload, not the memory's achievable BW). Reuses
v2.1's `resolve_spec` / `run_simulation` / `checks.py` verbatim (no re-derivation; no manual tCK).

NOT silicon — `simulated (Ramulator2 v2.1 LPDDR5_6400 streaming)`. Run AFTER tools/ramulator2/build.sh.

Run: ./.venv/bin/python tools/analysis/mem_ramulator2.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "tools/ramulator2/upstream"
# v2.1 module lives under python/; the latency_throughput harness is the `tests` package at the root.
sys.path[:0] = [str(UPSTREAM), str(UPSTREAM / "python")]

from tests.latency_throughput.utils.runner import run_simulation          # noqa: E402
from tests.latency_throughput.utils.spec import resolve_spec              # noqa: E402
from tests.latency_throughput.utils.checks import (                       # noqa: E402
    check_streaming_peak_bandwidth, check_streaming_bandwidth)

OUT = ROOT / "simulator/engines/ramulator2/lpddr5_eff.json"
SPEC_PEAK_GBS = 51.2   # our 64-bit LPDDR5-6400 spec peak (mem_lpddr5.json); efficiency scales onto this
ANALYTIC_EFF = 0.65    # the analytic assumption we cross-check (mem_lpddr5 sim_efficiency)

CFG = {"name": "LPDDR5", "dram_class": "LPDDR5", "org_preset": "LPDDR5_8Gb_x16",
       "timing_preset": "LPDDR5_6400", "controller_class": "LPDDR5", "stream_cls": 64}

_STREAM = dict(nop_counter=1, read_ratio=100, num_probe_requests=0,
               num_streaming_requests=1_000_000, frontend_clock_ratio=4,
               warmup_cycles=10_000, streaming_only=True)


def main():
    spec = resolve_spec(CFG)
    peak_stats = run_simulation(CFG, refresh_enabled=False, **_STREAM)   # NoRefresh -> channel peak
    ach_stats = run_simulation(CFG, refresh_enabled=True, **_STREAM)     # AllBank  -> achievable

    peak = check_streaming_peak_bandwidth(peak_stats, spec)              # measured vs max_theoretical
    ach = check_streaming_bandwidth(ach_stats, spec)                     # measured vs max_achievable
    peak_bw = peak["measured_streaming_bw"]                              # GB/s, single x16 channel
    ach_bw = ach["measured_streaming_bw"]
    peak_eff = peak_bw / spec.max_theoretical_bw
    efficiency = ach_bw / spec.max_theoretical_bw                        # the transferable quantity

    # saturation proof (B4): the no-refresh stream must reach >=95% of the channel theoretical peak.
    assert peak_eff >= 0.95, (f"NOT SATURATED: peak streaming {peak_bw:.2f} GB/s = {peak_eff:.1%} of "
                              f"theoretical {spec.max_theoretical_bw:.2f} (need >=95%); bump frontend_clock_ratio")
    # cross-check: achievable efficiency must agree with v2.1's own refresh-overhead estimate.
    refresh_eff = spec.max_achievable_bw / spec.max_theoretical_bw
    assert abs(efficiency - refresh_eff) < 0.05, (f"achievable eff {efficiency:.3f} disagrees with "
                              f"v2.1 refresh-overhead estimate {refresh_eff:.3f} (>5%)")

    eff_BW = round(SPEC_PEAK_GBS * efficiency, 1)
    out = {
        "_doc": "Ramulator2 v2.1 LPDDR5_6400 saturated-streaming BW cross-check. efficiency = achievable"
                " / max_theoretical (single x16 channel); eff_BW = 51.2 * efficiency (scaled onto the"
                " 64-bit spec peak; LPDDR5 has no pseudochannel doubling so efficiency transfers).",
        "eff_BW_GBs": eff_BW,
        "efficiency": round(efficiency, 4),
        "peak_efficiency_saturation": round(peak_eff, 4),
        "channel_width": spec.channel_width,
        "bytes_per_req": spec.bytes_per_req,
        "rate": spec.rate,
        "peak_GBs_single_channel": round(spec.max_theoretical_bw, 2),
        "achievable_GBs_single_channel": round(ach_bw, 2),
        "refresh_mode": "AllBank",
        "v2_1_commit": "278f1effc3838099a6ffe0ad5f9f572fea80c948",
        "analytic_efficiency_compared": ANALYTIC_EFF,
        "honesty": "simulated (Ramulator2 v2.1 LPDDR5_6400 streaming), NOT silicon",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"Ramulator2 v2.1 LPDDR5_6400: peak {peak_bw:.2f}/{spec.max_theoretical_bw:.2f} GB/s "
          f"({peak_eff:.1%} saturated) | achievable {ach_bw:.2f} -> efficiency {efficiency:.3f} "
          f"(analytic {ANALYTIC_EFF}) -> eff_BW {eff_BW} GB/s (51.2 x eff) -> {OUT}")


if __name__ == "__main__":
    main()
