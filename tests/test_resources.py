"""Phase 2.1 — SharedBandwidth saturating-knee model (simulator/runtime/resources.py).

The aggregate rises with concurrent streams until it saturates at the knee
(memory-wall shape); each stream's share falls; contention=False sums linearly.
This is the SIMULATED contention model (no concurrent silicon; issue #52) —
validated here as a shape, not against a measured knee.

    .venv/bin/pytest tests/test_resources.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.runtime.resources import SharedBandwidth, ComputeUnit  # noqa: E402


def test_aggregate_saturates_at_knee():
    bw = SharedBandwidth(eff_BW_GBs=24.2, knee_GBs=27.0)
    assert abs(bw.aggregate_GBs(1) - 24.2) < 1e-9      # single stream: full eff_BW
    assert abs(bw.aggregate_GBs(2) - 27.0) < 1e-9      # two streams: capped at knee
    assert abs(bw.aggregate_GBs(8) - 27.0) < 1e-9      # saturated, not 8x
    # rising-then-flat shape
    aggs = [bw.aggregate_GBs(k) for k in (1, 2, 3, 4)]
    assert aggs[0] < aggs[1] and aggs[1] == aggs[2] == aggs[3]


def test_per_stream_share_falls():
    bw = SharedBandwidth(eff_BW_GBs=24.2, knee_GBs=27.0)
    assert bw.per_stream_GBs(1) > bw.per_stream_GBs(2) > bw.per_stream_GBs(4)
    assert abs(bw.per_stream_GBs(4) - 27.0 / 4) < 1e-9


def test_contention_off_sums_linearly():
    bw = SharedBandwidth(eff_BW_GBs=24.2, knee_GBs=27.0)
    assert abs(bw.aggregate_GBs(4, contention=False) - 4 * 24.2) < 1e-9


def test_default_knee_is_single_channel():
    # no knee given -> aggregate never exceeds one stream's eff_BW (Card on-card DRAM)
    bw = SharedBandwidth(eff_BW_GBs=24.2)
    assert abs(bw.aggregate_GBs(4) - 24.2) < 1e-9


def test_stream_us():
    bw = SharedBandwidth(eff_BW_GBs=10.0)             # 10 GB/s
    assert abs(bw.stream_us(10e9, 1) - 1e6) < 1e-3    # 10 GB / 10 GB/s = 1 s = 1e6 us
    assert bw.stream_us(0, 1) == 0.0


def test_compute_unit_defaults():
    u = ComputeUnit("cim")
    assert u.name == "cim" and u.busy_until == 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok {fn.__name__}")
    print(f"\n{len(fns)} resource tests passed.")
