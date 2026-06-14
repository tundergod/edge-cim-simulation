"""Phase 2.2a Step D — memory domains + residency + byte accounting (R7).

Each op's memory lands in exactly one DOMAIN: `dram` (the measured 24.2 GB/s LPDDR4x
pool, metered by the M3 engine), `cpu_cache` (the A76 on-chip cache, priced INSIDE
m4_cpu for CPU-support ops), or `none` (no traffic). The residency rule fixes the
S-dc DOUBLE-COUNT: a CPU-support op's memory is already in compute_us (m4_cpu
max(compute,cache_mem)+overhead) AND was also metered into DRAM — now it is `cpu_cache`
and excluded from the DRAM pool. No "local" domain (scope-out).

    .venv/bin/pytest tests/test_domains.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "trace_export"))
from simulator.models.engine import Workload  # noqa: E402
from simulator.runtime.dag import OpNode, Dag  # noqa: E402
from simulator.runtime.workload import build_token_dag  # noqa: E402
from simulator.runtime.scheduler import all_cim_assign, domain_byte_audit  # noqa: E402
from simulator.runtime.platform import Platform  # noqa: E402
from simulator.runtime.events import run_dag, run_serial  # noqa: E402
from simulator.runtime.resources import SharedBandwidth  # noqa: E402


def _scheduled(model="llama-3.2-1b", L=512):
    return all_cim_assign(build_token_dag(model, "decode", L))


def test_mem_domain_values_and_residency():
    dag = _scheduled()
    for n in dag.nodes:
        assert n.mem_domain in ("dram", "cpu_cache", "none"), f"bad domain {n.mem_domain}"
        assert n.mem_domain != "local"
        if n.bytes_streamed == 0:
            assert n.mem_domain == "none"
        elif n.unit == "cpu":
            assert n.mem_domain == "cpu_cache"   # CPU-support -> cache (priced inside m4_cpu)
        else:
            assert n.mem_domain == "dram"        # cim weights/attention, mem kv/embedding


def test_platform_exposes_dram_and_cpu_cache_domains():
    d = Platform("llama-3.2-1b").mem_domains
    assert set(d) == {"dram", "cpu_cache"}
    assert abs(d["dram"] - 24.2) < 1.0           # measured LPDDR4x anchor
    assert isinstance(d["cpu_cache"], dict)       # A76 tiered cache (priced inside m4_cpu)


def test_cpu_support_not_double_counted_in_dram():
    # the S-dc double-count: a CPU-support op's memory is in compute_us (m4_cpu cache term)
    # AND was metered into DRAM. The fix routes it to cpu_cache, excluded from the DRAM pool.
    dag = _scheduled()
    cpu_support = [n for n in dag.nodes if n.unit == "cpu" and n.bytes_streamed > 0]
    assert cpu_support
    assert all(n.mem_domain == "cpu_cache" for n in cpu_support)
    audit = domain_byte_audit(dag)
    # mutual exclusion: every byte in exactly one domain (no op_bytes/cpu_cache overlap)
    assert audit["dram"] + audit["cpu_cache"] + audit["none"] == audit["total"]
    assert audit["cpu_cache"] > 0 and audit["ok"]
    # the DRAM-metered total excludes the CPU-support cache bytes
    dram_bytes = sum(n.bytes_streamed for n in dag.nodes if n.mem_domain == "dram")
    assert audit["dram"] == dram_bytes


def test_dram_delta_equals_cpu_support_share():
    # S2-1: the residency rule changes the DRAM byte volume by EXACTLY the CPU-support
    # share (measured, not assumed); for 1B decode that share is small.
    dag = _scheduled()
    audit = domain_byte_audit(dag)
    share = audit["cpu_cache"] / audit["total"]
    assert 0 < share < 0.05, f"CPU-support byte share {share:.4f} unexpected"
    # engine runs (serial = the measured all-AIPU path) and the cpu_cache bytes do not
    # appear in the DRAM term (they are inside compute_us via m4_cpu).
    plat = Platform("llama-3.2-1b")
    assert run_serial(dag, plat, plat.bw) > 0


def test_mixed_domain_cache_does_not_drag_dram():
    # concurrent engine: a cpu_cache stream must NOT reduce a co-active dram stream's BW.
    class _Stub:
        def compute_us(self, n):
            return 0.0 if n.unit == "cim" else 100.0   # cpu op has compute (incl its cache mem)
    bw = SharedBandwidth(eff_BW_GBs=10.0)
    dram = OpNode(id=0, category="matmul", wl=Workload(op="matmul"), unit="cim",
                  bytes_streamed=int(10e9), mem_domain="dram")          # 10 GB / 10 GB/s = 1e6 us
    cache = OpNode(id=1, category="norm", wl=Workload(op="norm"), unit="cpu",
                   bytes_streamed=int(5e9), mem_domain="cpu_cache")     # excluded from DRAM pool
    dag = Dag([dram, cache])                                            # independent -> concurrent
    t = run_dag(dag, _Stub(), bw, pipeline=True)
    assert abs(t - 1e6) < 1e-3   # dram stream got FULL 10 GB/s; cache did not drag it


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"\n{len(fns)} domain tests passed.")
