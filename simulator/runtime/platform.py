"""Platform — the simulated SoC: a set of Phase-1 unit models + a shared memory
channel (Phase 2.1). `compute_us(node)` prices an op's COMPUTE on its assigned
unit via the frozen Phase-1 models; the op's MEMORY traffic (node.bytes_streamed)
is metered separately by the M3 engine through the SharedBandwidth. The engine
takes max(compute, memory) per node.

Calibrated core is FIXED here, not in the user config (D12): the per-unit timing
equations (CimTileModel/CpuModel/MaliGpuModel params) and the mem_lpddr4x 24.2
GB/s anchor are loaded from committed params/specs and cannot be overridden by
SimConfig — only the forward-looking knee / interconnect / bw_efficiency /
capacity / topology are user-settable.
"""
from __future__ import annotations

from simulator.specs.loader import load_spec
from simulator.models.engine import Workload
from simulator.models.m1_cim_tile import CimTileModel
from simulator.models.m4_cpu import CpuModel
from simulator.models.m4_gpu import MaliGpuModel
from simulator.models.m7_energy import EnergyModel
from simulator.runtime.resources import SharedBandwidth

# op category -> the calibrated CPU support-op name (m4_cpu overhead_op_us keys)
_CPU_OP = {"norm": "rmsnorm", "rope": "rope_apply", "ffn": "swiglu",
           "softmax": "softmax", "residual": "residual"}


class Platform:
    """Bind unit models + a shared memory channel for one model. `knee_GBs`,
    `interconnect_efficiency`, `bw_efficiency` are the forward-looking tunables
    (SIMULATED); `memory_spec` selects the committed memory anchor."""

    def __init__(self, model, *, memory_spec="mem_lpddr4x", knee_GBs=None,
                 interconnect_efficiency=1.0, bw_efficiency=None):
        self.model = model
        mem = load_spec(memory_spec)
        eff = float(mem["eff_BW_GBs"])                       # measured anchor (e.g. 24.2)
        if bw_efficiency is not None:                        # forward-looking override (sweep knob)
            peak = mem.get("peak_BW_GBs") or (eff / float(mem.get("efficiency", 1.0)))
            eff = float(peak) * float(bw_efficiency)
        self.bw = SharedBandwidth(eff, knee_GBs=knee_GBs,
                                  interconnect_efficiency=interconnect_efficiency)
        self.cim = CimTileModel()
        self.cpu = CpuModel(load_spec("cpu_rk3588"))
        self.gpu = MaliGpuModel()
        self.energy = EnergyModel()

    @property
    def mem_domains(self):
        """The two memory pools the residency rule (R7) routes ops to: the shared `dram`
        (the measured 24.2 GB/s on-card LPDDR4x — the Metis Card's 16 GiB device DRAM, where
        weights/KV/embedding are resident, metered by the M3 engine) and `cpu_cache` (the A76
        tiered on-chip cache, priced INSIDE m4_cpu for CPU-support ops — a labelled proxy for
        the all-AIPU Card's on-chip support; NEVER the DRAM pool, which removes the S-dc
        double-count)."""
        return {"dram": self.bw.eff_BW, "cpu_cache": self.cpu.spec["cache_bw_GBs"]}

    def price(self, node):
        """Price one op's COMPUTE on its assigned unit (#55 per-op provenance). Returns
        {latency_us, compute_provenance, source_model} — which Phase-1 unit model produced
        the latency and a human-readable provenance string. latency_us=0.0 = no compute model
        (pure-memory op; cost is its bytes_streamed via the engine). NB the final compute-vs-
        memory BOUND is the M3 engine's max(compute, dram_memory) decision, not this method's."""
        u, cat, wl = node.unit, node.category, node.wl
        if u not in ("cim", "cpu", "gpu", "mem"):
            raise ValueError(f"M3 platform: unknown/unpriced unit {u!r} "
                             f"(valid: cim/cpu/gpu/mem; npu pricing is Wave 2.2)")
        if u == "cim":
            if cat == "matmul" and wl.K and wl.N:
                lat = self.cim.dev_lat_us(wl.M, wl.K, wl.N)
                prov = "CIM-GEMV dev_lat (M1 tile model, Alpha-calibrated)"
                if self.cim.is_extrapolated(wl.K, wl.N):
                    prov += "; EXTRAPOLATED beyond native envelope"
                return {"latency_us": float(lat), "compute_provenance": prov, "source_model": "m1_cim_tile"}
            return {"latency_us": 0.0, "source_model": "none",
                    "compute_provenance": "CIM attention/elementwise: memory-bound (no compute model in 2.1; cost = bytes)"}
        if u == "cpu":
            op = _CPU_OP.get(cat)
            if op is None:
                return {"latency_us": 0.0, "source_model": "none",
                        "compute_provenance": f"no CPU compute model for category '{cat}'"}
            wl2 = Workload(op=("softmax" if op.startswith("softmax") else op), kv=(wl.kv or 0),
                           extra={"model": self.model, "dtype": "fp32"})
            r = self.cpu.predict(wl2)
            return {"latency_us": r["latency_us"], "compute_provenance": r["provenance"], "source_model": "m4_cpu"}
        if u == "gpu" and cat == "attention":
            lat = self.gpu.attn_bmm_us(wl.kv or 1, heads=wl.heads, layers=1)
            return {"latency_us": float(lat), "compute_provenance": "Mali GPU attn_bmm (m4_gpu)", "source_model": "m4_gpu"}
        return {"latency_us": 0.0, "source_model": "none",
                "compute_provenance": "mem/unmodeled (cost = bytes via the DRAM pool)"}

    def compute_us(self, node):
        """Compute latency (us) of one op on its assigned unit. 0.0 = no compute model
        (pure-memory op; cost is its bytes_streamed via the engine). Delegates to price()."""
        return self.price(node)["latency_us"]

    def energy_J(self, node, compute_us):
        """Per-op energy estimate (J). CIM matmul from flops; memory from bytes;
        CPU support from active compute time. ESTIMATED (M7, +/-20% band)."""
        cat, wl = node.category, node.wl
        e = self.energy.dram_J(node.bytes_streamed) if node.bytes_streamed else 0.0
        if node.unit == "cim" and cat == "matmul":
            e += self.energy.cim_J(2 * wl.M * wl.K * wl.N)
        elif node.unit == "cpu" and compute_us > 0:
            e += self.energy.cpu_J(compute_us)
        return e
