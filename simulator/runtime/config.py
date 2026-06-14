"""SimConfig — the declarative user-input contract (D12, Phase 2.1).

One JSON experiment-config -> SimConfig dataclass -> runner.run(cfg). The
calibrated core (per-unit timing equations + the mem_lpddr4x 24.2 GB/s anchor)
is FIXED in code and is NOT a config field: any unknown key is rejected
fail-loud (you cannot override the calibrated core via the config). The
forward-looking layer (capacity, BW efficiency, topology, units, scheduler,
tunables, ablations) is freely settable; anything outside the calibrated
envelope (capacity > 16 GB, non-card topology, non-silicon backend, NPU) is
auto-tagged into `provenance` as extrapolated/simulated.

memory_capacity_GB is a FEASIBILITY gate (checked at runner.run(), which knows the
model footprint), NOT a throughput knob: decode tok/s is bandwidth-bound, so capacity
does not change it. A capacity below the model's resident weight footprint is rejected
fail-loud at run(); capacity-dependent behavior (residency/spill) is a later wave.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_KNOWN_TOP = {"workload", "platform", "scheduler", "tunables", "ablations", "sweep", "_doc"}
_KNOWN_WL = {"model", "task", "prefill_len", "decode_len", "context", "batch"}
_KNOWN_PLAT = {"memory_spec", "topology", "memory_capacity_GB", "bw_efficiency", "units", "engine"}
_KNOWN_SCHED = {"policy", "op_unit_overrides", "precision_boundary_placement"}
_KNOWN_TUN = {"knee_GBs", "interconnect_efficiency", "concurrency_overlap_factor", "pipeline"}
_KNOWN_ABL = {"concurrency_off", "contention_off", "compute_off"}

_CAL_MEMORY = "mem_lpddr4x"        # the measured 24.2 GB/s decode anchor
_CAL_TOPOLOGY = "cim_topo_card"    # the L4-anchored on-card-DRAM topology
_ENVELOPE_GB = 16                  # largest measured M.2 SKU
_SILICON_BACKENDS = {"analytic"}   # analytic CIM/CPU/GPU = silicon-calibrated; others = simulated


def _reject_unknown(d, known, where):
    extra = set(d) - known
    if extra:
        raise ValueError(
            f"SimConfig: unknown key(s) {sorted(extra)} in '{where}'. The calibrated core "
            f"(per-unit timing equations + mem_lpddr4x 24.2 anchor) is fixed in code and "
            f"cannot be overridden by the config (D12).")


def _default_units():
    return {"cim": True, "gpu": True, "npu": False, "cpu": True}


def _default_engine():
    return {"cim": "analytic", "memory": "analytic", "npu": "analytic"}


@dataclass
class SimConfig:
    model: str
    prefill_len: int = 256
    decode_len: int = 512
    context: int = 1024
    batch: int = 1
    task: str | None = None
    memory_spec: str = _CAL_MEMORY
    topology: str = _CAL_TOPOLOGY
    memory_capacity_GB: int = 16
    bw_efficiency: float | None = None
    units: dict = field(default_factory=_default_units)
    engine: dict = field(default_factory=_default_engine)
    scheduler: str = "all_cim"
    knee_GBs: float | None = None
    interconnect_efficiency: float = 1.0
    concurrency_overlap_factor: float = 1.0   # RESERVED for Wave 2.2 cross-unit overlap; no effect on the 2.1 serial path
    pipeline: bool = False                     # cross-op execution overlap. OFF (default) = single-accelerator
                                               # serial = the measured all-AIPU path (Metis 1c, SDK v1.3.1 exposes
                                               # no intra-frame pipeline). ON = SIMULATED forward-looking overlap.
    concurrency_off: bool = False
    contention_off: bool = False
    compute_off: bool = False                 # ablation: memory-only (zero unit compute) — isolates the non-circular compute correction
    sweep: dict | None = None
    provenance: list = field(default_factory=list)

    @staticmethod
    def from_json(path):
        return SimConfig.from_dict(json.loads(Path(path).read_text()))

    @staticmethod
    def from_dict(d):
        _reject_unknown(d, _KNOWN_TOP, "(top)")
        wl = d.get("workload", {})
        plat = d.get("platform", {})
        sched = d.get("scheduler", {})
        tun = d.get("tunables", {})
        abl = d.get("ablations", {})
        _reject_unknown(wl, _KNOWN_WL, "workload")
        _reject_unknown(plat, _KNOWN_PLAT, "platform")
        if isinstance(sched, dict):
            _reject_unknown(sched, _KNOWN_SCHED, "scheduler")
        _reject_unknown(tun, _KNOWN_TUN, "tunables")
        _reject_unknown(abl, _KNOWN_ABL, "ablations")
        if "model" not in wl:
            raise ValueError("SimConfig: workload.model is required")
        cfg = SimConfig(
            model=wl["model"], task=wl.get("task"),
            prefill_len=wl.get("prefill_len", 256), decode_len=wl.get("decode_len", 512),
            context=wl.get("context", 1024), batch=wl.get("batch", 1),
            memory_spec=plat.get("memory_spec", _CAL_MEMORY),
            topology=plat.get("topology", _CAL_TOPOLOGY),
            memory_capacity_GB=plat.get("memory_capacity_GB", 16),
            bw_efficiency=plat.get("bw_efficiency"),
            units=plat.get("units") or _default_units(),
            engine=plat.get("engine") or _default_engine(),
            scheduler=(sched.get("policy") if isinstance(sched, dict) else sched) or "all_cim",
            knee_GBs=tun.get("knee_GBs"),
            interconnect_efficiency=tun.get("interconnect_efficiency", 1.0),
            concurrency_overlap_factor=tun.get("concurrency_overlap_factor", 1.0),
            pipeline=tun.get("pipeline", False),
            concurrency_off=abl.get("concurrency_off", False),
            contention_off=abl.get("contention_off", False),
            compute_off=abl.get("compute_off", False),
            sweep=d.get("sweep"))
        if cfg.batch != 1:
            raise ValueError("SimConfig: v1 scope is batch=1 (hook reserved)")
        cfg._flag_provenance()
        return cfg

    def _flag_provenance(self):
        p = []
        if self.memory_capacity_GB > _ENVELOPE_GB:
            p.append(f"extrapolated: memory_capacity {self.memory_capacity_GB}GB > measured {_ENVELOPE_GB}GB")
        if self.topology != _CAL_TOPOLOGY:
            p.append(f"simulated: topology '{self.topology}' (not the L4-anchored {_CAL_TOPOLOGY})")
        if self.memory_spec != _CAL_MEMORY:
            p.append(f"simulated: memory_spec '{self.memory_spec}' (not the measured {_CAL_MEMORY} anchor)")
        if self.bw_efficiency is not None:
            p.append(f"simulated: bw_efficiency override = {self.bw_efficiency}")
        if self.pipeline:
            p.append("simulated: pipeline overlap enabled (cross-op double-buffering; the "
                     "measured all-AIPU 1c path exposes no intra-frame pipeline, SDK v1.3.1)")
        for u, e in self.engine.items():
            if e not in _SILICON_BACKENDS:
                p.append(f"simulated: {u} engine='{e}' (non-silicon backend)")
        if self.units.get("npu"):
            p.append("simulated: NPU enabled (no RKNPU2 silicon, #13)")
        self.provenance = p

    def is_calibrated_anchor(self):
        """True iff every input is within the silicon-calibrated envelope (no provenance flags)."""
        return not self.provenance
