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

from simulator.specs.loader import load_spec

_KNOWN_TOP = {"workload", "platform", "scheduler", "tunables", "ablations", "sweep", "_doc"}
_KNOWN_WL = {"model", "task", "prefill_len", "decode_len", "context", "batch"}
_KNOWN_PLAT = {"memory_spec", "topology", "memory_capacity_GB", "bw_efficiency", "units", "engine",
               "soc_profile", "cim_n_cores", "npu_scale", "cpu_gemv_efficiency", "cim_weight_bytes_per_param",
               "cim_weight_budget_frac"}
_KNOWN_SCHED = {"policy", "precision_boundary_placement", "sync_mode", "sync_us_override", "partition_grid",
                "host", "allow_cpu_compute", "prefill_chunks"}
_PREFILL_CHUNKS = (1, 2, 4, 8)                 # scheduler.prefill_chunks candidates (or "auto" = portfolio over them)
_KNOWN_TUN = {"knee_GBs", "interconnect_efficiency", "concurrency_overlap_factor", "pipeline"}
_KNOWN_ABL = {"concurrency_off", "contention_off", "compute_off"}

_CAL_MEMORY = "mem_lpddr4x"        # the measured 24.2 GB/s decode anchor
_CAL_TOPOLOGY = "cim_topo_card"    # the L4-anchored on-card-DRAM topology
_CAL_CIM_PARAMS = "simulator/models/params/m1_cim.json"  # the Metis-calibrated CIM compute fit (Alpha 13pt/Card)
_ENVELOPE_GB = 16                  # largest measured M.2 SKU
_SILICON_BACKENDS = {"analytic"}   # analytic CIM/CPU/GPU = silicon-calibrated; others = simulated

# Wave 2.3: the three wired topologies + their REQUIRED memory_spec (the single physics source is
# the topology spec; memory_spec is a consistency tag). None = no on-card DRAM (eff_BW from PCIe) ->
# reject any explicit LPDDR memory_spec as misleading. memory_spec defaults to a SENTINEL (None) so
# "omitted" is distinguishable from "explicitly set" (needed for the alpha+omitted case).
_VALID_TOPOLOGIES = {"cim_topo_card", "cim_topo_alpha", "cim_topo_edge"}
_TOPO_MEMORY = {"cim_topo_card": "mem_lpddr4x", "cim_topo_edge": "mem_lpddr5", "cim_topo_alpha": None}

# The wired scheduler names. runner.run() (which imports the actual SCHEDULERS registry) is the
# authoritative check; config cannot import it (layering), so this small frozenset lets the
# user-input contract reject an unknown/garbage scheduler at construction instead of silently
# provenance-labelling it as a real "simulated heterogeneous placement". Keep in sync with
# simulator/runtime/scheduler.py SCHEDULERS.
_KNOWN_SCHEDULERS = frozenset({"all_cim", "cim_hetero", "gpu_only", "npu_only", "cim_only", "cpu_only",
                               "heteroinfer", "tiled_hetero"})
_SYNC_MODES = ("fast", "clfinish", "none")   # mirror of tiling.SYNC_US (config cannot import tiling, layering)
_HOSTS = ("gpu", "npu", "cim", "auto")        # scheduler.host for tiled_hetero (None = scheduler default = gpu)
_CAL_CIM_N_CORES = 4                           # the Metis geometry the M1 CIM tile fit is calibrated at

# ADR-0008 SoC profiles (simulator/specs/soc_<name>.json). A profile fixes the CIM topology (its
# units.cim.topology) and supplies the memory wall; `topology` may be omitted (resolved from the
# profile) or must equal it. Keep in sync with simulator/runtime/platform.py SOC_PROFILES.
_VALID_SOC_PROFILES = frozenset({"sd8gen3_cim", "rk3588_cim"})


def _profile_topology(name):
    if name not in _VALID_SOC_PROFILES:
        raise ValueError(f"SimConfig: unknown soc_profile {name!r} (wired: {sorted(_VALID_SOC_PROFILES)})")
    return load_spec(f"soc_{name}")["units"]["cim"]["topology"]


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
    memory_spec: str | None = None      # SENTINEL: None = use the topology's required spec (resolved)
    topology: str = _CAL_TOPOLOGY
    memory_capacity_GB: int = 16
    bw_efficiency: float | None = None
    soc_profile: str | None = None      # ADR-0008: 'sd8gen3_cim' | 'rk3588_cim' (None = legacy unit set)
    units: dict = field(default_factory=_default_units)
    engine: dict = field(default_factory=_default_engine)
    scheduler: str = "all_cim"
    precision_boundary_placement: str = "consumer"   # which unit pays a conversion op (2.2b): consumer|producer
    sync_mode: str = "fast"                    # ADR-0009 cross-unit sync cost: fast (HeteroInfer predictable-wait
                                               # + polling: 10 us decode / 30 us prefill) | clfinish (400 us) | none (0)
    sync_us_override: float | None = None      # explicit per-boundary sync cost (us); wins over sync_mode
    partition_grid: int = 8                    # ADR-0009 solver ratio grid (fractions in multiples of 1/partition_grid)
    host: str | None = None                    # tiled_hetero host (graph-owning) unit: gpu|npu|cim|auto (None =
                                               # scheduler default: gpu if enabled else the first enabled unit);
                                               # 'auto' = solve once per candidate host, keep the lowest estimate
    allow_cpu_compute: bool | None = None      # tiled_hetero: let the solver place matmul/attention/support on the
                                               # CPU (None = scheduler default False = HeteroInfer's unit set;
                                               # True = "ours" with the CPU as a 4th/5th compute unit; SIMULATED)
    prefill_chunks: int | str = 1              # ADR-0009 cooperation revision C2: prefill token-chunk pipelining —
                                               # the prefill forward is built as C chunk sub-DAGs of P/C rows (causal
                                               # attention over the chunks already done) so the event engine overlaps
                                               # chunk c+1's GEMMs with chunk c's attention/support ops; 1 (default) =
                                               # the one-forward DAG as before; 2/4/8; "auto" = engine-ranked portfolio
                                               # over {1, 2, 4, 8}. SIMULATED (HeteroInfer §4.2.2 activation-centric
                                               # across the whole layer; no silicon). Decode untouched.
    cim_n_cores: int | None = None             # scalable axis: override of the profile's units.cim.n_cores
                                               # (None = profile / M1 default 4; any other value is EXTRAPOLATED)
    npu_scale: float = 1.0                     # scalable axis: multiplier on the NPU compute throughput (TOPS);
                                               # != 1 is EXTRAPOLATED (no silicon)
    cpu_gemv_efficiency: float | None = None   # sensitivity knob: CPU M=1 GEMV kernel efficiency (fraction of NEON
                                               # peak) overriding the profile's units.cpu.gemm.gemv_efficiency (P1
                                               # 0.217 = HeteroInfer Fig.12 ratio, an ASSUMPTION); a value above the
                                               # GEMM efficiency lifts both to it. Flagged `simulated` when overridden.
    cim_weight_budget_frac: float | None = None   # ADR-0009 cooperation revision C1: override of the profile's
                                               # units.cim.weight_budget_frac = the share of the CIM resident capacity
                                               # (knee x n_cores/4 weight elements) the capacity-aware weight split may
                                               # pin (None = profile, 1.0; 0 disables the capacity_split candidate).
                                               # Needs a soc_profile; flagged `simulated` when it differs from the profile.
    cim_weight_bytes_per_param: float | None = None   # sensitivity knob (round-2): streamed CIM weight bytes per
                                               # parameter overriding the profile's units.cim.weight_bytes_per_param
                                               # (1.0 = INT8 crossbar weights as stored; 0.5 = W4 storage unpacked
                                               # into the INT8 array on load — a DESIGN OPTION, no silicon). Needs a
                                               # soc_profile; flagged `simulated` when it differs from the profile.
    knee_GBs: float | None = None
    interconnect_efficiency: float = 1.0
    concurrency_overlap_factor: float = 1.0   # RESERVED for Wave 2.2 cross-unit overlap; not wired on the 2.1
                                               # serial path (a non-default value is rejected fail-loud in validate())
    pipeline: bool = False                     # cross-op execution overlap. OFF (default) = single-accelerator
                                               # serial = the measured all-AIPU path (Card 1c = single AIPU core;
                                               # measured tok/s at/below the serial no-overlap bound, 4c/1c ~1.1x).
                                               # ON = SIMULATED forward-looking overlap.
    concurrency_off: bool = False
    contention_off: bool = False
    compute_off: bool = False                 # ablation: memory-only (zero unit compute) — isolates the non-circular compute correction
    sweep: dict | None = None
    provenance: list = field(default_factory=list)

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Re-check invariants + refresh provenance. Called at construction AND at the run() public
        boundary — SimConfig is a MUTABLE dataclass, so a field broken after construction (e.g.
        cfg.batch = 4) must still fail loud, and provenance must reflect the config as actually run."""
        if self.batch != 1:
            raise ValueError("SimConfig: v1 scope is batch=1 (hook reserved)")
        if self.prefill_len < 1 or self.decode_len < 1 or self.context < 1:
            raise ValueError(
                f"SimConfig: prefill_len/decode_len/context must each be >= 1, got "
                f"prefill_len={self.prefill_len!r}, decode_len={self.decode_len!r}, "
                f"context={self.context!r} (a degenerate length yields an internally "
                f"inconsistent metrics dict — the runner clamps to 1 but echoes the raw value).")
        if self.prefill_len > self.context:
            raise ValueError(
                f"SimConfig: prefill_len ({self.prefill_len}) must not exceed context "
                f"({self.context}).")
        if not isinstance(self.scheduler, str):
            raise ValueError(f"SimConfig: scheduler must be a str, got {type(self.scheduler).__name__}")
        if self.scheduler not in _KNOWN_SCHEDULERS:
            raise ValueError(f"SimConfig: unknown scheduler {self.scheduler!r} "
                             f"(wired: {sorted(_KNOWN_SCHEDULERS)})")
        if self.precision_boundary_placement not in ("consumer", "producer"):
            raise ValueError(f"SimConfig: precision_boundary_placement must be 'consumer' or "
                             f"'producer', got {self.precision_boundary_placement!r}")
        if self.sync_mode not in _SYNC_MODES:
            raise ValueError(f"SimConfig: sync_mode must be one of {_SYNC_MODES}, got {self.sync_mode!r}")
        if self.sync_us_override is not None and self.sync_us_override < 0:
            raise ValueError(f"SimConfig: sync_us_override must be >= 0, got {self.sync_us_override!r}")
        if not isinstance(self.partition_grid, int) or self.partition_grid < 2:
            raise ValueError(f"SimConfig: partition_grid must be an int >= 2, got {self.partition_grid!r}")
        if self.host is not None and self.host not in _HOSTS:
            raise ValueError(f"SimConfig: scheduler.host must be one of {_HOSTS} (or omitted), got {self.host!r}")
        pc = self.prefill_chunks
        if not (pc == "auto" or (isinstance(pc, int) and not isinstance(pc, bool) and pc in _PREFILL_CHUNKS)):
            raise ValueError(f"SimConfig: scheduler.prefill_chunks must be one of {_PREFILL_CHUNKS} or 'auto', "
                             f"got {pc!r}")
        if self.cim_n_cores is not None and (isinstance(self.cim_n_cores, bool)
                                             or not isinstance(self.cim_n_cores, int) or self.cim_n_cores < 1):
            raise ValueError(f"SimConfig: platform.cim_n_cores must be an int >= 1, got {self.cim_n_cores!r}")
        if not isinstance(self.npu_scale, (int, float)) or isinstance(self.npu_scale, bool) or self.npu_scale <= 0:
            raise ValueError(f"SimConfig: platform.npu_scale must be a number > 0, got {self.npu_scale!r}")
        if self.cpu_gemv_efficiency is not None and (
                not isinstance(self.cpu_gemv_efficiency, (int, float)) or isinstance(self.cpu_gemv_efficiency, bool)
                or not (0 < self.cpu_gemv_efficiency <= 1)):
            raise ValueError(f"SimConfig: platform.cpu_gemv_efficiency must be a number in (0, 1], "
                             f"got {self.cpu_gemv_efficiency!r}")
        if self.cim_weight_bytes_per_param is not None:
            v = self.cim_weight_bytes_per_param
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not (0 < v <= 4):
                raise ValueError(f"SimConfig: platform.cim_weight_bytes_per_param must be a number in (0, 4], "
                                 f"got {v!r}")
            if self.soc_profile is None:
                raise ValueError("SimConfig: platform.cim_weight_bytes_per_param needs a soc_profile (the legacy "
                                 "path meters op_profile's INT8 bytes on every unit, byte-identical L4)")
        if self.cim_weight_budget_frac is not None:
            v = self.cim_weight_budget_frac
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not (0 <= v <= 1):
                raise ValueError(f"SimConfig: platform.cim_weight_budget_frac must be a number in [0, 1], got {v!r}")
            if self.soc_profile is None:
                raise ValueError("SimConfig: platform.cim_weight_budget_frac needs a soc_profile (the legacy path has "
                                 "no CIM residency model)")
        if self.concurrency_overlap_factor != 1.0:
            raise ValueError(
                f"SimConfig: concurrency_overlap_factor is RESERVED for Wave 2.2 cross-unit overlap "
                f"and is not wired on the 2.1 serial path; only the default 1.0 is accepted "
                f"(got {self.concurrency_overlap_factor!r}).")
        # Wave 2.3 topology x memory_spec contract (fail-loud at construction AND at run()):
        if self.topology not in _VALID_TOPOLOGIES:
            raise ValueError(f"SimConfig: unknown topology {self.topology!r} "
                             f"(wired: {sorted(_VALID_TOPOLOGIES)})")
        required = _TOPO_MEMORY[self.topology]
        if self.memory_spec is not None and self.memory_spec != required:
            raise ValueError(
                f"SimConfig: topology {self.topology!r} requires memory_spec "
                f"{required!r} but got {self.memory_spec!r}. The topology spec is the single "
                f"bandwidth source; memory_spec is a consistency tag (omit it to use the "
                f"topology default). cim_topo_alpha has no on-card DRAM so it rejects any LPDDR spec.")
        if self.bw_efficiency is not None and self.bw_efficiency <= 0:
            raise ValueError(
                f"SimConfig: bw_efficiency is a peak-scaling sensitivity knob and must be > 0 "
                f"(a non-physical effective bandwidth would result), got {self.bw_efficiency!r}.")
        if self.interconnect_efficiency <= 0:
            raise ValueError(
                f"SimConfig: interconnect_efficiency must be > 0 (a non-physical effective "
                f"bandwidth would result), got {self.interconnect_efficiency!r}. "
                f"(Mirrors the SharedBandwidth guard, resources.py.)")
        if self.knee_GBs is not None and self.knee_GBs <= 0:
            raise ValueError(
                f"SimConfig: knee_GBs is the saturation knee and must be > 0, "
                f"got {self.knee_GBs!r}. (Mirrors the SharedBandwidth guard, resources.py.)")
        # ADR-0008 soc_profile contract: a known profile; its CIM topology is the only valid `topology`
        # (the profile is the single SoC source); bw_efficiency (card-only peak scaling) is rejected.
        if self.soc_profile is not None:
            ptopo = _profile_topology(self.soc_profile)
            if self.topology != ptopo:
                raise ValueError(
                    f"SimConfig: soc_profile {self.soc_profile!r} fixes the CIM topology to {ptopo!r} "
                    f"but got topology {self.topology!r}; omit platform.topology (resolved from the "
                    f"profile) or set it to the profile's own.")
            if self.bw_efficiency is not None:
                raise ValueError(
                    f"SimConfig: bw_efficiency cannot combine with soc_profile {self.soc_profile!r} "
                    f"(the profile's memory block is the single bandwidth source).")
        if self.bw_efficiency is not None and self.topology != _CAL_TOPOLOGY:
            raise ValueError(
                f"SimConfig: bw_efficiency is a card-only peak-scaling knob; topology "
                f"{self.topology!r} has its own physics-derived effective wall (e.g. edge = LPDDR5 x "
                f"noc_efficiency, alpha = PCIe) that peak-scaling would silently bypass. Sweep eff via "
                f"the topology spec, not bw_efficiency. (Mirrors the Platform guard, #63.)")
        self._flag_provenance()
        return self

    def resolved_memory_spec(self):
        """The concrete memory spec for this topology: the explicit value if given, else the
        topology default (None for alpha — no on-card DRAM; eff_BW comes from the PCIe wall)."""
        return self.memory_spec if self.memory_spec is not None else _TOPO_MEMORY[self.topology]

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
        u = plat.get("units")
        eng = plat.get("engine")
        if u is not None:
            _reject_unknown(u, set(_default_units()), "platform.units")
        if eng is not None:
            _reject_unknown(eng, set(_default_engine()), "platform.engine")
        if "model" not in wl:
            raise ValueError("SimConfig: workload.model is required")
        # None (omitted/null) -> default; any explicit value (incl. "" garbage) reaches validate().
        _sched_pol = sched.get("policy") if isinstance(sched, dict) else sched
        # ADR-0008: an omitted topology is resolved from the soc_profile (an explicit one is checked
        # against it in validate()); the profile name itself is validated there too.
        _prof = plat.get("soc_profile")
        _topo = plat.get("topology", _profile_topology(_prof) if _prof is not None else _CAL_TOPOLOGY)
        cfg = SimConfig(
            model=wl["model"], task=wl.get("task"),
            prefill_len=wl.get("prefill_len", 256), decode_len=wl.get("decode_len", 512),
            context=wl.get("context", 1024), batch=wl.get("batch", 1),
            memory_spec=plat.get("memory_spec"),   # None sentinel -> resolved per topology
            topology=_topo,
            memory_capacity_GB=plat.get("memory_capacity_GB", 16),
            bw_efficiency=plat.get("bw_efficiency"),
            soc_profile=_prof,
            cim_n_cores=plat.get("cim_n_cores"),
            npu_scale=plat.get("npu_scale", 1.0),
            cpu_gemv_efficiency=plat.get("cpu_gemv_efficiency"),
            cim_weight_bytes_per_param=plat.get("cim_weight_bytes_per_param"),
            cim_weight_budget_frac=plat.get("cim_weight_budget_frac"),
            units=_default_units() if u is None else u,
            engine=_default_engine() if eng is None else eng,
            scheduler="all_cim" if _sched_pol is None else _sched_pol,
            precision_boundary_placement=(sched.get("precision_boundary_placement", "consumer")
                                          if isinstance(sched, dict) else "consumer"),
            sync_mode=(sched.get("sync_mode", "fast") if isinstance(sched, dict) else "fast"),
            sync_us_override=(sched.get("sync_us_override") if isinstance(sched, dict) else None),
            partition_grid=(sched.get("partition_grid", 8) if isinstance(sched, dict) else 8),
            host=(sched.get("host") if isinstance(sched, dict) else None),
            allow_cpu_compute=(sched.get("allow_cpu_compute") if isinstance(sched, dict) else None),
            prefill_chunks=(sched.get("prefill_chunks", 1) if isinstance(sched, dict) else 1),
            knee_GBs=tun.get("knee_GBs"),
            interconnect_efficiency=tun.get("interconnect_efficiency", 1.0),
            concurrency_overlap_factor=tun.get("concurrency_overlap_factor", 1.0),
            pipeline=tun.get("pipeline", False),
            concurrency_off=abl.get("concurrency_off", False),
            contention_off=abl.get("contention_off", False),
            compute_off=abl.get("compute_off", False),
            sweep=d.get("sweep"))   # SimConfig.__post_init__ validates invariants + flags provenance
        return cfg

    def _flag_provenance(self):
        p = []
        if self.memory_capacity_GB > _ENVELOPE_GB:
            p.append(f"extrapolated: memory_capacity {self.memory_capacity_GB}GB > measured {_ENVELOPE_GB}GB")
        if self.topology == "cim_topo_alpha":
            p.append("counterfactual: topology 'cim_topo_alpha' (Metis Alpha is LLM-incapable — "
                     "-1301 closed firmware + no on-card DRAM; a decode number here is a "
                     "counterfactual host-PCIe-streaming estimate, NOT a runnable config)")
        elif self.topology != _CAL_TOPOLOGY:
            p.append(f"simulated: topology '{self.topology}' (not the L4-anchored {_CAL_TOPOLOGY})")
        # CIM COMPUTE engine provenance: a topology may swap the accelerator geometry via
        # cim_compute_params. Exact-string compare (a differently-formatted path reads as non-Metis —
        # the conservative direction). Non-Metis params = a different fit with NO silicon -> `simulated`.
        _ccp = load_spec(self.topology).get("cim_compute_params")
        if _ccp is not None and _ccp != _CAL_CIM_PARAMS:
            p.append(f"simulated: CIM compute params '{_ccp}' (non-Metis geometry, no silicon ground "
                     f"truth; not the Metis-calibrated {_CAL_CIM_PARAMS})")
        rms = self.resolved_memory_spec()
        if rms is not None and rms != _CAL_MEMORY:
            p.append(f"simulated: memory_spec '{rms}' (not the measured {_CAL_MEMORY} anchor)")
        if self.soc_profile is not None:
            # the profile's own honesty line (borrowed for sd8gen3_cim, simulated for rk3588_cim) + the
            # fact that its memory block overrides the topology-resolved wall (ADR-0008 precedence).
            p.append(load_spec(f"soc_{self.soc_profile}")["honesty"]
                     + "; memory wall = the profile's single_stream_eff_BW/knee (overrides the topology's)")
        if self.bw_efficiency is not None:
            p.append(f"simulated: bw_efficiency override = {self.bw_efficiency}")
        # scalable axes (ADR-0008 "n_cores = scalable axis"): the M1 CIM fit is calibrated at the Metis
        # 4-core geometry and the NPU models at their anchored TOPS; any other value is off-anchor.
        if self.cim_n_cores is not None:
            _default_cores = (load_spec(f"soc_{self.soc_profile}")["units"]["cim"]["n_cores"]
                              if self.soc_profile is not None else _CAL_CIM_N_CORES)
            if self.cim_n_cores != _default_cores:
                p.append(f"extrapolated: cim_n_cores override = {self.cim_n_cores} (M1 CIM tile fit calibrated "
                         f"at n_cores={_default_cores}; other core counts scale the tile width, no silicon)")
        if self.npu_scale != 1.0:
            p.append(f"extrapolated: npu_scale = {self.npu_scale} (NPU compute throughput multiplier; "
                     f"no silicon at that scale)")
        if self.cpu_gemv_efficiency is not None:
            _g = (load_spec(f"soc_{self.soc_profile}") if self.soc_profile is not None
                  else load_spec("soc_rk3588_cim"))["units"]["cpu"]["gemm"]
            _default_eff = _g.get("gemv_efficiency", _g["efficiency"])
            if self.cpu_gemv_efficiency != _default_eff:
                p.append(f"simulated: cpu_gemv_efficiency override = {self.cpu_gemv_efficiency} (CPU M=1 GEMV kernel "
                         f"efficiency; profile value {_default_eff} is itself an assumption anchored on HeteroInfer "
                         f"Fig.12, no CPU GEMM silicon)")
        if self.cim_weight_budget_frac is not None and self.soc_profile is not None:
            _default_b = float(load_spec(f"soc_{self.soc_profile}")["units"]["cim"].get("weight_budget_frac", 1.0))
            if float(self.cim_weight_budget_frac) != _default_b:
                p.append(f"simulated: cim_weight_budget_frac override = {self.cim_weight_budget_frac} (share of the CIM "
                         f"resident capacity the capacity-aware weight split may pin; profile {_default_b}; ADR-0009 "
                         f"cooperation revision C1, no silicon)")
        if self.cim_weight_bytes_per_param is not None and self.soc_profile is not None:
            _default_w = float(load_spec(f"soc_{self.soc_profile}")["units"]["cim"].get("weight_bytes_per_param", 1.0))
            if float(self.cim_weight_bytes_per_param) != _default_w:
                p.append(f"simulated: cim_weight_bytes_per_param override = {self.cim_weight_bytes_per_param} "
                         f"(streamed CIM weight bytes per parameter; profile {_default_w} = INT8 crossbar weights as "
                         f"stored; 0.5 = W4 storage unpacked into the INT8 array on load, a design option with no "
                         f"silicon)")
        # knee_GBs / interconnect_efficiency are forward-looking sweep knobs that DO alter the decode
        # bandwidth wall (they feed SharedBandwidth) but have no concurrent-unit silicon (#52) -> a
        # non-default value is off-anchor and must be flagged, else a swept result reads as calibrated.
        if self.knee_GBs is not None:
            p.append(f"simulated: knee_GBs override = {self.knee_GBs} (contention saturation knob; "
                     f"no concurrent-unit silicon, #52)")
        if self.interconnect_efficiency != 1.0:
            p.append(f"simulated: interconnect_efficiency = {self.interconnect_efficiency} "
                     f"(forward-looking NoC de-rate knob; not silicon-measured)")
        if self.pipeline:
            p.append("simulated: pipeline overlap enabled (cross-op double-buffering; the "
                     "measured all-AIPU Card 1c single-core decode shows no cross-op overlap)")
        if self.scheduler != "all_cim":
            p.append(f"simulated: '{self.scheduler}' heterogeneous placement (only AllCim is "
                     f"silicon-gated; no concurrent-unit silicon — Aetina out for repair, #52)")
        if self.prefill_chunks != 1:
            p.append(f"simulated: prefill_chunks = {self.prefill_chunks!r} (prefill token-chunk pipelining across "
                     f"units, HeteroInfer §4.2.2 activation-centric applied to the whole layer; the C=1 forward "
                     f"prices dense attention while a chunked one prices block-causal attention; no silicon)")
        if self.sync_mode != "fast" or self.sync_us_override is not None:
            p.append(f"simulated: sync_mode={self.sync_mode!r} sync_us_override={self.sync_us_override!r} "
                     f"(cross-unit sync cost; literature-anchored, no heterogeneous-sync silicon)")
        for u, e in self.engine.items():
            if e not in _SILICON_BACKENDS:
                p.append(f"simulated: {u} engine='{e}' (non-silicon backend)")
        if self.units.get("npu"):
            p.append("simulated: NPU enabled (no RKNPU2 silicon, #13)")
        if self.compute_off:
            p.append("counterfactual: compute_off memory-only ablation (zero unit compute; isolates "
                     "the non-circular CIM-compute correction — must NOT be read as a calibrated anchor)")
        if self.concurrency_off:
            p.append("ablation: concurrency_off (forced-serial counterfactual; not a calibrated anchor)")
        if self.contention_off:
            p.append("ablation: contention_off (no-knee linear-BW counterfactual; not a calibrated anchor)")
        self.provenance = p

    def is_calibrated_anchor(self):
        """True iff every input is within the silicon-calibrated envelope (no provenance flags)."""
        return not self.provenance
