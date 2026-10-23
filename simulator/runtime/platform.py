"""Platform — the simulated SoC: a set of Phase-1 unit models + a shared memory
channel (Phase 2.1; heterogeneous unit pricing + SoC profiles in ADR-0008). `price(node)`
prices an op's COMPUTE on its assigned unit; the op's MEMORY traffic (node.bytes_streamed)
is metered separately by the M3 engine through the SharedBandwidth. The engine takes
max(compute, memory) per node.

Calibrated core is FIXED here, not in the user config (D12): the per-unit timing
equations (CimTileModel/CpuModel/MaliGpuModel params) and the mem_lpddr4x 24.2
GB/s anchor are loaded from committed params/specs and cannot be overridden by
SimConfig — only the forward-looking knee / interconnect / bw_efficiency /
capacity / topology are user-settable.

SoC PROFILES (ADR-0008): `soc_profile` ('sd8gen3_cim' | 'rk3588_cim', simulator/specs/soc_*.json)
selects the per-unit models for ALL four units (gpu/npu/cpu/cim) and the shared-memory model
(single-stream eff BW + aggregate knee). Precedence: the profile fixes the CIM topology (a
conflicting `topology` is rejected), its memory block OVERRIDES the topology's resolved eff_BW,
`knee_GBs` comes from the profile unless explicitly given (sensitivity knob), and `bw_efficiency`
is rejected. With soc_profile UNSET every pre-existing (unit, category) combination is priced
BYTE-IDENTICALLY to Phase 2.5 (the L4 gate); only the combinations that used to raise or were
unreachable (gpu/npu GEMM + support ops, cpu GEMM/sampling, npu anything) pick up the
`rk3588_cim` profile's unit parameters so a heterogeneous scheduler can run without a profile.
"""
from __future__ import annotations

import json
from pathlib import Path

from simulator.specs.loader import load_spec
from simulator.models.engine import Workload
from simulator.models.m1_cim_tile import CimTileModel, CimAttentionModel
from simulator.models.m4_cpu import CpuModel, CpuGemmModel
from simulator.models.m4_gpu import MaliGpuModel
from simulator.models.m4_gpu_anchor import GpuAnchorRooflineModel
from simulator.models.m4_gpu_roofline import GpuRooflineModel
from simulator.models.m4_npu import NpuModel, NpuStaircaseModel
from simulator.models.m7_energy import EnergyModel
from simulator.runtime.resources import SharedBandwidth

# op category -> the calibrated CPU support-op name (m4_cpu overhead_op_us keys)
_CPU_OP = {"norm": "rmsnorm", "rope": "rope_apply", "ffn": "swiglu",
           "softmax": "softmax", "residual": "residual", "sampling": "sampling_argmax"}

SOC_PROFILES = ("sd8gen3_cim", "rk3588_cim")
_LEGACY_UNITS_PROFILE = "rk3588_cim"   # unit params for the NEW combos when soc_profile is unset

# The (unit, category) support matrix — supports() and price() agree by construction. 'merge'
# (a scheduler's split/merge + sync node) and 'convert' (precision cast) are priceable on every unit.
_SUPPORTED = {
    "cim": frozenset({"matmul", "attention", "convert", "merge"}),
    "gpu": frozenset({"matmul", "attention", "norm", "rope", "ffn", "softmax", "residual", "convert", "merge"}),
    "npu": frozenset({"matmul", "attention", "norm", "rope", "ffn", "softmax", "residual", "convert", "merge"}),
    "cpu": frozenset({"matmul", "attention", "norm", "rope", "ffn", "softmax", "residual", "sampling",
                      "convert", "merge"}),
    "mem": frozenset({"embedding", "kv_cache", "convert", "merge"}),
}


def load_soc_profile(name):
    """simulator/specs/soc_<name>.json, fail-loud on an unknown name."""
    if name not in SOC_PROFILES:
        raise ValueError(f"unknown soc_profile {name!r} (wired: {list(SOC_PROFILES)})")
    return load_spec(f"soc_{name}")


class Platform:
    """Bind unit models + a shared memory channel for one model. `knee_GBs`,
    `interconnect_efficiency`, `bw_efficiency` are the forward-looking tunables
    (SIMULATED); `memory_spec` selects the committed memory anchor; `soc_profile`
    selects a whole SoC (ADR-0008)."""

    def __init__(self, model, *, memory_spec="mem_lpddr4x", topology=None, knee_GBs=None,
                 interconnect_efficiency=1.0, bw_efficiency=None, soc_profile=None,
                 cim_n_cores=None, npu_scale=1.0):
        """`cim_n_cores` / `npu_scale` are the ADR-0008 scalable axes (None / 1.0 = the profile's own
        geometry; anything else is EXTRAPOLATED and flagged by SimConfig)."""
        self.model = model
        self.profile_name = soc_profile
        self.profile = None
        if soc_profile is not None:
            self.profile = load_soc_profile(soc_profile)
            ptopo = self.profile["units"]["cim"]["topology"]
            if topology is not None and topology != ptopo:
                raise ValueError(
                    f"Platform: soc_profile {soc_profile!r} fixes the CIM topology to {ptopo!r} "
                    f"(got topology={topology!r}); omit topology or pass the profile's own.")
            topology = ptopo
            if bw_efficiency is not None:
                raise ValueError(
                    f"Platform: bw_efficiency is a card-only knob and cannot combine with soc_profile "
                    f"{soc_profile!r} (the profile's memory block is the single bandwidth source).")
        # Single bandwidth source = the topology spec (via the existing M2 MemoryModel resolver):
        # card -> on-card LPDDR4x 24.2 (literal, byte-identical); alpha -> PCIe 3.9 (no on-card DRAM);
        # edge -> mem_lpddr5 eff x noc_efficiency. The per-call floor (alpha 911.1 us) is exposed for
        # the runner to add to TTFT only. When no topology is given, fall back to the bare memory_spec.
        # CIM COMPUTE engine params: resolved from the topology's `cim_compute_params` (repo-root path);
        # all three wired topologies point at m1_cim.json so this equals the CimTileModel() default (L4
        # byte-identical). A topology declaring a DIFFERENT params file swaps the accelerator geometry;
        # config._flag_provenance tags any non-Metis params `simulated` (no silicon).
        cim_params = None
        if topology is not None:
            from simulator.models.m2_memory import MemoryModel
            topo_spec = load_spec(topology)
            mm = MemoryModel(topo_spec)
            eff = float(mm.eff_BW_GBs) if mm.eff_BW_GBs else float(mm.pcie_BW_GBs)  # alpha: no DRAM -> PCIe
            self.per_call_floor_us = float(mm.floor_us)
            peak_mem = load_spec(memory_spec) if memory_spec else None
            _ccp = topo_spec.get("cim_compute_params")
            if _ccp:
                # Must be a repo-relative path that resolves INSIDE the repo: a spec may only reference
                # params committed to the repo, so the same commit is reproducible on any machine.
                # Reject absolute paths and `..` escapes (fail-loud) rather than read an out-of-repo file.
                repo_root = Path(__file__).resolve().parents[2]
                if Path(_ccp).is_absolute():
                    raise ValueError(f"cim_compute_params must be a repo-relative path, got absolute {_ccp!r}")
                _cpath = (repo_root / _ccp).resolve()
                if not _cpath.is_relative_to(repo_root):
                    raise ValueError(f"cim_compute_params {_ccp!r} escapes the repo root "
                                     f"({repo_root}); a topology may only reference in-repo params")
                cim_params = json.loads(_cpath.read_text())
        else:
            peak_mem = load_spec(memory_spec)
            eff = float(peak_mem["eff_BW_GBs"])              # measured anchor (e.g. 24.2)
            self.per_call_floor_us = 0.0
        if bw_efficiency is not None:                        # forward-looking override (sensitivity knob)
            if bw_efficiency <= 0:                            # non-physical effective BW -> reject at this boundary
                raise ValueError(f"Platform: bw_efficiency must be > 0 (got {bw_efficiency!r})")
            # card-only: alpha/edge have a physics-derived wall (PCIe / LPDDR5 x NoC) that peak-scaling
            # would silently bypass -> reject (single physics source = the topology spec). #63.
            if topology is not None and topology != "cim_topo_card":
                raise ValueError(
                    f"Platform: bw_efficiency is a card-only sensitivity knob; topology {topology!r} has "
                    f"its own physics-derived effective wall (e.g. edge = LPDDR5 x noc_efficiency) that "
                    f"peak-scaling would discard. Sweep eff via the topology spec, not bw_efficiency.")
            if peak_mem is None:
                raise ValueError("Platform: bw_efficiency override needs a memory_spec to read peak BW "
                                 "from (topology without on-card DRAM has no peak to scale)")
            peak = peak_mem.get("peak_GBs") or peak_mem.get("peak_BW_GBs") or (eff / float(peak_mem.get("efficiency", 1.0)))
            eff = float(peak) * float(bw_efficiency)
        if self.profile is not None:
            # the profile's memory block OVERRIDES the topology-resolved wall (ADR-0008 precedence);
            # an explicit knee_GBs (sensitivity knob) still wins over the profile knee.
            mem = self.profile["memory"]
            eff = float(mem["single_stream_eff_BW_GBs"])
            if knee_GBs is None:
                knee_GBs = float(mem["knee_GBs"])
        self.bw = SharedBandwidth(eff, knee_GBs=knee_GBs,
                                  interconnect_efficiency=interconnect_efficiency)
        self._build_units(cim_params, cim_n_cores, npu_scale)
        self.energy = EnergyModel()

    # ---- unit construction -----------------------------------------------------------------
    def _build_units(self, cim_params, cim_n_cores=None, npu_scale=1.0):
        """Legacy (no profile): the Phase-2.5 calibrated core EXACTLY as before (cim/cpu/gpu-attn)
        + rk3588_cim unit params for the NEW combos. Profile: everything from the profile.
        `cim_n_cores` overrides the CIM core count on either path; `npu_scale` multiplies the NPU TOPS."""
        legacy = self.profile is None
        src = load_soc_profile(_LEGACY_UNITS_PROFILE) if legacy else self.profile
        units = src["units"]
        self.sync_us = dict(src["sync_us"])
        self.launch_us = {u: float(units[u]["elementwise_launch_us"]) for u in ("gpu", "npu")}
        self.default_dtype = {u: units[u].get("default_dtype") for u in units}
        # CIM: tile params from the topology; n_cores = the profile's scalable axis (None = params default)
        n_cores = (None if legacy else int(units["cim"]["n_cores"])) if cim_n_cores is None else int(cim_n_cores)
        self.cim = CimTileModel(params=cim_params, n_cores=n_cores)
        self.cim_n_cores = int(self.cim.n_cores)
        attn_model = "memory_bound_zero" if legacy else units["cim"]["attention_model"]
        if attn_model == "alpha_floor_fit":
            self.cim_attn = CimAttentionModel(self.cim)
        elif attn_model == "memory_bound_zero":
            self.cim_attn = None
        else:
            raise ValueError(f"unknown cim.attention_model {attn_model!r} (alpha_floor_fit | memory_bound_zero)")
        # CPU: the calibrated support-op roofline (fixed spec on the legacy path) + the GEMM roofline
        cpu_spec = load_spec("cpu_rk3588" if legacy else units["cpu"]["spec"])
        self.cpu = CpuModel(cpu_spec)
        g = units["cpu"]["gemm"]
        self.cpu_gemm = CpuGemmModel(cpu_spec, cores=g["cores"], cluster=g["cluster"],
                                     efficiency=g["efficiency"], stream_BW_GBs=self.bw.eff_BW)
        # GPU: 'mali_fit' = Phase-1 micro-benchmark attention (MaliGpuModel, unchanged) + the Mali
        # roofline_fit for GEMM (shape trend only); 'anchor_roofline' = the HeteroInfer-anchored model
        gm = units["gpu"]["model"]
        if gm == "mali_fit":
            self.gpu = MaliGpuModel()
            self.gpu_gemm = GpuRooflineModel(load_spec(units["gpu"]["spec"]))
        elif gm == "anchor_roofline":
            self.gpu = GpuAnchorRooflineModel(load_spec(units["gpu"]["spec"]))
            self.gpu_gemm = self.gpu
        else:
            raise ValueError(f"unknown gpu.model {gm!r} (mali_fit | anchor_roofline)")
        self.gpu_model = gm
        # NPU: 'analytic_systolic' = the RKNPU2 NpuModel (engine analytic|onnxim|scalesim);
        # 'staircase_anchor' = the HeteroInfer-anchored NpuStaircaseModel
        nm = units["npu"]["model"]
        nspec = load_spec(units["npu"]["spec"])
        if nm == "analytic_systolic":
            self.npu = NpuModel(nspec, engine=units["npu"].get("engine", "analytic"))
        elif nm == "staircase_anchor":
            self.npu = NpuStaircaseModel(nspec, engine=units["npu"].get("engine", "analytic"))
        else:
            raise ValueError(f"unknown npu.model {nm!r} (analytic_systolic | staircase_anchor)")
        self.npu_scale = float(npu_scale)
        if self.npu_scale != 1.0:
            self.npu.scale_compute(self.npu_scale)

    @property
    def mem_domains(self):
        """The two memory pools the residency rule (R7) routes ops to: the shared `dram`
        (the resolved decode memory wall for THIS topology — the measured 24.2 GB/s on-card LPDDR4x
        on cim_topo_card; the counterfactual PCIe 3.9 GB/s on alpha; the simulated LPDDR5×noc on edge —
        where weights/KV/embedding are resident, metered by the M3 engine) and `cpu_cache` (the A76
        tiered on-chip cache, priced INSIDE m4_cpu for CPU-support ops — a labelled proxy for
        the all-AIPU Card's on-chip support; NEVER the DRAM pool, which removes the S-dc
        double-count)."""
        return {"dram": self.bw.eff_BW, "cpu_cache": self.cpu.spec["cache_bw_GBs"]}

    # ---- pricing ---------------------------------------------------------------------------
    @staticmethod
    def supports(unit, category):
        """True iff price() would not raise for (unit, category) — the matrix a solver enumerates."""
        return unit in _SUPPORTED and category in _SUPPORTED[unit]

    def price(self, node):
        """Price one op's COMPUTE on its assigned unit (#55 per-op provenance). Returns
        {latency_us, compute_provenance, source_model} — which Phase-1 unit model produced
        the latency and a human-readable provenance string. latency_us=0.0 = no compute model
        (pure-memory op; cost is its bytes_streamed via the engine). NB the final compute-vs-
        memory BOUND is the M3 engine's max(compute, dram_memory) decision, not this method's.

        Decided ONLY from node.unit / node.category / node.wl / node.pricing_group (never ids).
        `wl.extra['fixed_us']` (default 0) is the whole cost of a 'merge' node and an ADDITIVE
        term for any other category (scheduler-attached launch/sync overhead). Unsupported
        (unit, category) combos raise NotImplementedError (supports() is the oracle)."""
        u, cat, wl = node.unit, node.category, node.wl
        if u not in _SUPPORTED:
            raise ValueError(f"M3 platform: unknown/unpriced unit {u!r} (valid: {sorted(_SUPPORTED)})")
        fixed = float(wl.extra.get("fixed_us", 0.0)) if wl.extra else 0.0
        if cat == "merge":
            return {"latency_us": fixed, "source_model": "merge",
                    "compute_provenance": "fixed-cost merge/sync (set by scheduler)"}
        if cat not in _SUPPORTED[u]:
            raise NotImplementedError(
                f"M3 platform: no pricing for category {cat!r} on unit {u!r} "
                f"(supported on {u}: {sorted(_SUPPORTED[u])}); Platform.supports() is the oracle.")
        r = self._price_core(u, cat, node)
        if fixed:
            r = dict(r)
            r["latency_us"] = float(r["latency_us"]) + fixed
            r["compute_provenance"] += f"; + fixed_us={fixed:g} (scheduler-attached launch/sync overhead)"
        return r

    def _price_core(self, u, cat, node):
        wl = node.wl
        if cat == "convert":   # precision-boundary cast: memory-bound, cost = bytes via M3 (compute 0)
            return {"latency_us": 0.0, "source_model": "convert",
                    "compute_provenance": f"precision cast {wl.extra.get('from')}->{wl.extra.get('to')} "
                                          f"(memory-bound; cost = bytes_streamed via the M3 pool)"}
        if u == "cim":
            return self._price_cim(cat, node)
        if u == "cpu":
            return self._price_cpu(cat, node)
        if u == "gpu":
            return self._price_gpu(cat, node)
        if u == "npu":
            return self._price_npu(cat, node)
        return {"latency_us": 0.0, "source_model": "none",
                "compute_provenance": "mem/unmodeled (cost = bytes via the DRAM pool)"}

    # -- composite attention (R2): QK^T + S.V priced ONCE on the rep (pricing_group == id) ----------
    @staticmethod
    def _is_rep(node, unit):
        if node.pricing_group is None:
            raise NotImplementedError(
                f"{unit.upper()} attention bmm without a pricing_group: the QK^T+S·V composite must be "
                f"grouped (build_token_dag pairs them) so the composite is priced once, not per-bmm.")
        return node.pricing_group == node.id

    def _price_cim(self, cat, node):
        wl = node.wl
        if cat == "matmul" and wl.K and wl.N:
            lat = self.cim.dev_lat_us(wl.M, wl.K, wl.N)
            prov = "CIM-GEMV dev_lat (M1 tile model, Card-calibrated Phase 1.5; Alpha 13-pt cross-check)"
            if self.cim.is_extrapolated(wl.K, wl.N) or (
                    wl.M > 1 and self.cim.prefill_extrapolated(wl.M, wl.K, wl.N)):
                prov += "; EXTRAPOLATED beyond native envelope"
            return {"latency_us": float(lat), "compute_provenance": prov, "source_model": "m1_cim_tile"}
        if cat == "attention" and self.cim_attn is not None and "hd" in wl.extra:
            if not self._is_rep(node, "cim"):
                return {"latency_us": 0.0, "source_model": "m1_cim_attention",
                        "compute_provenance": "CIM attention grouped (composite priced on the QK^T rep)"}
            lat, extrap = self.cim_attn.composite_us(wl.kv or 1, heads=wl.heads, hd=wl.extra["hd"], M=wl.M)
            prov = ("CIM attention QK^T+S·V composite = Alpha attn_floor fit a+b*kv per bmm x heads "
                    "(measured basis: single head, hd=128, kv<=1025, M=1; heads serial = assumption)")
            if extrap:
                prov += "; EXTRAPOLATED (hd/kv/M outside the measured basis)"
            return {"latency_us": float(lat), "source_model": "m1_cim_attention", "compute_provenance": prov}
        return {"latency_us": 0.0, "source_model": "none",
                "compute_provenance": "CIM attention/elementwise: memory-bound (no compute model in 2.1; cost = bytes)"}

    def _price_cpu(self, cat, node):
        wl = node.wl
        if cat == "matmul":
            r = self.cpu_gemm.predict(wl)
            return {"latency_us": r["latency_us"], "compute_provenance": r["provenance"], "source_model": "m4_cpu_gemm"}
        if cat == "attention":
            if "hd" not in wl.extra:   # scale/mask elementwise: legacy 0 (cost = bytes)
                return {"latency_us": 0.0, "source_model": "none",
                        "compute_provenance": "no CPU compute model for category 'attention'"}
            if not self._is_rep(node, "cpu"):
                return {"latency_us": 0.0, "source_model": "m4_cpu_gemm",
                        "compute_provenance": "CPU attention grouped (composite priced on the QK^T rep)"}
            r = self.cpu_gemm.predict(wl)
            return {"latency_us": r["latency_us"], "compute_provenance": r["provenance"], "source_model": "m4_cpu_gemm"}
        op = _CPU_OP[cat]
        # NB the descriptive placement precision (node.precision='fp16', ADR-0004c) is
        # intentionally decoupled from the PRICING dtype: m4_cpu is calibrated only on fp32
        # cpu_ops.json (the fp16 set was numpy-emulated/unrepresentative), so we price fp32.
        wl2 = Workload(op=("softmax" if op.startswith("softmax") else op), kv=(wl.kv or 0),
                       extra={"model": self.model, "dtype": "fp32"})
        r = self.cpu.predict(wl2)
        return {"latency_us": r["latency_us"], "compute_provenance": r["provenance"], "source_model": "m4_cpu"}

    def _elementwise(self, u, cat):
        lat = self.launch_us[u]
        return {"latency_us": float(lat), "source_model": "elementwise_floor",
                "compute_provenance": (f"simulated: {u} elementwise '{cat}' = per-kernel launch floor "
                                       f"{lat:g} us (profile assumption, no {u} elementwise silicon); "
                                       f"data movement = bytes_streamed metered by the M3 pool")}

    def _price_gpu(self, cat, node):
        wl = node.wl
        if cat == "attention":
            if "hd" not in wl.extra:   # the bmm branch of wl_from_row sets 'hd'; scale/mask don't
                if self.gpu_model == "mali_fit":
                    raise NotImplementedError(
                        f"GPU pricing for non-bmm attention subtype (op={wl.extra.get('aten', '?')}): "
                        f"attn_bmm_us models the QK^T/S·V bmm ONLY (scale/mask go to CPU support).")
                return self._elementwise("gpu", "attention-scale/mask")
            # R2 group-aware composite: attn_bmm_us is the COMBINED QK^T+S·V cost, so price it
            # ONCE per pricing_group — on the rep (pricing_group == id); grouped members return 0.
            if not self._is_rep(node, "gpu"):
                return {"latency_us": 0.0, "source_model": "m4_gpu",
                        "compute_provenance": "GPU attention grouped (composite priced on the QK^T rep)"}
            if self.gpu_model == "mali_fit":
                lat = self.gpu.attn_bmm_us(wl.kv or 1, heads=wl.heads, layers=1)
                return {"latency_us": float(lat), "source_model": "m4_gpu",
                        "compute_provenance": "Mali GPU attn_bmm composite QK^T+S·V (group rep, priced once)"}
            r = self.gpu.predict(wl)
            return {"latency_us": r["latency_us"], "source_model": "m4_gpu_anchor",
                    "compute_provenance": r["provenance"] + " (group rep, priced once)"}
        if cat == "matmul":
            r = self.gpu_gemm.predict(wl)
            if self.gpu_model == "anchor_roofline":
                return {"latency_us": r["latency_us"], "source_model": "m4_gpu_anchor", "compute_provenance": r["provenance"]}
            return {"latency_us": r["latency_us"], "source_model": "m4_gpu_roofline",
                    "compute_provenance": "Mali GEMM KNOWN-UNOPTIMISED kernel (shape trend only, NOT an absolute): "
                                          + r["provenance"]}
        return self._elementwise("gpu", cat)   # norm/rope/ffn/softmax/residual

    def _price_npu(self, cat, node):
        wl = node.wl
        if cat == "attention":
            if "hd" not in wl.extra:
                return self._elementwise("npu", "attention-scale/mask")
            if not self._is_rep(node, "npu"):
                return {"latency_us": 0.0, "source_model": "m4_npu",
                        "compute_provenance": "NPU attention grouped (composite priced on the QK^T rep)"}
            r = self.npu.predict(wl)
            return {"latency_us": r["latency_us"], "source_model": "m4_npu",
                    "compute_provenance": r["provenance"] + " (group rep, priced once)"}
        if cat == "matmul":
            r = self.npu.predict(wl)
            return {"latency_us": r["latency_us"], "source_model": "m4_npu", "compute_provenance": r["provenance"]}
        return self._elementwise("npu", cat)

    def compute_us(self, node):
        """Compute latency (us) of one op on its assigned unit. 0.0 = no compute model
        (pure-memory op; cost is its bytes_streamed via the engine). Delegates to price()."""
        return self.price(node)["latency_us"]

    def energy_J(self, node, compute_us):
        """Per-op energy estimate (J). CIM matmul from flops; memory from bytes;
        CPU support from active compute time. ESTIMATED (M7, +/-20% band). GPU/NPU compute
        energy is NOT modelled (no power telemetry for either; ADR-0008 known gap)."""
        cat, wl = node.category, node.wl
        # DRAM energy only for DRAM-domain traffic — cpu_cache (CPU-support) bytes are on-chip,
        # priced inside the cpu_J term below (consistent with the latency-side domain split, R7).
        e = self.energy.dram_J(node.bytes_streamed) if (node.bytes_streamed and node.mem_domain == "dram") else 0.0
        if node.unit == "cim" and cat == "matmul":
            e += self.energy.cim_J(2 * wl.M * wl.K * wl.N)
        elif node.unit == "cpu" and compute_us > 0:
            e += self.energy.cpu_J(compute_us)
        return e
