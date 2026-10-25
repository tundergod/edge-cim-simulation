"""M6 — op->unit scheduler (Phase 2.2b): a `Scheduler` ABC + plugins.

`Scheduler.assign(dag, cfg) -> dag` is a pure, idempotent annotator that sets only
`node.unit` + `node.mem_domain` (it never touches category/wl/deps). `AllCimScheduler`
is the 2.1/2.2a all-CIM baseline (mirrors the L4 vendor all-AIPU INT8 placement: matmul
on CIM, support ops on CPU, attention on CIM as memory-bound, kv-append + embedding
through memory) — the ONLY hard-silicon-gated path (AllCim L4). `CimHeteroScheduler`
(the project's CIM-INT8 matmul × GPU-FP16 attention mixed-precision config) lands later
in this wave and is SIMULATED. The thin `all_cim_assign` wrapper is kept for callers.

ADR-0009 adds the heterogeneous family (all SIMULATED, `pipeline=True`, need a
`platform=` with the pricing contract price/compute_us/supports): `gpu_only` /
`npu_only` single-engine baselines; `heteroinfer` (faithful HeteroInfer §4: support +
attention on the GPU, every matmul decided by the §4.4 solver among gpu-only / npu-only /
weight-centric N-split / activation-centric chunks / hybrid); `tiled_hetero` (this
research: every category splittable per plan §3.2 across the enabled units, the same
solver over all legal dims). Tiling schedulers REWRITE the DAG (sub-nodes + merge nodes,
tiling.split_node), insert precision conversions, charge syncs (tiling.apply_sync) and
record the per-op decision in `dag.plans`.
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod

from simulator.runtime import tiling
from simulator.runtime.solver import SolverContext, solve, STD_LENGTHS

ALLCIM_MAP = {
    "matmul": "cim",
    "attention": "cim",
    "softmax": "cpu", "norm": "cpu", "rope": "cpu", "ffn": "cpu", "residual": "cpu",
    "kv_cache": "mem", "embedding": "mem",
}


def _residency(node):
    """Memory DOMAIN of an op (R7). The Metis Card is all-AIPU with 16 GiB on-card LPDDR4x
    (verified via axdevice): weights/KV/embedding are resident there and stream at ~24.2 GB/s
    -> `dram`. CPU-support activations (norm/softmax/...) are a few KB/token and live in
    on-chip SRAM, priced INSIDE m4_cpu (max(compute,cache_mem)+overhead) -> `cpu_cache`, so
    they must NOT also be metered into the DRAM pool (the S-dc double-count). NB: the Card has
    no RK3588; m4_cpu(A76) is a labelled PROXY for the AIPU's on-chip support (≈0.4% of decode).
    No traffic -> `none`. No 'local' domain (scope-out)."""
    if node.bytes_streamed == 0:
        return "none"
    if node.unit == "cpu" and node.category not in ("matmul", "merge") and not (
            node.category == "attention" and "hd" in node.wl.extra):
        return "cpu_cache"          # the calibrated CPU-support set (+ scale/mask, convert): on-chip
    return "dram"                   # a CPU matmul/bmm/merge part streams weights/KV/partials from DRAM


class Scheduler(ABC):
    """Pure, idempotent op->unit placement. `assign` sets node.unit + node.mem_domain (and may
    insert conversion ops) and returns a dag; calling it twice gives the same result. `pipeline`
    declares the engine execution mode: False = single-accelerator serial (AllCim, measured);
    True = multi-unit cross-op overlap (heterogeneous, SIMULATED)."""
    name: str = ""
    pipeline: bool = False
    required_units: tuple = ()        # units that MUST be enabled for this scheduler (fail-loud gate)

    @abstractmethod
    def assign(self, dag, cfg=None, *, platform=None):
        """`platform` (the pricing contract) is required by the solver-driven schedulers and
        ignored by the fixed-map ones."""
        raise NotImplementedError


class AllCimScheduler(Scheduler):
    """All-CIM baseline (the L4-gated all-AIPU INT8 placement). LEGACY SERIAL, NO SYNC CHARGED:
    it runs under the single-accelerator serial engine (pipeline=False) and never calls
    `apply_sync`, although every matmul->support edge is a cim<->cpu boundary (~305 per decode
    token on Llama-3.2-1B). It is the measured-silicon anchor, NOT a fair baseline for the
    heterogeneous schedulers — compare those against `cim_only` (same placement, syncs charged,
    pipelined engine); the runner labels its sync_mode "n/a-legacy"."""
    name = "all_cim"
    pipeline = False
    required_units = ("cim", "cpu")

    def assign(self, dag, cfg=None, *, platform=None):
        for n in dag.nodes:
            if n.category not in ALLCIM_MAP:        # fail-loud: a new M5 category must be mapped explicitly
                raise ValueError(f"AllCimScheduler: no unit mapping for category {n.category!r} "
                                 f"(known: {sorted(ALLCIM_MAP)}); map it explicitly, do not silently place.")
            n.unit = ALLCIM_MAP[n.category]
            n.mem_domain = _residency(n)
        return dag


class CimHeteroScheduler(Scheduler):
    """The project's mixed-precision config (SIMULATED): CIM-INT8 matmul × GPU-FP16 attention.
    matmul->cim, the QK^T/S·V bmm->gpu, scale/mask + support->cpu, kv_cache/embedding->mem;
    inserts int8<->fp16 conversion ops at the GPU boundary. Runs concurrent (pipeline=True)."""
    name = "cim_hetero"
    pipeline = True
    required_units = ("cim", "gpu", "cpu")

    def assign(self, dag, cfg=None, *, platform=None):
        from simulator.runtime.precision import insert_conversions   # deferred: avoids import cycle
        for n in dag.nodes:
            if n.category == "convert":
                continue                                            # preserve an already-inserted
                                                                    # convert's unit (idempotent re-assign)
            if n.category == "matmul":
                n.unit = "cim"
            elif n.category == "attention":
                n.unit = "gpu" if "hd" in n.wl.extra else "cpu"      # QK^T/S·V bmm vs scale/mask
            elif n.category in ("softmax", "norm", "rope", "ffn", "residual"):
                n.unit = "cpu"
            elif n.category in ("kv_cache", "embedding"):
                n.unit = "mem"
            else:                                                   # fail-loud: don't silently mis-route
                raise ValueError(f"CimHeteroScheduler: no unit mapping for category {n.category!r}; "
                                 f"map it explicitly, do not silently place on mem.")
        placement = getattr(cfg, "precision_boundary_placement", "consumer") if cfg else "consumer"
        dag = insert_conversions(dag, placement=placement)          # returns a rebuilt Dag
        for n in dag.nodes:
            n.mem_domain = _residency(n)
        return dag


_FIXED_MEM = ("kv_cache", "embedding")
_SUPPORT = ("norm", "rope", "ffn", "softmax", "residual")


def _phase_and_rows(dag):
    """('prefill'|'decode', tokens in flight) from the DAG: a matmul with M > 1 means prefill."""
    rows = max((n.wl.M for n in dag.nodes if n.category == "matmul"), default=1)
    return ("prefill" if rows > 1 else "decode"), rows


def _sync_us(cfg, phase):
    return tiling.resolve_sync_us(getattr(cfg, "sync_mode", "fast"), phase,
                                  getattr(cfg, "sync_us_override", None))


_ELEMENTWISE_FLOOR_UNITS = ("gpu", "npu", "cpu")   # cpu: round-2 revision (profile-path kernel floor)


def _is_elementwise(n):
    """Categories the gpu/npu price at the launch floor (support ops + non-bmm attention)."""
    return n.category in _SUPPORT or (n.category == "attention" and "hd" not in n.wl.extra)


def tag_fused_chains(dag):
    """ADR-0009 rev F4 — fused elementwise kernel chains. An elementwise node on gpu/npu/cpu is a
    FOLLOWER (`wl.extra['fused_head'] = False`, 0 launch floor) when it has exactly one
    dependency, that producer is an elementwise node on the same unit, and the producer feeds
    nothing else (a single-dependency chain = what a kernel fuser emits as one kernel); every
    other elementwise node is a HEAD (`fused_head = True`, pays the unit's launch floor).
    `fused_group` = the head's id (reporting only). Recomputed from scratch -> idempotent;
    non-elementwise / other-unit nodes carry no tag. Returns the number of chains. The CPU tag
    only matters on a SoC-profile path with `units.cpu.gemm.kernel_floor_us` > 0 (round-2
    revision); the legacy schedulers never call this, so the L4 path is untouched."""
    heads = 0
    for n in dag.nodes:
        ex = n.wl.extra
        if not (_is_elementwise(n) and n.unit in _ELEMENTWISE_FLOOR_UNITS):
            ex.pop("fused_head", None); ex.pop("fused_group", None)
            continue
        follower = False
        if len(n.deps) == 1:
            d = dag[n.deps[0]]
            follower = (_is_elementwise(d) and d.unit == n.unit and dag.successors(d.id) == [n.id]
                        and d.wl.extra.get("fused_group") is not None)
        if follower:
            ex["fused_head"] = False
            ex["fused_group"] = dag[n.deps[0]].wl.extra["fused_group"]
        else:
            ex["fused_head"] = True
            ex["fused_group"] = n.id
            heads += 1
    return heads


def apply_unit_bytes(dag, platform):
    """ADR-0008 rev F13 — rescale every matmul node's `bytes_streamed` to its unit's
    `weight_bytes_per_param` (weight part only; activation bytes unchanged) from the remembered
    INT8 basis (`wl.extra['bytes_int8']`), so the engine meters exactly what the solver priced.
    A platform without the table (legacy / fakes) is a no-op. Idempotent."""
    wbpp = getattr(platform, "weight_bytes_per_param", None) or {}
    for n in dag.nodes:
        if n.category != "matmul" or not (wbpp or n.wl.extra.get("cim_resident") or n.wl.extra.get("chunk_weights_shared")):
            continue                      # no table (legacy / fakes): only a resident / shared-weight part is rescaled (C1/C2)
        n.wl.extra.setdefault("bytes_int8", n.bytes_streamed)
        n.bytes_streamed = tiling.unit_bytes(n, wbpp.get(n.unit, 1.0))
    return dag


def _allocate_cim_budget(requests, budget, align):
    """Cooperation revision C1, decode pass 2: allocate the global residency `budget` (weight
    elements) to the pass-1 `requests` ({nid: {saving_us, K, N_cim, elems}}) in decreasing saving
    density (saving / elems; ties -> lower id); an op gets its full request when it fits the
    remainder, else the aligned-down columns the remainder affords; ops with no positive saving
    get nothing. Returns {nid: N_cim}."""
    q = max(1, int(align))
    order = sorted((nid for nid, r in requests.items() if r["saving_us"] > 0 and r["elems"] > 0),
                   key=lambda nid: (-requests[nid]["saving_us"] / requests[nid]["elems"], nid))
    alloc, rem = {}, float(budget)
    for nid in order:
        r = requests[nid]
        if r["elems"] <= rem:
            alloc[nid] = r["N_cim"]
            rem -= r["elems"]
        else:
            n = (int(rem // r["K"]) // q) * q
            if n > 0:
                alloc[nid] = n
                rem -= n * r["K"]
    return alloc


def _finish(dag, cfg, sync_us, platform=None):
    """Common tail of every ADR-0009 scheduler: precision conversions (consumer pays by default,
    as cim_hetero), per-unit weight bytes (F13), residency, fused elementwise chains (F4),
    per-boundary sync charging. Idempotent. The DAG order is topological (build_token_dag /
    split_node preserve it), which tag_fused_chains relies on."""
    from simulator.runtime.precision import insert_conversions   # deferred: avoids import cycle
    placement = getattr(cfg, "precision_boundary_placement", "consumer") if cfg else "consumer"
    plans = dag.plans
    dag = insert_conversions(dag, placement=placement)
    dag.plans = plans
    apply_unit_bytes(dag, platform)
    for n in dag.nodes:
        if n.category == "merge" and n.bytes_streamed > 0:
            n.mem_domain = "dram"
        else:
            n.mem_domain = _residency(n)
    tag_fused_chains(dag)
    tiling.apply_sync(dag, sync_us)
    return dag


class SingleUnitScheduler(Scheduler):
    """Single-engine baseline (SIMULATED): matmul + attention on `unit`; support ops on
    `support_unit` (default = the same unit -> no cross-unit syncs; for a CIM engine, which
    cannot run norm/rope/swiglu/softmax/residual, the support ops go to `support_unit` and
    every CIM<->support boundary pays the same per-boundary sync as the heterogeneous
    schedulers -- the FAIR single-engine CIM baseline, unlike the legacy serial `all_cim`);
    embedding / kv_cache through memory; sampling on the CPU."""
    pipeline = True

    def __init__(self, unit, support_unit=None, name=None):
        self.unit = unit
        self.support_unit = support_unit or unit
        self.name = name or f"{unit}_only"
        self.required_units = tuple(dict.fromkeys((unit, self.support_unit, "cpu")))

    def assign(self, dag, cfg=None, *, platform=None):
        phase, _ = _phase_and_rows(dag)
        for n in dag.nodes:
            if n.category in ("convert", "merge"):
                continue
            if n.category in _FIXED_MEM:
                n.unit = "mem"
            elif n.category == "sampling":
                n.unit = "cpu"
            elif n.category == "matmul" or (n.category == "attention" and tiling.is_attention_rep(n)
                                             or (n.category == "attention" and "hd" in n.wl.extra)):
                n.unit = self.unit
            elif n.category in _SUPPORT or n.category == "attention":   # scale/mask = support
                n.unit = self.support_unit
            else:
                raise ValueError(f"{self.name}: no unit mapping for category {n.category!r}; "
                                 f"map it explicitly, do not silently place.")
            dag.plans[n.id] = {"category": n.category, "strategy": f"{n.unit}_fixed", "dim": None,
                               "parts": [(n.unit, 1.0)]}
        out = _finish(dag, cfg, _sync_us(cfg, phase), platform)
        out.host = self.unit
        out.plan_source = f"single:{self.unit}"
        return out


def single_support_unit(unit, platform):
    """The support-op unit of a single-unit plan on `unit`: the unit itself when it can run the
    support ops, else the CPU (the CIM cannot run norm/rope/swiglu/softmax/residual) — exactly
    what gpu_only / npu_only / cpu_only / cim_only do."""
    sup = getattr(platform, "supports", None)
    return unit if (sup is None or all(sup(unit, c) for c in _SUPPORT)) else "cpu"


class TiledHeteroScheduler(Scheduler):
    """This research's scheduler (SIMULATED): every op (or attention block) is decided by the
    per-op solver over the enabled units in {gpu, npu, cim} (+ cpu when `allow_cpu_compute`)
    and the dims legal for its category (plan §3.2): matmul N/M/K, attention heads/kv
    (GQA-aligned), support ops rows (prefill only). `support_unit` pins the support ops +
    attention block to one unit instead of solving (HeteroInfer's GPU block); `matmul_dims`
    / `split_attention` / `split_support` restrict the search. With units {gpu,npu},
    support_unit='gpu', matmul_dims=('N',) and no attention/support splitting this is
    exactly HeteroInferScheduler (same code path). Decisions are made in topological order
    (producers placed first, so boundary syncs are known); ratios are static per op.
    `align` = {unit: (quantum, min)} on the split axis; default = the systolic quantum
    (tiling.DEFAULT_ALIGN, NPU 32) with the platform's per-unit `min_split_rows` (the SoC
    profile's minimum sub-tensor rule, HeteroInfer §4.4 constraint (2); rev F3) as the minimum.
    `allow_cpu_compute` default False = the SAME unit set the gpu_only / npu_only baselines
    use (CPU = control plane + sampling, HeteroInfer §4; cim_only's CPU support placement is
    forced by the CIM's op coverage, not a choice); True adds the CPU GEMM/attention/support
    kernel-time model to the search.
    `host` (constructor, else `cfg.host`): the graph-owning unit — a non-host single placement
    pays the hand-back sync and merges land on it. None = gpu if enabled else the first enabled
    unit; 'auto' = a PORTFOLIO (round-2 revision): the whole per-op solve once per candidate host
    (every enabled unit) PLUS the single-unit plans (every enabled compute unit as the sole compute
    unit, support ops on it or on the CPU when it cannot run them = exactly the gpu_only / npu_only /
    cim_only / cpu_only placements), every candidate timed by the EVENT ENGINE, the overall best
    kept. So "tiled_hetero >= the best single-unit baseline" holds BY CONSTRUCTION (same cfg):
    the portfolio contains the baselines. Recorded: `dag.host`, `dag.plan_source` in
    {"solver:<host>", "single:<unit>"}, `dag.plan_pass` (greedy | deferred | neighbours, solver
    plans only), `dag.host_candidates` = {plan_source: {est_us, engine_us}}.
    `refine` (round-2 revision; True for the registered 'tiled_hetero', False by default and for
    heteroinfer, which stays the paper's per-op solver): after the per-op greedy pass (hand-back
    sync charged on every non-host single placement), re-solve every op with the hand-back
    charged ONCE per boundary at its consumer (`SolverContext.handback = "deferred"`: a chain of
    ops each cheapest on the NPU becomes one NPU run with one sync-in/out pair instead of
    staying on the host), then once more given the units of its producers AND consumers from
    that pass ("neighbours": undoes an isolated off-host move), and keep whichever of the three
    plans the engine times fastest (ties -> the earlier pass; deterministic). Re-assigning a DAG
    this scheduler already planned only re-applies the common tail (idempotent)."""
    pipeline = True
    required_units = ("cpu",)

    def __init__(self, *, name="tiled_hetero", units=None, allow_cim=True, allow_cpu_compute=False,
                 support_unit=None, matmul_dims=("N", "M", "K"), split_attention=True,
                 split_support=True, align=None, std_lengths=STD_LENGTHS, host=None, grid=None,
                 refine=False):
        self.name = name
        self.refine = bool(refine)
        self.units = units                      # None -> cfg.units (enabled accelerators)
        self.allow_cim = allow_cim
        self.allow_cpu_compute = allow_cpu_compute
        self.support_unit = support_unit
        self.matmul_dims = tuple(matmul_dims)
        self.split_attention = split_attention
        self.split_support = split_support
        self.align = dict(align or {})
        self.std_lengths = tuple(std_lengths)
        self.host = host
        self.grid = grid

    # ---- unit set ---------------------------------------------------------------------
    def _units(self, cfg):
        if self.units is not None:
            enabled = list(self.units)
        else:
            flags = getattr(cfg, "units", None) or {"gpu": True, "npu": True, "cim": True, "cpu": True}
            enabled = [u for u in ("gpu", "npu", "cim") if flags.get(u)]
            if self._allow_cpu(cfg) and flags.get("cpu", True):
                enabled.append("cpu")
        if not self.allow_cim:
            enabled = [u for u in enabled if u != "cim"]
        if not self._allow_cpu(cfg):
            enabled = [u for u in enabled if u != "cpu"]
        if not enabled:
            raise ValueError(f"{self.name}: no compute unit enabled (units={getattr(cfg, 'units', None)}, "
                             f"allow_cim={self.allow_cim}, allow_cpu_compute={self.allow_cpu_compute})")
        return tuple(enabled)

    def _allow_cpu(self, cfg=None):
        """CPU as a compute unit: a config `scheduler.allow_cpu_compute` (True/False) overrides the
        constructor default; HeteroInferScheduler pins the constructor value (CPU control-plane only)."""
        if self.host is not None and self.name == "heteroinfer":
            return self.allow_cpu_compute
        c = getattr(cfg, "allow_cpu_compute", None)
        return self.allow_cpu_compute if c is None else bool(c)

    def _host(self, units, cfg=None):
        """Resolved host: constructor `host` wins, else `cfg.host` (gpu|npu|cim|auto), else default."""
        host = self.host if self.host is not None else getattr(cfg, "host", None)
        if host is None:
            return "gpu" if "gpu" in units else units[0]
        if host != "auto" and host not in units:
            raise ValueError(f"{self.name}: host unit {host!r} not in enabled units {units}")
        return host

    @staticmethod
    def _cim_budget(platform, units):
        """(budget_elems, align) of the capacity-aware weight split (C1): the platform's
        `cim_weight_budget_elems` (profile path: knee x s x units.cim.weight_budget_frac) and the
        CIM column quantum `cim_align_cols` (the core width); (None, 1) when the CIM is not in the
        unit set or the platform has no residency model (legacy / fakes) -> no capacity candidates."""
        b = getattr(platform, "cim_weight_budget_elems", None)
        if "cim" not in units or not b or b <= 0:
            return None, 1
        return float(b), max(1, int(getattr(platform, "cim_align_cols", 1)))

    def _ctx(self, platform, cfg, dag, units, host, phase, rows, handback="always", hints=None, cim_alloc=None):
        if platform is None:
            raise ValueError(f"{self.name}: assign() needs platform= (pricing contract: "
                             f"price/compute_us/supports)")
        budget, cim_align = self._cim_budget(platform, units)
        align = dict(self.align)
        for u, rows in (getattr(platform, "min_split_rows", None) or {}).items():   # profile minimum (F3)
            if u not in align:
                align[u] = (tiling.DEFAULT_ALIGN.get(u, (1, 1))[0], max(1, int(rows)))
        model = getattr(cfg, "model", None)
        if model is not None and "heads" not in align:
            from simulator.runtime.workload import op_profile   # deferred: workload sets up the tools path
            c = op_profile.Model(model).config
            group = max(1, int(c["heads"]) // max(1, int(c["kv_heads"])))
            align["heads"] = (group, group)          # GQA-group aligned head split
        dims = {"matmul": self.matmul_dims,
                "attention": ("heads", "kv") if self.split_attention else (),
                **{c: (("rows",) if self.split_support else ()) for c in _SUPPORT}}
        # fixed consumers for the deferred/neighbours hand-back rule (round-2): mem nodes, sampling,
        # and the pinned support/attention block (`support_unit`)
        fixed = {}
        for n in dag.nodes:
            if n.category in _FIXED_MEM:
                fixed[n.id] = "mem"
            elif n.category == "sampling":
                fixed[n.id] = "cpu"
            elif self.support_unit is not None and (n.category in _SUPPORT or n.category == "attention"):
                fixed[n.id] = self.support_unit
        return SolverContext(platform=platform, units=units, host=host, sync_us=_sync_us(cfg, phase),
                             grid=int(self.grid or getattr(cfg, "partition_grid", 8)),
                             std_lengths=self.std_lengths, align=align, rows=rows, dims=dims,
                             allow_chunking=True, handback=handback, fixed_units=fixed, hints=dict(hints or {}),
                             cim_cap_elems=budget, cim_align=cim_align, cim_resident=(phase == "decode"),
                             cim_alloc=cim_alloc)

    # ---- engine ranking of candidate plans ----------------------------------------------
    @staticmethod
    def _engine_us(dag, platform, cfg):
        """Token latency of a finished plan under the EVENT ENGINE (the runner's ablation flags
        honoured); a platform without a bandwidth model (bare fakes) falls back to the solver's
        own estimate."""
        from simulator.runtime.events import run_dag          # deferred: events is a leaf module
        bw = getattr(platform, "bw", None)
        if bw is None:
            return float(getattr(dag, "host_est_us", 0.0))
        return float(run_dag(dag, platform, bw,
                             concurrency=not getattr(cfg, "concurrency_off", False),
                             contention=not getattr(cfg, "contention_off", False),
                             price_compute=not getattr(cfg, "compute_off", False)))

    @staticmethod
    def _hints(dag):
        """{original op id: unit} of a solved plan (a split -> its merge unit = the host) for the
        'neighbours' hand-back rule."""
        out = {}
        for nid, p in dag.plans.items():
            parts = p.get("parts") or []
            out[nid] = parts[0][0] if (p.get("dim") is None and parts) else dag.host
        return out

    def _solve_host(self, dag, cfg, platform, units, host, refine):
        """The per-op greedy plan for `host` (+ the two refinement passes when `refine`), the
        engine-fastest kept; ties -> the earlier pass. Returns (plan, engine_us)."""
        base = self._assign_host(copy.deepcopy(dag), cfg, platform, units, host)
        base.plan_pass = "greedy"
        base_us = self._engine_us(base, platform, cfg)
        return self._refine(dag, cfg, platform, units, base, base_us) if refine else (base, base_us)

    def _refine(self, dag, cfg, platform, units, base, base_us):
        """Round-2 refinement of a greedy plan `base` (engine time `base_us`) for its host: the
        'deferred' pass, then the 'neighbours' pass seeded with the deferred units; the engine
        picks. Returns (plan, engine_us); `plan.greedy_engine_us` records the greedy time."""
        host = base.host
        best, best_us = base, base_us
        alloc = getattr(base, "cim_alloc", None)      # C1: the greedy pass's residency allocation is reused
        p2 = self._assign_host(copy.deepcopy(dag), cfg, platform, units, host, handback="deferred", cim_alloc=alloc)
        p2.plan_pass = "deferred"
        p3 = self._assign_host(copy.deepcopy(dag), cfg, platform, units, host, handback="neighbours",
                               hints=self._hints(p2), cim_alloc=alloc)
        p3.plan_pass = "neighbours"
        for cand in (p2, p3):
            t = self._engine_us(cand, platform, cfg)
            if t < best_us - 1e-9:
                best, best_us = cand, t
        best.greedy_engine_us = base_us
        return best, best_us

    def _single_plan(self, dag, cfg, platform, unit):
        """The single-unit plan on `unit` (support ops on it, or on the CPU when it cannot run them)
        = the gpu_only / npu_only / cim_only / cpu_only placement under this cfg."""
        out = SingleUnitScheduler(unit, support_unit=single_support_unit(unit, platform)).assign(
            copy.deepcopy(dag), cfg, platform=platform)
        out.host_est_us = None
        out.plan_pass = None
        return out

    def assign(self, dag, cfg=None, *, platform=None):
        units = self._units(cfg)
        host = self._host(units, cfg)
        if getattr(dag, "planned_by", None) == self.name:
            # idempotent re-assign: the plan (solver or single) is kept, only the common tail is
            # re-applied (conversions / unit bytes / residency / fusion tags / sync: all idempotent)
            keep = {k: getattr(dag, k) for k in ("host", "plan_source", "plan_pass", "host_candidates", "host_est_us",
                                                 "greedy_engine_us", "cim_budget_elems", "cim_budget_used", "cim_alloc",
                                                 "prefill_chunks", "chunk_candidates") if hasattr(dag, k)}
            phase, _ = _phase_and_rows(dag)
            out = _finish(dag, cfg, _sync_us(cfg, phase), platform)
            for k, v in keep.items():
                setattr(out, k, v)
            out.planned_by = self.name
            return out
        if host != "auto":
            out, _ = self._solve_host(dag, cfg, platform, units, host, self.refine)
            out.plan_source = f"solver:{host}"
            out.planned_by = self.name
            return out
        # host=auto (round-2 portfolio): the per-op solve once per candidate host + every single-unit
        # plan, all timed by the EVENT ENGINE (not the solver's serial estimate: the estimates tie
        # within <0.1% while the engine separates them by several %, matrix v0 anomaly 5; and the
        # per-op greedy never recovers an all-one-unit pipeline, matrix v1 anomaly 2). The refinement
        # passes run on the engine-best solver host only (cost: 2 extra solves, not 2 x |units|).
        # `host_candidates` keeps the numbers per candidate for reporting. Ties -> first found
        # (solver hosts in unit order, then singles) -> deterministic.
        cands = []                                          # (key, plan)
        est = {}
        for h in units:
            out, eng = self._solve_host(dag, cfg, platform, units, h, refine=False)
            out.plan_source = f"solver:{h}"
            est[out.plan_source] = {"est_us": float(out.host_est_us), "engine_us": eng}
            cands.append(((eng, float(out.host_est_us)), out))
        if self.refine:
            i = min(range(len(cands)), key=lambda j: cands[j][0])
            (eng, _), base = cands[i]
            out, eng2 = self._refine(dag, cfg, platform, units, base, eng)
            out.plan_source = base.plan_source
            est[out.plan_source] = {"est_us": float(out.host_est_us), "engine_us": eng2,
                                    "greedy_engine_us": eng, "pass": out.plan_pass}
            cands[i] = ((eng2, float(out.host_est_us)), out)
        for u in units:
            out = self._single_plan(dag, cfg, platform, u)
            eng = self._engine_us(out, platform, cfg)
            est[out.plan_source] = {"est_us": None, "engine_us": eng}
            cands.append(((eng, float("inf")), out))
        best = min(cands, key=lambda c: c[0])[1]            # min is stable: ties -> first found
        best.host_candidates = est
        best.planned_by = self.name
        return best

    def _assign_host(self, dag, cfg, platform, units, host, handback="always", hints=None, cim_alloc=None):
        phase, rows = _phase_and_rows(dag)
        ctx = self._ctx(platform, cfg, dag, units, host, phase, rows, handback=handback, hints=hints,
                        cim_alloc=cim_alloc)
        if ctx.cim_resident and ctx.cim_cap_elems is not None and cim_alloc is None:
            # C1 decode pass 1: every matmul may request up to the cap; its saving from pinning is
            # recorded; the global budget is then allocated by saving density and pass 2 (below) runs
            # with the allocation. Pass 1 runs on a copy; the plan it produces is discarded.
            p1 = self._ctx(platform, cfg, dag, units, host, phase, rows, handback=handback, hints=hints)
            self._solve_dag(copy.deepcopy(dag), p1, host, rows)
            cim_alloc = _allocate_cim_budget(p1.cim_requests, p1.cim_cap_elems, p1.cim_align)
            if cim_alloc:
                # the per-op estimate has no cross-op overlap, so a pinned plan can lose under the engine:
                # time the pinned plan against the same solve with nothing pinned and keep the faster
                # (ties -> unpinned = the pre-C1 plan) -> "with C1 >= without C1" per host, by construction
                pinned = self._assign_host(copy.deepcopy(dag), cfg, platform, units, host, handback=handback,
                                           hints=hints, cim_alloc=cim_alloc)
                plain = self._assign_host(dag, cfg, platform, units, host, handback=handback, hints=hints, cim_alloc={})
                t_pin, t_plain = self._engine_us(pinned, platform, cfg), self._engine_us(plain, platform, cfg)
                out = pinned if t_pin < t_plain - 1e-9 else plain
                out.cim_pin_engine_us = {"pinned": t_pin, "unpinned": t_plain}
                return out
            cim_alloc = {}
            ctx.cim_alloc = cim_alloc
        out = self._solve_dag(dag, ctx, host, rows)
        out = _finish(out, cfg, ctx.sync_us, platform)
        out.host = host
        out.host_est_us = sum(float(p.get("T_est_us", 0.0)) for p in out.plans.values())
        if ctx.cim_cap_elems is not None:
            out.cim_budget_elems = ctx.cim_cap_elems
            out.cim_alloc = cim_alloc                          # decode: {nid: N_cim}; prefill: None (per-op cap)
            out.cim_budget_used = sum(int(p.get("cim_resident_elems", 0)) for p in out.plans.values())
        return out

    def _solve_dag(self, dag, ctx, host, rows):
        """The per-op solve over `dag` in topological order under `ctx` (units set in place, splits
        rewrite the DAG); returns the solved Dag (before `_finish`)."""
        consumed = set()
        for nid in tiling._topo(dag, [n.id for n in dag.nodes]):
            if nid in consumed or nid not in dag._by_id:
                continue
            n = dag[nid]
            if n.category in ("convert", "merge") or n.tile is not None:
                continue                                  # already-tiled / inserted nodes keep their unit
            if n.category in _FIXED_MEM:
                n.unit = "mem"
                dag.plans[nid] = {"category": n.category, "strategy": "mem_fixed", "dim": None, "parts": [("mem", 1.0)]}
                continue
            if n.category == "sampling":
                n.unit = "cpu"
                dag.plans[nid] = {"category": n.category, "strategy": "cpu_fixed", "dim": None, "parts": [("cpu", 1.0)]}
                continue
            is_block = tiling.is_attention_rep(n)
            if n.category == "attention" and "hd" in n.wl.extra and not is_block:
                raise ValueError(f"{self.name}: S.V bmm {nid} reached before its QK^T rep "
                                 f"(blocks are decided at the rep)")
            pinned = (self.support_unit if (n.category in _SUPPORT or n.category == "attention") else None)
            if pinned is not None:
                chain = tiling._block(dag, nid)
                for cid in chain:
                    dag[cid].unit = pinned
                consumed.update(chain)
                dag.plans[nid] = {"category": n.category, "strategy": f"{pinned}_fixed", "dim": None,
                                  "parts": [(pinned, 1.0)], "block": chain}
                continue
            best, ncand = solve(ctx, dag, nid)
            chain = tiling._block(dag, nid)
            consumed.update(chain)
            dag.plans[nid] = {"category": n.category, "strategy": best.strategy, "dim": best.dim,
                              "parts": list(best.plan), "T_est_us": best.T_est, "n_candidates": ncand,
                              "block": chain}
            if n.category == "matmul" and ctx.cim_cap_elems is not None:
                # C1: the CIM-held weight elements of a capacity split (resident in decode = pinned across
                # tokens, charged against the DAG budget; prefill = written once per forward, not pinned)
                dag.plans[nid]["cim_resident_elems"] = best.cim_elems if best.resident else 0
                dag.plans[nid]["cim_split_elems"] = best.cim_elems
            if best.dim is None:
                for cid in chain:
                    dag[cid].unit = best.plan[0][0]
                    if best.resident:
                        dag[cid].wl.extra["cim_resident"] = True
            else:
                dag = tiling.split_node(dag, nid, best.dim, best.plan, merge_unit=host,
                                        sync_us=ctx.sync_us, align=ctx.align, rows=rows,
                                        resident=((0,) if best.resident else ()))
        return dag


class HeteroInferScheduler(TiledHeteroScheduler):
    """Faithful HeteroInfer (§4.1 / §4.2 / §4.4; SIMULATED): RMSNorm, RoPE, attention, SwiGLU,
    residual on the GPU; QKV / O / gate-up / down (and lm_head) matmuls decided per op by the
    §4.4 solver among gpu-only, npu-only, weight-centric N-split (ratio grid), activation-
    centric standard-length chunks with a GPU remainder, and hybrid; attention never split;
    CPU control-plane only (sampling); embedding / KV cache through memory. The GPU is the
    host graph: an NPU-only op pays the hand-back sync (T_NPU^all + T_sync), GPU-only does not.
    `heteroinfer_layer` (registry) = the paper's "Hetero-layer": the same solver restricted to a
    whole-op gpu/npu choice — `matmul_dims=()` (no weight-centric split) and `std_lengths=()` (no
    activation-centric chunking), so every matmul is gpu_only or npu_only and no merge node exists."""
    required_units = ("gpu", "npu", "cpu")

    def __init__(self, *, name="heteroinfer", matmul_dims=("N",), std_lengths=STD_LENGTHS, **kw):
        super().__init__(name=name, units=("gpu", "npu"), allow_cim=False,
                         allow_cpu_compute=False, support_unit="gpu", matmul_dims=tuple(matmul_dims),
                         split_attention=False, split_support=False, host="gpu",
                         std_lengths=tuple(std_lengths), **kw)


SCHEDULERS = {"all_cim": AllCimScheduler(), "cim_hetero": CimHeteroScheduler(),
              "gpu_only": SingleUnitScheduler("gpu"), "npu_only": SingleUnitScheduler("npu"),
              "cim_only": SingleUnitScheduler("cim", support_unit="cpu"),
              "cpu_only": SingleUnitScheduler("cpu"),      # round-2: the llama.cpp-style all-CPU baseline (SIMULATED)
              "heteroinfer": HeteroInferScheduler(), "tiled_hetero": TiledHeteroScheduler(refine=True),
              # paper "Hetero-layer" (§5 baseline): HeteroInfer without tensor partition / chunking (whole-op gpu|npu)
              "heteroinfer_layer": HeteroInferScheduler(name="heteroinfer_layer", matmul_dims=(), std_lengths=())}


def plans_summary(dag):
    """{strategy: count} over dag.plans (for reporting)."""
    out = {}
    for p in dag.plans.values():
        out[p["strategy"]] = out.get(p["strategy"], 0) + 1
    return dict(sorted(out.items()))


def all_cim_assign(dag):
    """Thin wrapper kept for existing callers (= AllCimScheduler().assign)."""
    return AllCimScheduler().assign(dag)


def domain_byte_audit(dag):
    """Byte-accounting oracle (R7): every node's bytes are attributed to exactly ONE memory
    domain (mutual exclusion, no op_bytes double-count). Returns per-domain totals + `ok`. In
    AllCim there are no cross-chip transfer_bytes (single accelerator), so total == op bytes."""
    out = {"dram": 0, "cpu_cache": 0, "none": 0, "total": 0}
    for n in dag.nodes:
        out[n.mem_domain] = out.get(n.mem_domain, 0) + n.bytes_streamed
        out["total"] += n.bytes_streamed
    out["ok"] = (out["dram"] + out["cpu_cache"] + out["none"] == out["total"])
    return out
