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
    """All-CIM baseline (the L4-gated all-AIPU INT8 placement)."""
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


def _finish(dag, cfg, sync_us):
    """Common tail of every ADR-0009 scheduler: precision conversions (consumer pays by default,
    as cim_hetero), residency, per-boundary sync charging. Idempotent."""
    from simulator.runtime.precision import insert_conversions   # deferred: avoids import cycle
    placement = getattr(cfg, "precision_boundary_placement", "consumer") if cfg else "consumer"
    plans = dag.plans
    dag = insert_conversions(dag, placement=placement)
    dag.plans = plans
    for n in dag.nodes:
        if n.category == "merge" and n.bytes_streamed > 0:
            n.mem_domain = "dram"
        else:
            n.mem_domain = _residency(n)
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
        return _finish(dag, cfg, _sync_us(cfg, phase))


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
    `align` = {unit: (quantum, min)} on the split axis (default NPU (32, 32) — the
    systolic tile; no silicon for a larger NPU minimum sub-tensor)."""
    pipeline = True
    required_units = ("cpu",)

    def __init__(self, *, name="tiled_hetero", units=None, allow_cim=True, allow_cpu_compute=False,
                 support_unit=None, matmul_dims=("N", "M", "K"), split_attention=True,
                 split_support=True, align=None, std_lengths=STD_LENGTHS, host=None, grid=None):
        self.name = name
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
            if self.allow_cpu_compute and flags.get("cpu", True):
                enabled.append("cpu")
        if not self.allow_cim:
            enabled = [u for u in enabled if u != "cim"]
        if not self.allow_cpu_compute:
            enabled = [u for u in enabled if u != "cpu"]
        if not enabled:
            raise ValueError(f"{self.name}: no compute unit enabled (units={getattr(cfg, 'units', None)}, "
                             f"allow_cim={self.allow_cim}, allow_cpu_compute={self.allow_cpu_compute})")
        return tuple(enabled)

    def _host(self, units):
        if self.host is not None:
            if self.host not in units:
                raise ValueError(f"{self.name}: host unit {self.host!r} not in enabled units {units}")
            return self.host
        return "gpu" if "gpu" in units else units[0]

    def _ctx(self, platform, cfg, dag, units, host, phase, rows):
        if platform is None:
            raise ValueError(f"{self.name}: assign() needs platform= (pricing contract: "
                             f"price/compute_us/supports)")
        align = dict(self.align)
        model = getattr(cfg, "model", None)
        if model is not None and "heads" not in align:
            from simulator.runtime.workload import op_profile   # deferred: workload sets up the tools path
            c = op_profile.Model(model).config
            group = max(1, int(c["heads"]) // max(1, int(c["kv_heads"])))
            align["heads"] = (group, group)          # GQA-group aligned head split
        dims = {"matmul": self.matmul_dims,
                "attention": ("heads", "kv") if self.split_attention else (),
                **{c: (("rows",) if self.split_support else ()) for c in _SUPPORT}}
        return SolverContext(platform=platform, units=units, host=host, sync_us=_sync_us(cfg, phase),
                             grid=int(self.grid or getattr(cfg, "partition_grid", 8)),
                             std_lengths=self.std_lengths, align=align, rows=rows, dims=dims,
                             allow_chunking=True)

    def assign(self, dag, cfg=None, *, platform=None):
        phase, rows = _phase_and_rows(dag)
        units = self._units(cfg)
        host = self._host(units)
        ctx = self._ctx(platform, cfg, dag, units, host, phase, rows)
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
            if best.dim is None:
                for cid in chain:
                    dag[cid].unit = best.plan[0][0]
            else:
                dag = tiling.split_node(dag, nid, best.dim, best.plan, merge_unit=host,
                                        sync_us=ctx.sync_us, align=ctx.align, rows=rows)
        return _finish(dag, cfg, ctx.sync_us)


class HeteroInferScheduler(TiledHeteroScheduler):
    """Faithful HeteroInfer (§4.1 / §4.2 / §4.4; SIMULATED): RMSNorm, RoPE, attention, SwiGLU,
    residual on the GPU; QKV / O / gate-up / down (and lm_head) matmuls decided per op by the
    §4.4 solver among gpu-only, npu-only, weight-centric N-split (ratio grid), activation-
    centric standard-length chunks with a GPU remainder, and hybrid; attention never split;
    CPU control-plane only (sampling); embedding / KV cache through memory. The GPU is the
    host graph: an NPU-only op pays the hand-back sync (T_NPU^all + T_sync), GPU-only does not."""
    required_units = ("gpu", "npu", "cpu")

    def __init__(self, **kw):
        super().__init__(name="heteroinfer", units=("gpu", "npu"), allow_cim=False,
                         allow_cpu_compute=False, support_unit="gpu", matmul_dims=("N",),
                         split_attention=False, split_support=False, host="gpu", **kw)


SCHEDULERS = {"all_cim": AllCimScheduler(), "cim_hetero": CimHeteroScheduler(),
              "gpu_only": SingleUnitScheduler("gpu"), "npu_only": SingleUnitScheduler("npu"),
              "cim_only": SingleUnitScheduler("cim", support_unit="cpu"),
              "heteroinfer": HeteroInferScheduler(), "tiled_hetero": TiledHeteroScheduler()}


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
