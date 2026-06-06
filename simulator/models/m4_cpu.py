"""M4 — CPU (RK3588 big.LITTLE) support-op timing — CALIBRATED instruction-count ROOFLINE (Phase 1.2, D1).

The decode/prefill support ops the profile assigns to the CPU (RMSNorm, RoPE-apply, residual add,
SwiGLU, softmax, greedy argmax sampling) are timed by a roofline over an INSTRUCTION COUNT, not by a
lookup table of measured constants:

    latency_us = max(compute_us, memory_us) + overhead_op
    compute_us = (n_elem * ops_per_elem) / (sum_assigned[W*IPC*freq]) / eta_c
    memory_us  = working_set_bytes / (BW_tier(working_set) * eta_bw)

sum_assigned = the NEON fp32-lane throughput of the assigned cores (W lanes * IPC * freq, summed over
the cores). The CALIBRATION BASIS is ONE A76 core, single-thread (the cpu_rk3588 spec records this);
multi-core just sums more cores (EXTRAPOLATED). A55 has IPC=1 (single NEON pipe); A55 / multicore are
SIMULATED, not measured.

ROOFLINE TERMS:
  - compute: exp() (softmax/swiglu) is the COST DRIVER -> a large transcendental ops_per_elem
    (OPS_PER_ELEM), NOT a reduction/elementwise binary split. rmsnorm/rope-apply add a few ops/elem;
    residual and the argmax compare are ~1 op/elem.
  - memory: BW_tier is the CACHE layer chosen by the WORKING SET (L1/L2/L3 by capacity) -> decode
    support-op working sets reside in L1/L2/L3 and NEVER touch host LPDDR. ('swap LPDDR4->5 recompute'
    applies to PREFILL only, not these decode ops.) The cache BW comes from the cpu_rk3588 spec
    (A76 TRM estimate, `assumption`); it is the A76's OWN SRAM, not the Metis AIPU SRAM tier.
  - overhead_op: a per-op fixed cost (rmsnorm/rope/residual/sampling-dispatch are constant-dominated
    at decode sizes).

CALIBRATION (tools/analysis/fit_m4_cpu_instrcount.py -> params/m4_cpu_instrcount.json): eta_c (numpy
fraction of peak NEON fp32 throughput) and overhead_op are CALIBRATED to fp32 cpu_ops.json. eta_bw is
an ASSUMPTION (no bandwidth-resolved op in the fp32 decode data; no CPU mem-BW micro-benchmark) — the
memory term only binds for the largest working set (qwen vocab -> L3). OPS_PER_ELEM / BYTE_PASSES are
structural `assumption`s (instruction-count physics).

HONESTY: CPU = CALIBRATED to fp32 cpu_ops.json. fp16/int8 are NOT separately modeled -- predict()
returns the fp32-calibrated latency for ANY dtype; it does NOT capture the A76's fp16 numpy-emulation
overhead (measured fp16 is ~4x SLOWER than fp32), so it does NOT upper-bound fp16. The recompose CPU-
support term is therefore fp32. A55 + multicore = SIMULATED (extrapolated, not measured).
"""
import json
from pathlib import Path

from simulator.models.engine import UnitEngine, Workload

_PARAMS = Path(__file__).parent / "params" / "m4_cpu_instrcount.json"

# --- Model architecture dims (authoritative: same source as characterization/aetina/run_cpu_ops.py).
MODELS = {
    "llama-3.2-1b": dict(H=2048, F=8192, heads=32, hd=64, V=128256),
    "llama-3.2-3b": dict(H=3072, F=8192, heads=24, hd=128, V=128256),
    "llama-3.1-8b": dict(H=4096, F=14336, heads=32, hd=128, V=128256),
    "qwen2.5-7b":   dict(H=3584, F=18944, heads=28, hd=128, V=152064),
}

# --- Structural instruction-count physics (ASSUMPTION). exp() = the cost driver.
_EXP = 30  # transcendental instruction-weight: exp() expands to ~30 fused fp ops (assumption)
OPS_PER_ELEM = {            # arithmetic ops per element
    "residual": 1,             # one add
    "rmsnorm": 5,              # square, accumulate, rsqrt, scale, weight-mul
    "rope_apply": 6,           # 2 muls + 2 muls + add/sub per rotated pair, + stack/reshape
    "swiglu": _EXP + 4,        # silu = x/(1+exp(-x)) -> exp dominates, + mul/add/gate
    "softmax": _EXP + 3,       # exp(x - max) / sum -> exp dominates, + max-sub + div
    "sampling_argmax": 1,      # one compare per logit (reduction scan)
}
BYTE_PASSES = {             # working-set streams (read+write passes) for the memory term
    "residual": 2, "rmsnorm": 4, "rope_apply": 5, "swiglu": 3, "softmax": 3, "sampling_argmax": 1,
}
ETA_BW = 0.6  # cache-BW achieved fraction — ASSUMPTION (no CPU mem-BW micro-benchmark; audit gap)

_DEFAULTS = {  # baked calibrated factors so the engine runs before the fit script is re-run.
    "eta_c": 0.1521,
    "overhead_op_us": {"residual": 0.79, "rmsnorm": 16.81, "rope_apply": 22.53,
                       "sampling_argmax": 7.32, "softmax": 15.5, "swiglu": 0.0},
}


def _n_elem(op, c):
    """Size variable -> element count for the op. softmax/swiglu/sampling are the resolved sweeps."""
    if op == "rmsnorm":
        return c["H"]
    if op == "rope_apply":
        return c["heads"] * c["hd"]
    if op == "residual":
        return c["H"]
    if op == "swiglu":
        return c["F"]
    if op == "sampling_argmax":
        return c["V"]
    if op.startswith("softmax"):  # softmax_kv{kv} (parse kv) or softmax (kv from c["kv"]): heads*(kv+1)
        kv = int(op[len("softmax_kv"):]) if op.startswith("softmax_kv") else c["kv"]
        return c["heads"] * (kv + 1)
    raise KeyError(f"unknown CPU op: {op}")


def _working_set_bytes(base, n_elem):
    """Working-set bytes (fp32, all streams) for the memory term."""
    return n_elem * 4 * BYTE_PASSES[base]


def _peak_lane_ops(spec, cores=1, cluster="a76"):
    """sum_assigned[W*IPC*freq]: NEON fp32-lane throughput (lane-ops/s) of `cores` `cluster` cores."""
    cl = spec["clusters"][cluster]
    return spec["neon"]["fp32_lanes"] * cl["ipc"] * cl["freq_ghz"] * 1e9 * cores


def _tier_bw(spec, working_set_bytes):
    """Pick the CACHE BW tier (GB/s) by working set: L1 -> L2 -> L3 (never host LPDDR for decode)."""
    cache, bw = spec["cache"], spec["cache_bw_GBs"]
    if working_set_bytes <= cache["l1d_KiB_per_core"] * 1024:
        return bw["l1d_per_core"]
    if working_set_bytes <= cache["l2_KiB_per_core"] * 1024:
        return bw["l2_per_core"]
    return bw["l3_shared_per_core"]


class CpuModel(UnitEngine):
    """CALIBRATED CPU instruction-count roofline. Spec bound at construction; predict() eats a Workload.

    Workload: op, plus EITHER the size vars directly (heads/hd for rope, kv+heads for softmax, K for
    rmsnorm/residual hidden, N for swiglu F / sampling V) OR extra={'model','dtype','kv'} for the
    op_us() convenience path. cores/cluster live in extra (default: 1 A76 core = calibration basis)."""

    def __init__(self, spec, engine="analytic"):
        super().__init__(spec, engine)
        p = json.loads(_PARAMS.read_text()) if _PARAMS.exists() else _DEFAULTS
        self.eta_c = float(p["eta_c"])
        self.overhead = p["overhead_op_us"]

    def _latency(self, base, n_elem, cores, cluster):
        """max(compute, memory) + overhead_op -> (latency_us, bound)."""
        peak = _peak_lane_ops(self.spec, cores, cluster)
        compute_us = n_elem * OPS_PER_ELEM[base] / peak * 1e6 / self.eta_c
        wsb = _working_set_bytes(base, n_elem)            # total streamed bytes (the BW volume)
        single_copy = n_elem * 4                          # cache RESIDENCY = single-copy footprint, NOT x passes
        memory_us = wsb / (_tier_bw(self.spec, single_copy) * ETA_BW * 1e9) * 1e6
        ovh = self.overhead[base]
        core = max(compute_us, memory_us)
        bound = "compute" if compute_us >= memory_us else "memory"
        return core + ovh, bound

    def predict(self, wl: Workload) -> dict:
        """Frozen dict {latency_us, bound, provenance}. CPU = CALIBRATED to fp32 cpu_ops.json."""
        base = "softmax" if wl.op.startswith("softmax") else wl.op
        cores = int(wl.extra.get("cores", 1))
        cluster = wl.extra.get("cluster", "a76")
        # element count: prefer explicit Workload size vars; fall back to a named model in extra.
        if "model" in wl.extra:
            c = dict(MODELS[wl.extra["model"]], kv=wl.kv)
            n_elem = _n_elem(base, c)   # base normalizes softmax* -> 'softmax' (uses c['kv'], not an op-string kv)
        else:
            n_elem = self._n_elem_from_wl(base, wl)
        lat, bound = self._latency(base, n_elem, cores, cluster)
        dtype = wl.extra.get("dtype", wl.dtype)
        sim = cluster != "a76" or cores != 1
        tag = ("CALIBRATED to fp32 cpu_ops.json (1 A76 core)" if dtype == "fp32"
               else "fp32-calibrated value returned for dtype=%s; fp16 numpy-emulation overhead NOT "
                    "modeled -> does NOT upper-bound fp16" % dtype)
        if sim:
            tag += "; A55/multicore EXTRAPOLATED = simulated"
        prov = (f"{tag}: max(compute,memory)+overhead_op; eta_c calibrated, eta_bw=assumption; "
                f"{cores}x {cluster}, {bound}-bound")
        return {"latency_us": float(lat), "bound": bound, "provenance": prov}

    @staticmethod
    def _n_elem_from_wl(base, wl):
        """Element count straight from Workload size vars (no named model)."""
        if base == "rope_apply":
            return max(1, wl.heads) * (wl.extra.get("hd") or wl.K or 1)
        if base == "softmax":
            return max(1, wl.heads) * (wl.kv + 1)
        if base in ("rmsnorm", "residual"):
            return wl.K or wl.N or 1
        if base in ("swiglu", "sampling_argmax"):
            return wl.N or wl.K or 1
        raise KeyError(f"unknown CPU op: {wl.op}")

    def op_us(self, op, model, dtype="fp32", kv=None):
        """Convenience: per-token support-op latency (us) for a named model. Returns predict().latency_us.
        Defaults to fp32 = the CALIBRATED quantity (the model is fit on fp32 cpu_ops.json; fp16/int8 are
        NOT separately modeled, so a non-fp32 dtype returns the fp32 value, not an fp16 bound)."""
        wl = Workload(op="softmax" if op.startswith("softmax") else op, kv=kv or 0,
                      extra={"model": model, "dtype": dtype})
        return self.predict(wl)["latency_us"]
