"""Shared unit-engine interface for Phase 1.2 component models.

Every non-micro-benchmark unit (CPU, NPU, GPU, memory, SRAM, CIM) is a model ENGINE
bound to a swappable SPEC at construction: `Engine(spec, engine='analytic')`. Swapping
a model = swapping the spec file; the engine code is unchanged. `predict(workload)` takes
ONLY a Workload and returns a FROZEN dict (exactly these three keys):

    {latency_us: float >= 0, bound: 'compute'|'memory'|'floor', provenance: str}

Units with a heavy variant (memory -> Ramulator2, NPU -> ONNXim; both Phase 1.3) keep the
SAME constructor signature `(spec, engine=)` and the SAME predict() contract; the heavy
engine wraps an adapter + per-shape cache internally. Freezing the return keys lets Phase
1.3 drop heavy engines in (`engine='ramulator2'|'onnxim'`) without touching this API. Any
richer detail an engine wants to expose lives behind its own methods, NOT in this dict.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

RETURN_KEYS = ("latency_us", "bound", "provenance")
BOUNDS = ("compute", "memory", "floor")


@dataclass
class Workload:
    """One op to time. shape = (M,K,N) GEMM/GEMV; kv/heads/layers for attention; nbytes
    for an explicit byte count when it is not derivable from the shape (e.g. kv-append).
    dtype default int8 (GEMM); non-GEMM support ops are fp16/fp32 (see per-engine spec)."""
    op: str
    M: int = 1
    K: int = 0
    N: int = 0
    kv: int = 0
    dtype: str = "int8"
    nbytes: int = 0
    heads: int = 1
    layers: int = 1
    extra: dict = field(default_factory=dict)


class UnitEngine(ABC):
    """Base for every per-unit timing engine. Spec is bound at construction; predict()
    eats only a Workload. `engine` selects the backend ('analytic' for Phase 1.2; heavy
    backends added in Phase 1.3 keep this exact signature)."""

    def __init__(self, spec, engine="analytic"):
        self.spec = spec
        self.engine = engine

    @abstractmethod
    def predict(self, wl: "Workload") -> dict:
        """Return the frozen dict {latency_us, bound, provenance}. Subclasses fill in."""
        raise NotImplementedError


def check_return(d):
    """Validate a predict() result is the frozen contract. Returns d (or raises)."""
    if set(d) != set(RETURN_KEYS):
        raise AssertionError(f"return keys {sorted(d)} != frozen {sorted(RETURN_KEYS)}")
    if d["bound"] not in BOUNDS:
        raise AssertionError(f"bound {d['bound']!r} not in {BOUNDS}")
    if not isinstance(d["latency_us"], (int, float)) or d["latency_us"] < 0:
        raise AssertionError(f"latency_us must be a non-negative number, got {d['latency_us']!r}")
    if not (isinstance(d["provenance"], str) and d["provenance"]):
        raise AssertionError("provenance must be a non-empty string")
    return d
