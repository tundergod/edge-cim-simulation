"""Torch-free consumer API for the Phase 2.2a value-flow fixture.

Split out of trace_fixture.py so the SIMULATOR RUNTIME (M5/workload.py) can read the
committed structure WITHOUT importing torch (only the offline fixture *generator*,
trace_fixture.py, needs torch). Holds: the precision contract, the committed lengths,
fixture load, the fail-loud value-flow validator, and the compute-subgraph contraction.
"""
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sweep_matrix import categorize  # noqa: E402  (9-class op + src categorizer)

# sim_precision = the SIMULATED unit-native placement precision per category
# (ADR-0004c: CIM=INT8, Mali GPU=FP16, CPU-support=FP16/FP32). NOT trace_dtype (#8):
# the eager trace is fp32; GEMM is quantised to INT8 on the CIM in the simulator.
PRECISION_CONTRACT = {
    "matmul": "int8",      # CIM (q/k/v/o, gate/up/down, lm_head)
    "attention": "fp16",   # Mali GPU (QK^T / S·V bmm)
    "softmax": "fp16",     # CPU support
    "norm": "fp16",        # CPU support (internal accumulation fp32; placement fp16)
    "rope": "fp16",        # CPU support
    "ffn": "fp16",         # CPU support (SiLU)
    "residual": "fp16",    # CPU support
    "kv_cache": "int8",    # INT8 KV cache (Metis scope)
    "embedding": "fp16",   # FP16 table (memory)
}

# committed fixture lengths = op_profile anchors (avoid the head_dim 64/128 aliasing
# where QK^T and S·V collapse to one sig); two lengths/phase cross-check counts AND
# give M5 a positional shape-template per node (fit lo<->hi, instantiate at any L).
PREFILL_FIX = [256, 1024]
DECODE_FIX = [512, 1024]
FIXTURE_DIR = Path(__file__).resolve().parents[2] / "traces" / "fixture"


def load_fixture(key):
    with gzip.open(FIXTURE_DIR / f"{key}.json.gz", "rt") as fh:
        return json.load(fh)


def validate_value_flow(records):
    """Fail-loud (R5): every in_value must be produced by an earlier op or be
    explicitly external; alias_of keys must be real outputs. Returns True or raises."""
    produced, external = set(), set()
    for i, r in enumerate(records):
        external |= set(r.get("external_in", []))   # external once declared stays external
        for v in r["in_values"]:                     # (a weight/const is consumed by many ops)
            if v not in produced and v not in external:
                raise ValueError(f"op {i} {r['op']}: in_value {v} unresolved "
                                 f"(no earlier producer, not external)")
        for ov in r.get("alias_of", {}):
            if ov not in r["out_values"]:
                raise ValueError(f"op {i} {r['op']}: alias_of output {ov} not in out_values")
        produced |= set(r["out_values"])
    return True


def _compute_preds(i, records, producer, is_compute):
    """Nearest compute-op ancestors of op i: walk input value-ids back through
    uncategorized (view/host) producers until a compute op or an external leaf."""
    preds, seen = set(), set()
    stack = list(records[i]["in_values"])
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        p = producer.get(v)
        if p is None:                 # external value (weight/const/input) — not a compute dep
            continue
        if is_compute[p]:
            preds.add(p)
        else:
            stack.extend(records[p]["in_values"])
    return preds


def compute_subgraph(records):
    """Contract the full value graph onto categorized compute ops: a compute node's
    `deps` are the nearest compute ancestors (data deps), re-indexed 0..C-1 in trace
    order. This is the independent intra-layer DAG truth (R1)."""
    producer = {}
    for i, r in enumerate(records):
        for v in r["out_values"]:
            producer.setdefault(v, i)
    cats = [categorize(r) for r in records]
    is_compute = [c is not None for c in cats]
    local = {}
    for i in range(len(records)):
        if is_compute[i]:
            local[i] = len(local)
    nodes = []
    for i, r in enumerate(records):
        if not is_compute[i]:
            continue
        preds = _compute_preds(i, records, producer, is_compute)
        nodes.append({
            "idx": local[i],
            "op": r["op"], "src": r["src"], "category": cats[i],
            "in_shapes": r["in_shapes"], "out_shape": r["out_shape"],
            "trace_dtype": r["trace_dtype"],
            "sim_precision": PRECISION_CONTRACT[cats[i]],
            "deps": sorted(local[p] for p in preds),
        })
    return nodes
