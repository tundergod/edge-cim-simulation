"""Phase 2.2a Step A — trace-truth value-flow fixture.

Re-runs the Phase-0.1 eager FakeTensor trace (op_inventory's TorchDispatchMode +
config), but additionally records the **producer->consumer value edges**: each
tensor gets a monotonic value-id on first sight, an op's `in_values` are the
value-ids it consumes and `out_values` the ids it produces. Contracting that full
value graph onto the categorized compute ops yields the real intra-layer
data-dependency DAG (#54) — the INDEPENDENT ground truth the Step-C structural
oracle validates the M5 template DAG against (R6 anti-self-confirmation).

Two load-bearing details (verified):
- **id-reuse (S1-1):** CPython recycles id() of a GC'd object, so without a
  strong reference a fresh tensor can inherit a dead tensor's id and fabricate a
  value edge (measured: 20 sequential FakeTensors -> 6 distinct ids). The recorder
  holds a strong ref to every tensor (`_keep`) so producer ids never recycle.
- **sim_precision != trace_dtype (#8):** the eager trace runs fp32; the SIMULATED
  placement quantises per ADR-0004c (CIM matmul INT8, Mali attention FP16, CPU
  support FP16). `trace_dtype` records the eager reality; `sim_precision` the
  contract. They differ by construction.

Run (writes committed fixtures): ./.venv/bin/python tools/trace_export/trace_fixture.py
"""
import gzip
import json
import sys
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, str(Path(__file__).parent))
import op_inventory  # noqa: E402  (_src, _cfg, _out_shape, MODELS)
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
# where QK^T and S·V collapse to one sig); two lengths/phase cross-check counts.
PREFILL_FIX = [256, 1024]
DECODE_FIX = [512, 1024]
FIXTURE_DIR = Path(__file__).resolve().parents[2] / "traces" / "fixture"

_DTYPE = {torch.float16: "fp16", torch.float32: "fp32", torch.bfloat16: "bf16",
          torch.float64: "fp64", torch.int64: "int64", torch.int32: "int32",
          torch.int16: "int16", torch.int8: "int8", torch.uint8: "uint8",
          torch.bool: "bool"}


def _dtype_str(dt):
    return _DTYPE.get(dt, str(dt).replace("torch.", ""))


def _flat_tensors(xs):
    """Tensors in xs, mirroring op_inventory._shapes order (positional + one level
    of list/tuple, e.g. torch.cat operands)."""
    out = []
    for a in xs:
        if isinstance(a, torch.Tensor):
            out.append(a)
        elif isinstance(a, (list, tuple)):
            out += [t for t in a if isinstance(t, torch.Tensor)]
    return out


def _out_tensors(rv):
    if isinstance(rv, torch.Tensor):
        return [rv]
    if isinstance(rv, (tuple, list)):
        return [t for t in rv if isinstance(t, torch.Tensor)]
    return []


def _storage_id(t):
    """Stable storage identity (aliasing views share it under FakeTensor)."""
    try:
        return t.untyped_storage()._cdata
    except Exception:
        return None


def _trace_dtype(outs, ins):
    for t in outs:
        if t.dtype.is_floating_point:
            return _dtype_str(t.dtype)
    for t in outs + ins:
        return _dtype_str(t.dtype)
    return None


class EdgeRecorder(TorchDispatchMode):
    """Records the ordered aten op stream WITH value-flow edges. Holds a strong ref
    to every tensor (`_keep`) so GC can never recycle a producer's id into a fake
    edge (S1-1). value-ids are monotonic and assigned on first sight."""

    def __init__(self):
        self.records = []
        self._keep = []          # strong refs to every tensor seen (anti id-reuse)
        self._vid = {}           # id(tensor) -> value_id
        self._producer = {}      # value_id -> op index that produced it (absent = external)
        self._next = 0

    def _input_vid(self, t):
        """value-id of an input tensor; (vid, is_external). Unseen at an input
        position => external (weight/const/cache/input_ids)."""
        self._keep.append(t)
        tid = id(t)
        if tid in self._vid:
            return self._vid[tid], False
        vid = self._next
        self._next += 1
        self._vid[tid] = vid
        return vid, True

    def _out_vid(self, t, op_index):
        """Fresh value-id for a produced output tensor."""
        self._keep.append(t)
        vid = self._next
        self._next += 1
        self._vid[id(t)] = vid
        self._producer[vid] = op_index
        return vid

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        in_tensors = _flat_tensors(args) + _flat_tensors(list(kwargs.values()))
        in_values, external_in = [], []
        in_storage = {}
        for t in in_tensors:
            vid, is_ext = self._input_vid(t)
            in_values.append(vid)
            if is_ext:
                external_in.append(vid)
            sp = _storage_id(t)
            if sp is not None:
                in_storage.setdefault(sp, vid)
        src = op_inventory._src()
        rv = func(*args, **kwargs)
        op_index = len(self.records)
        out_tensors = _out_tensors(rv)
        out_values, alias_of = [], {}
        for t in out_tensors:
            vid = self._out_vid(t, op_index)
            out_values.append(vid)
            sp = _storage_id(t)
            if sp is not None and sp in in_storage:   # output aliases an input's storage (view)
                alias_of[vid] = in_storage[sp]
        self.records.append({
            "op": str(func),
            "in_shapes": [list(t.shape) for t in in_tensors],
            "out_shape": op_inventory._out_shape(rv),
            "src": src,
            "in_values": in_values,
            "out_values": out_values,
            "external_in": external_in,
            "alias_of": alias_of,
            "trace_dtype": _trace_dtype(out_tensors, in_tensors),
        })
        return rv


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


def trace_phase(name, cfg, phase, L):
    """Ordered value-flow records for one forward (prefill seq=L, or decode 1 token
    at past=L). Inputs built OUTSIDE the recorder so input-construction isn't traced
    (mirrors op_inventory)."""
    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache
    from torch._subclasses.fake_tensor import FakeTensorMode
    fake = FakeTensorMode(allow_non_fake_inputs=True)
    with fake:
        model = AutoModelForCausalLM.from_config(cfg).eval()
        if phase == "prefill":
            ids = torch.zeros((1, L), dtype=torch.long)
            with EdgeRecorder() as rec:
                model(input_ids=ids, use_cache=False)
        else:
            cache = DynamicCache()
            pre = torch.zeros((1, L), dtype=torch.long)
            model(input_ids=pre, past_key_values=cache, use_cache=True)
            dec = torch.zeros((1, 1), dtype=torch.long)
            pos = torch.tensor([L])
            with EdgeRecorder() as rec:
                model(input_ids=dec, past_key_values=cache, use_cache=True,
                      cache_position=pos)
    return rec.records


def build_fixture(key, repo):
    cfg = op_inventory._cfg(repo)
    out = {"model": repo, "key": key, "prefill": {}, "decode": {}}
    for L in PREFILL_FIX:
        recs = trace_phase(key, cfg, "prefill", L)
        validate_value_flow(recs)
        out["prefill"][str(L)] = compute_subgraph(recs)
    for L in DECODE_FIX:
        recs = trace_phase(key, cfg, "decode", L)
        validate_value_flow(recs)
        out["decode"][str(L)] = compute_subgraph(recs)
    return out


def load_fixture(key):
    with gzip.open(FIXTURE_DIR / f"{key}.json.gz", "rt") as fh:
        return json.load(fh)


def main():
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for key, repo in op_inventory.MODELS.items():
        print(f"[{key}] tracing value-flow ...", flush=True)
        fx = build_fixture(key, repo)
        with gzip.open(FIXTURE_DIR / f"{key}.json.gz", "wt") as fh:
            json.dump(fx, fh)
        n_pre = len(fx["prefill"][str(PREFILL_FIX[-1])])
        n_dec = len(fx["decode"][str(DECODE_FIX[-1])])
        print(f"  -> traces/fixture/{key}.json.gz  (prefill nodes={n_pre}, decode nodes={n_dec})")


if __name__ == "__main__":
    main()
