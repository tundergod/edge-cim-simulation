"""Phase 0.1 — LLM op inventory via meta/FakeTensor + TorchDispatchMode.

Traces the aten op stream + shapes a model executes for prefill (several lengths)
and decode (one token at several KV-cache lengths), with no real weights / GPU.
Forces eager attention so attention stays decomposed (QK^T / softmax / S.V) rather
than fusing into a single SDPA op.

Run: ./.venv/bin/python tools/trace_export/op_inventory.py
"""
import json
import sys
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch._subclasses.fake_tensor import FakeTensorMode
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

MODELS = {
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
}
PREFILLS = [128, 256, 512, 1024]
KV_LENS = [128, 512, 1024]
OUT = Path("measurements/op_inventory")


def _shapes(xs):
    out = []
    for a in xs:
        if isinstance(a, torch.Tensor):
            out.append(list(a.shape))
    return out


def _in_shapes(args, kwargs):
    # positional + kwargs tensor args (issue #3: don't silently drop kwargs operands)
    return _shapes(args) + _shapes(kwargs.values())


def _out_shape(rv):
    if isinstance(rv, torch.Tensor):
        return list(rv.shape)
    if isinstance(rv, (tuple, list)):
        return [list(t.shape) for t in rv if isinstance(t, torch.Tensor)]
    return None


def _src():
    """Semantic origin of this op = the innermost transformers frame, as
    `ModuleClass.func` (e.g. LlamaRMSNorm.forward) or bare func for module-level
    helpers (eager_attention_forward, apply_rotary_pos_emb, sdpa_mask). Used to
    categorize overloaded aten ops (bmm/add/mul/cat) by origin, not op name (#5).
    Cross-model: matches Qwen2RMSNorm/Qwen2MLP/… by class suffix."""
    f = sys._getframe()
    while f is not None:
        fn = f.f_code.co_filename
        if "transformers/models/" in fn or "masking_utils" in fn:
            name = f.f_code.co_name
            slf = f.f_locals.get("self")
            if slf is not None:
                name = f"{type(slf).__name__}.{name}"
            return name
        f = f.f_back
    return None


class Recorder(TorchDispatchMode):
    def __init__(self):
        self.records = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        rv = func(*args, **kwargs)
        self.records.append(
            {
                "op": str(func),
                "in_shapes": _in_shapes(args, kwargs),
                "out_shape": _out_shape(rv),
                "src": _src(),
            }
        )
        return rv


def _cfg(repo):
    cfg = AutoConfig.from_pretrained(repo)
    cfg._attn_implementation = "eager"
    return cfg


def trace_model(repo):
    fake = FakeTensorMode(allow_non_fake_inputs=True)
    with fake:
        cfg = _cfg(repo)
        model = AutoModelForCausalLM.from_config(cfg).eval()
        samples = {"prefill": {}, "decode": {}}
        # prefill at each length
        for n in PREFILLS:
            ids = torch.zeros((1, n), dtype=torch.long)
            with Recorder() as rec:
                model(input_ids=ids, use_cache=False)
            samples["prefill"][str(n)] = rec.records
        # decode (one token) at each KV length
        for kv in KV_LENS:
            cache = DynamicCache()
            pre = torch.zeros((1, kv), dtype=torch.long)
            model(input_ids=pre, past_key_values=cache, use_cache=True)
            dec = torch.zeros((1, 1), dtype=torch.long)
            pos = torch.tensor([kv])
            with Recorder() as rec:
                model(input_ids=dec, past_key_values=cache, use_cache=True,
                      cache_position=pos)
            samples["decode"][str(kv)] = rec.records
        meta = {
            "n_layers": cfg.num_hidden_layers,
            "hidden": cfg.hidden_size,
            "heads": cfg.num_attention_heads,
            "kv_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            "head_dim": getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
            "intermediate": cfg.intermediate_size,
            "vocab": cfg.vocab_size,
        }
    return meta, samples


def distinct_ops(samples):
    s = set()
    for phase in samples.values():
        for recs in phase.values():
            for r in recs:
                s.add(r["op"])
    return sorted(s)


def dedupe(samples):
    """Collapse the ordered records to unique (op, in_shapes, out_shape) signatures
    per (phase, length), with an occurrence count. This is the inventory."""
    out = {}
    for phase, by_len in samples.items():
        out[phase] = {}
        for length, recs in by_len.items():
            seen = {}
            for r in recs:
                key = (r["op"], json.dumps(r["in_shapes"]), json.dumps(r["out_shape"]), r.get("src"))
                if key not in seen:
                    seen[key] = {**r, "count": 0}
                seen[key]["count"] += 1
            out[phase][length] = list(seen.values())
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for key, repo in MODELS.items():
        print(f"[{key}] tracing {repo} ...", flush=True)
        meta, samples = trace_model(repo)
        inv = dedupe(samples)
        doc = {"model": repo, "config": meta,
               "distinct_ops": distinct_ops(samples), "inventory": inv}
        (OUT / f"{key}.json").write_text(json.dumps(doc, indent=1))
        nsig = sum(len(v) for p in inv.values() for v in p.values())
        print(f"  -> {OUT/key}.json  ({len(doc['distinct_ops'])} distinct ops, {nsig} unique sigs)", flush=True)


if __name__ == "__main__":
    main()
