"""Phase 0.1 step 6 — compact representative traces (simulator input samples).

Per (model, Layer-A task): ordered op x shape stream for a prefill at the task's
mean length (capped to CAP, within the 2K base context) + a representative decode
step at a mid-generation KV length. Records are compact [op, in_shapes, out_shape].
Full / long-context per-prompt traces are generated on-demand by M5 in Phase 2
(ADR-0002/0007), so this is a representative set, not exhaustive.

Run: PYTHONPATH=tools/trace_export ./.venv/bin/python tools/trace_export/gen_traces.py
"""
import json
from pathlib import Path

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from op_inventory import Recorder, _cfg, MODELS

CAP = 1024
WL = json.loads(Path("measurements/op_inventory/workload_lengths.json").read_text())
OUT = Path("traces")


def _compact(recs):
    return [[r["op"], r["in_shapes"], r["out_shape"]] for r in recs]


def trace_point(model, prefill_len, kv_len):
    # build inputs OUTSIDE the Recorder so input-construction ops aren't captured
    ids = torch.zeros((1, prefill_len), dtype=torch.long)
    with Recorder() as rec:
        model(input_ids=ids, use_cache=False)
    prefill = _compact(rec.records)
    cache = DynamicCache()
    pre = torch.zeros((1, kv_len), dtype=torch.long)
    model(input_ids=pre, past_key_values=cache, use_cache=True)
    dec = torch.zeros((1, 1), dtype=torch.long)
    pos = torch.tensor([kv_len])
    with Recorder() as rec:
        model(input_ids=dec, past_key_values=cache, use_cache=True, cache_position=pos)
    decode = _compact(rec.records)
    return prefill, decode


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for key, repo in MODELS.items():
        fake = FakeTensorMode(allow_non_fake_inputs=True)
        n = 0
        with fake:
            model = AutoModelForCausalLM.from_config(_cfg(repo)).eval()
            for task, stats in WL.items():
                if "error" in stats:
                    continue
                m = stats[key]
                P = min(int(round(m["prefill"]["mean"])), CAP)
                D = int(round(m["decode"]["mean"]))
                K = min(P + max(D // 2, 1), CAP)
                prefill, decode = trace_point(model, P, K)
                (OUT / f"{key}_{task}.json").write_text(json.dumps(
                    {"model": repo, "task": task, "prefill_len": P, "kv_len": K, "cap": CAP,
                     "note": "compact [op,in_shapes,out_shape]; capped to CAP; "
                             "long-context/full traces on-demand in Phase 2",
                     "prefill_ops": prefill, "decode_ops": decode}))
                n += 1
        print(f"{key}: wrote {n} task traces")


if __name__ == "__main__":
    main()
