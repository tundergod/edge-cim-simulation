"""Phase 0.1 step 4 — cross-check the fake-trace op set against a real-weights run.

Runs Llama-3.2-1B with REAL weights on CPU (prefill + 1-token decode), then
compares its op set to the meta/FakeTensor inventory on the semantic-op subset
(raw-aten housekeeping like detach/prim differs between fake and real and is
deliberately ignored).

Run: PYTHONPATH=tools/trace_export ./.venv/bin/python tools/trace_export/realweight_check.py
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from op_inventory import Recorder
from expected_ops import SEMANTIC

REPO = "meta-llama/Llama-3.2-1B"
SEM_PRIMS = set().union(*SEMANTIC.values())


def real_ops():
    model = AutoModelForCausalLM.from_pretrained(REPO, attn_implementation="eager").eval()
    ops = set()
    with torch.no_grad():
        with Recorder() as rec:
            model(input_ids=torch.zeros((1, 16), dtype=torch.long), use_cache=False)
        ops |= {r["op"] for r in rec.records}
        cache = DynamicCache()
        model(input_ids=torch.zeros((1, 16), dtype=torch.long), past_key_values=cache, use_cache=True)
        with Recorder() as rec:
            model(input_ids=torch.zeros((1, 1), dtype=torch.long), past_key_values=cache,
                  use_cache=True, cache_position=torch.tensor([16]))
        ops |= {r["op"] for r in rec.records}
    return ops


def main():
    real = real_ops() & SEM_PRIMS
    fake = set(json.loads(Path("measurements/op_inventory/llama-3.2-1b.json").read_text())["distinct_ops"]) & SEM_PRIMS
    print("real ∩ semantic:", sorted(real))
    print("fake ∩ semantic:", sorted(fake))
    if real == fake:
        print("PASS — semantic op sets match")
    else:
        print(f"FAIL — only-real={sorted(real - fake)}  only-fake={sorted(fake - real)}")


if __name__ == "__main__":
    main()
