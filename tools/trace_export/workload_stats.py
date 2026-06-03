"""Phase 0.1 step 5 — prefill/decode token-length stats per (task, model).

decode-length field per dataset:
  ShareGPT          : first user->assistant pair (user=prefill, assistant=decode)
  GSM8K             : answer (CoT + '#### N')          ; prefill = question
  LongBench-TriviaQA: answers[0]                       ; prefill = context + input
  HumanEval         : canonical_solution               ; prefill = prompt

Run: ./.venv/bin/python tools/trace_export/workload_stats.py
"""
import json
import statistics as st
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoTokenizer

MODELS = {
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
}
N = 300
OUT = Path("measurements/op_inventory/workload_lengths.json")
SHAREGPT_CANDIDATES = [
    "theblackcat102/sharegpt-english",  # English-first per workload scope
    "Aeala/ShareGPT_Vicuna_unfiltered",
    "anon8231489123/ShareGPT_Vicuna_unfiltered",
]


def _take(ds, n=N):
    return ds.select(range(min(n, len(ds))))


def load_gsm8k():
    ds = _take(load_dataset("openai/gsm8k", "main", split="test"))
    return [(r["question"], r["answer"]) for r in ds]


def load_longbench_triviaqa():
    # datasets 4.x rejects LongBench's loading script -> read triviaqa.jsonl from data.zip.
    import zipfile
    zpath = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    rows = []
    with zipfile.ZipFile(zpath) as z:
        cand = [m for m in z.namelist() if "triviaqa" in m.lower() and m.endswith(".jsonl")]
        cand.sort(key=lambda m: ("_e" in m, len(m)))  # prefer plain triviaqa.jsonl
        if not cand:
            raise RuntimeError(f"no triviaqa in zip; members: {z.namelist()[:20]}")
        with z.open(cand[0]) as fh:
            for i, line in enumerate(fh):
                if i >= N:
                    break
                r = json.loads(line)
                rows.append(((r.get("context", "") + "\n" + r.get("input", "")).strip(),
                             (r["answers"][0] if r.get("answers") else "")))
    return rows


def load_humaneval():
    ds = _take(load_dataset("openai/openai_humaneval", split="test"))
    return [(r["prompt"], r["canonical_solution"]) for r in ds]


def _first_pair(convs):
    user = next((c["value"] for c in convs if c.get("from") in ("human", "user")), None)
    asst = next((c["value"] for c in convs if c.get("from") in ("gpt", "assistant")), None)
    return user, asst


def load_sharegpt():
    last = None
    for repo in SHAREGPT_CANDIDATES:
        try:
            ds = _take(load_dataset(repo, split="train"))
            pairs = []
            for r in ds:
                convs = r.get("conversations") or r.get("conversation") or []
                u, a = _first_pair(convs)
                if u and a:
                    pairs.append((u, a))
            if pairs:
                print(f"  ShareGPT loaded from {repo} ({len(pairs)} pairs)")
                return pairs
        except Exception as e:  # noqa
            last = f"{repo}: {type(e).__name__}: {e}"
    raise RuntimeError(f"no ShareGPT candidate loaded; last error: {last}")


TASKS = {
    "sharegpt": load_sharegpt,
    "gsm8k": load_gsm8k,
    "longbench-triviaqa": load_longbench_triviaqa,
    "humaneval": load_humaneval,
}


def summ(vals):
    s = sorted(vals)
    p95 = s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
    return {"mean": round(st.mean(s), 1), "median": st.median(s), "p95": p95, "n": len(s)}


def main():
    toks = {k: AutoTokenizer.from_pretrained(r) for k, r in MODELS.items()}
    out = {}
    for task, loader in TASKS.items():
        print(f"[{task}] loading ...", flush=True)
        try:
            pairs = loader()
        except Exception as e:  # noqa
            out[task] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  ERROR: {out[task]['error']}")
            continue
        out[task] = {}
        for mk, tok in toks.items():
            pf = [len(tok(p, add_special_tokens=True).input_ids) for p, _ in pairs]
            dc = [len(tok(d, add_special_tokens=False).input_ids) for _, d in pairs]
            out[task][mk] = {"prefill": summ(pf), "decode": summ(dc)}
        ex = out[task]["llama-3.2-1b"]
        print(f"  llama-1b: prefill mean {ex['prefill']['mean']}, decode mean {ex['decode']['mean']}")
    OUT.write_text(json.dumps(out, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
