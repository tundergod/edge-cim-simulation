"""Phase 0.3 B — production Metis Card end-to-end LLM (the L4 anchor).

Runs vendor precompiled INT8 LLMs via axllm --show-stats for llama-3.2-1b/3b and
llama-3.1-8b, each 1-core and 4-core, context 1024 (slug built-in). Parses
tokenization / prefill / TTFT / gen / tok-s / CPU% over a fixed prompt set; reports
median + spread per (model, cores). Feeds C5 (two-pillar prediction) and L4.

Run on metiscard inside axelera-env: python run_vendor_llm.py
"""
import json, re, statistics, subprocess, time
from pathlib import Path

OUT = Path.home() / "edge-cim-simulation/measurements/metis_card"
OUT.mkdir(parents=True, exist_ok=True)
SDK = Path.home() / "tundergod/voyager-sdk"

SLUGS = {
    "llama-3.2-1b": {"1c": "llama-3-2-1b-1024-static", "4c": "llama-3-2-1b-1024-4core-static"},
    "llama-3.2-3b": {"1c": "llama-3-2-3b-1024-static", "4c": "llama-3-2-3b-1024-4core-static"},
    "llama-3.1-8b": {"1c": "llama-3-1-8b-1024-static", "4c": "llama-3-1-8b-1024-4core-static"},
}
PROMPTS = [
    "Explain gravity in one sentence.",
    "Write a Python function to compute the nth Fibonacci number.",
    "Summarize the main causes of World War I in three short points.",
]
STAT = re.compile(r"Tokenization:\s*([\d.]+)ms.*Prefill:\s*([\d.]+)(us|ms|s).*TTFT:\s*([\d.]+)s"
                  r".*Gen:\s*([\d.]+)s.*Tokens/sec:\s*([\d.]+).*Tokens:\s*(\d+)")
CPU = re.compile(r"CPU %:\s*([\d.]+)%")
U = {"us": 1e-3, "ms": 1.0, "s": 1e3}  # -> ms


def run(slug, prompt, timeout=300):
    cmd = (f"cd {SDK} && source axelera-env/bin/activate && "
           f"axllm {slug} --prompt {json.dumps(prompt)} --show-stats --no-history 2>&1")
    r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout)
    m = STAT.search(r.stdout)
    if not m:
        return {"error": "no_stats", "tail": r.stdout[-300:]}
    c = CPU.search(r.stdout)
    return {"tokenization_ms": float(m.group(1)), "prefill_ms": float(m.group(2)) * U[m.group(3)],
            "ttft_s": float(m.group(4)), "gen_s": float(m.group(5)),
            "tok_s": float(m.group(6)), "tokens": int(m.group(7)),
            "cpu_pct": float(c.group(1)) if c else None}


def main():
    results = {}
    for model, cores in SLUGS.items():
        for nc, slug in cores.items():
            runs = []
            for p in PROMPTS:
                try:
                    res = run(slug, p)
                except subprocess.TimeoutExpired:
                    res = {"error": "timeout"}
                runs.append({"prompt": p[:40], **res})
                tag = res.get("error", f"{res.get('tok_s',0):.2f} tok/s, {res.get('tokens',0)} tok")
                print(f"[{model}/{nc}] {p[:35]!r:38s} -> {tag}", flush=True)
            ok = [r for r in runs if "tok_s" in r]
            results[f"{model}/{nc}"] = {
                "model": model, "cores": nc, "slug": slug, "context": 1024, "runs": runs,
                "tok_s_median": statistics.median(r["tok_s"] for r in ok) if ok else None,
                "ttft_s_median": statistics.median(r["ttft_s"] for r in ok) if ok else None,
                "prefill_ms_median": statistics.median(r["prefill_ms"] for r in ok) if ok else None,
            }
            (OUT / "vendor_llm_int8.json").write_text(json.dumps(results, indent=1))
    # 4c/1c speedup + decode-bandwidth sanity
    print("\n=== summary (median tok/s) ===")
    for model in SLUGS:
        m1 = results[f"{model}/1c"]["tok_s_median"]
        m4 = results[f"{model}/4c"]["tok_s_median"]
        sp = f"{m4/m1:.2f}x" if m1 and m4 else "n/a"
        print(f"  {model}: 1c={m1} 4c={m4} speedup={sp}")


if __name__ == "__main__":
    main()
