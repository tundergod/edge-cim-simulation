# Phase 0.1 — op inventory & workload findings

Software-only run on the dev Mac (Apple M3 Pro / macOS arm64 / CPU), torch 2.12.0 / transformers 5.9.0. Op graph extracted via meta/FakeTensor + `TorchDispatchMode` with forced eager attention (ADR-0007). No board, no GPU, no real weights (except the 1B real-weights cross-check).

## Model configs (shape parametrization)

Decode op shapes parametrize as `hidden / heads / kv_heads / head_dim / seq / kv_len`; attention is GQA (kv_heads < heads) in all four.

| model | layers | hidden | heads | kv_heads | head_dim | ffn | vocab | distinct aten ops |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | 16 | 2048 | 32 | 8 | 64 | 8192 | 128256 | 38 |
| Llama-3.2-3B | 28 | 3072 | 24 | 8 | 128 | 8192 | 128256 | 38 |
| Llama-3.1-8B | 32 | 4096 | 32 | 8 | 128 | 14336 | 128256 | 38 |
| Qwen2.5-7B | 28 | 3584 | 28 | 4 | 128 | 18944 | 152064 | 39 |

## Op set

All four decompose (eager) into the same ~38 aten primitives. Semantic ops → aten:
- **matmul** (QKV/O/FFN/lm_head) = `mm` (Llama, bias-free) or **`addmm`** (Qwen2.5 — QKV has bias, the only op-set difference, +1 op)
- **attention** QK^T / S·V = `bmm`; **softmax** = `_softmax`
- **RMSNorm** = pow+mean+add+rsqrt+mul; **RoPE** = cos+sin+neg+cat+mul+add; **SwiGLU** = silu+mul; **embedding** = `embedding`
- housekeeping (view/transpose/slice/arange/where/cumsum/causal-mask…) — whitelisted, host-side, excluded from the sweep matrix

**Completeness cross-check (`expected_ops.py`): ALL PASS** — every semantic op's primitives present, no unmatched ops, for all four models.
**Real-weights 1B cross-check (`realweight_check.py`): PASS** — Llama-3.2-1B run with real weights on CPU produces the same semantic op set as the meta/FakeTensor trace (raw-aten housekeeping like `detach`/`prim` differs and is ignored).

## Decode shapes scale with KV length (verified)

Tracing one decode token at `kv_len ∈ {128,512,1024}` (each via a prefill of that length into a `DynamicCache`): softmax out = `[1, heads, 1, kv_len+1]` → `[1,32,1,129]`/`[1,32,1,513]`/`[1,32,1,1025]` for 1B. Attention is decomposed (`_softmax`+`bmm`), **not** a fused SDPA op (eager forced).

## Workload length profiles (per model tokenizer, N=300, mean prefill / mean decode)

| task | dataset | Llama-1B | Qwen-7B | spectrum |
|---|---|---|---|---|
| chat | ShareGPT | 175 / 344 | 181 / 349 | **decode-heavy** |
| QA/reasoning | GSM8K | 59 / 102 | 61 / 123 | roughly balanced |
| code | HumanEval | 132 / 54 | 134 / 54 | moderate |
| long-context | LongBench-TriviaQA | 11753 / 4 | 12180 / 4 | **prefill-heavy** |

(3B/8B match 1B — same tokenizer family.) The four tasks span the full prefill-heavy ↔ decode-heavy spectrum, i.e. the CIM compute-bound ↔ memory-bound axis.

**Comparison to HeteroInfer Table 4** (the plan's sanity anchor): decode lengths agree in regime (LongBench ≈4 vs their 5 ✓; GSM8K/chat decode same order). Prefill **differs by setup, not error**: HeteroInfer GSM8K prefill 296 reflects **few-shot** prompting (ours is the raw zero-shot question ≈59); HeteroInfer LongBench prefill 1787 reflects **context truncation** (raw TriviaQA is ≈11.7k tokens — genuine long-context). For HeteroInfer external validation (ADR-0003) we would replicate their few-shot/truncation; under our own 2K/8K context scope LongBench prefill would be capped at 8K.

## Export surprises / notes

- **datasets 4.x removed script-based loading** → `THUDM/LongBench` (a script repo) can't `load_dataset`; read `triviaqa.jsonl` from its `data.zip` instead. GSM8K/HumanEval need namespaced IDs (`openai/gsm8k`, `openai/openai_humaneval`).
- **Qwen2.5 QKV bias** → `addmm` (Llama uses `mm`); recorded as the matmul op (mm-or-addmm).
- **ShareGPT** source = `Aeala/ShareGPT_Vicuna_unfiltered`; prefill = first user turn, decode = first assistant turn.

## KV-length silicon-anchor status

Op-inventory decode traced at `kv_len ∈ {128,512,1024}` (ADR-0002). Silicon anchors for L4 later: **Llama precompiled context ≤ 1024** (>1024 silently yields 0 tokens per voyager-sdk.md); phi3 reaches 2048 but is out of model scope. So 2048+ decode points are sim-only extrapolation, not silicon-anchored.

## Sweep matrix (→ Phase 0.2)

`sweep_matrix.json`: **580** distinct `(op, in_shapes, out_shape)` signatures, categorized by **semantic origin** — unambiguous ops by name, **overloaded ops (bmm/add/mul/cat/sub) by the emitting transformers module** the tracer records as `src` (issue #5; the #2 `sub` patch generalized to all overloaded ops). Counts: matmul 105, attention 104, rope 190, norm 90, ffn 30, residual 20, softmax 21, embedding 20. Host-side mask/position ops excluded; RoPE-frequency `bmm` correctly separated from attention `bmm`; `cat` operands captured (KV-cache concat) and cos/sin/neg included in `rope`. This is the op×shape set Phase 0.2 micro-benchmarks per unit.

## Deviations from plan

- **Operand activation/weight tag (step 2): not implemented.** Under FakeTensorMode params are fake tensors and appear transposed at the aten boundary, so id/shape tagging is unreliable. Instead the full `(M,K,N)` op signature is recorded — exactly the matmul/bmm spec a Phase 0.2 benchmark needs, so the sweep matrix is not polluted.
- **Layer-B-labelled traces (step 6): not emitted.** `gen_traces.py` writes only the 16 Layer-A task traces. The Layer-B prefill×decode sweep shapes are already fully captured in the op inventories (prefill {128,256,512,1024}, decode kv {128,512,1024}) and in `sweep_matrix.json`, so a separate `{prefill}x{decode}` trace per point would be redundant; regenerate on-demand if needed.
- **ShareGPT source:** the dedicated English set (`theblackcat102/sharegpt-english`) failed to load; fell back to `Aeala/ShareGPT_Vicuna_unfiltered` (ShareGPT is predominantly English). Length impact is minor; English-first ordering is retained for future runs.

## Artifacts

- `measurements/op_inventory/{llama-3.2-1b,llama-3.2-3b,llama-3.1-8b,qwen2.5-7b}.json` — config + distinct ops + deduped inventory + expected-ops check
- `measurements/op_inventory/workload_lengths.json`, `measurements/op_inventory/sweep_matrix.json`
- `traces/{model}_{task}.json.gz` — 16 gzipped compact representative traces (~98% redundant layer-replicated ops → ~170 KB total; prefill@mean capped 1024 + a decode step); full/long-context traces regenerate on-demand via `gen_traces.py` (ADR-0002/0007)
- `tools/trace_export/{op_inventory,expected_ops,realweight_check,workload_stats,gen_traces,sweep_matrix}.py`, `requirements.phase0.txt`
