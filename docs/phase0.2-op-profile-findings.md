# Phase 0.2 — op-statistics / workload-op profile findings

Software-only run on the dev Mac (Apple M3 Pro / arm64), torch trace from Phase 0.1.
This phase turns Phase 0.1's deduped sweep matrix (580 distinct (op, shape) sigs / 9
categories) into a per-`(model × workload)` **profile**: for every sig, how many times
it executes (prefill vs decode), plus FLOPs / bytes / operational intensity from shape,
and a `measured` flag (is it on the Phase 0.1 discrete sweep grid). The weights this
produces are what compose per-op board latencies (Phase 0.3) into end-to-end cost, and
they are the predicted side of the roofline.

## Method (and why it is trustworthy)

- **Counting = op_inventory as oracle, not a hand rule.** Per-sig counts are read from
  Phase 0.1 `op_inventory` `count` fields, which already aggregate collided sigs
  (Llama q≡o, k≡v, gate≡up each = 2×layers; Qwen q-proj is `addmm` and o-proj is `mm`, so
  they do **not** collide). No "× layers" is ever hand-rolled.
- **Shapes at off-grid workload lengths = data-driven length-template.** Each sig's length
  dims are detected by diffing two collision-free inventory anchors (prefill 256/1024,
  decode 512/1024 — chosen so a length never equals `head_dim` 64/128, which would alias
  QK^T and S·V); a differing scalar fits `value = a·L + b` (integers; e.g. decode attention
  inner dim = `kv+1`). Counts are length-independent and carried from the inventory.
- **Validation (zero new tracing).** Templates built from the two anchors must reproduce
  the **held-out** inventory points exactly — prefill {128, 512}, decode {128} — sig and
  count, for all four models. At length 128 (= `head_dim` for 3B/8B/Qwen) QK^T and S·V alias
  to one sig; the merged count is reproduced by summing. `op_profile.py main()` runs this
  and refuses to emit on any mismatch. **All four models pass.**
- **FLOPs/bytes.** matmul/bmm = `2·M·K·N` (M,K,N from the in_shapes per op: mm `[[M,K],[K,N]]`,
  addmm `[[N],[M,K],[K,N]]`, bmm `[[B,M,K],[B,K,N]]`). bytes = (in + weight + out elements)
  × dtype; **GEMM operands INT8 (1 B)** (Metis scope), **non-GEMM FP16 (2 B)** (vendor actual);
  weights counted once per token (streamed, non-resident). `embedding` is a gather (touches
  only the gathered rows, no arithmetic). intensity = FLOPs/bytes is **predicted-side** — the
  measured roofline knee is Phase 0.3 / Phase 1.

## Headline findings

**1. Weight-stationary matmul dominates compute in every workload.** matmul is ≥0.86 of
prefill FLOPs and ≥0.58 of decode FLOPs across all 16 (model, task) cells; for the four
short/medium tasks it is ≥0.99 of both. This is the structural reason a **CIM-centric**
design — which excels precisely at weight-stationary GEMM/GEMV — is well-matched to LLM
inference, with the non-matmul remainder offloaded.

**2. Decode is uniformly memory-bound; prefill is compute-bound and scales with length.**
Decode matmul operational intensity is **2.0 FLOP/byte for every model and task** (M=1 GEMV:
each weight byte is used for ~one MAC) — squarely memory-bound, the regime where CIM
weight-residency helps most. Prefill matmul intensity rises with prefill length:
~115 (GSM8K, P≈59) → ~250 (HumanEval, P≈132) → ~335 (ShareGPT, P≈175) → **3300–5550**
(LongBench, P≈11.7k) — compute-bound, increasingly so for long context. This prefill→decode
collapse is Fig 2 / Fig 2b.

**3. Long-context prefill memory is attention, not weights.** For LongBench (P≈11.7k), matmul
is 86–89% of prefill FLOPs but only **3–5% of prefill bytes**; prefill memory is **65%
attention + 26% softmax** (8B) — the O(S²) score matrix. In LongBench decode (kv≈11.8k),
bytes split matmul 54% / attention 23% / kv_cache 22%. So the long-context regime is the one
place a weights-only memory model is wrong, and where attention/KV-cache offload matters most.

**4. The dominant ops are the ones the board measures.** Top decode sigs by total FLOPs
(8B, ShareGPT) are the FFN GEMVs — gate/up `[1,4096]×[4096,14336]` (2.59 TFLOP), down
`[1,14336]×[4096]` (1.29 TFLOP), the attention projections, then lm_head — all `measured=true`
(M=1 decode GEMVs are on the sweep grid). The low overall `measured` fraction for
decode-heavy tasks (e.g. 34/2125 for ShareGPT) is entirely the **per-kv-position attention
bmm / softmax / kv_cache** sigs, which are off-grid and reach the board model via the Phase-1
fitted latency equation (not a discrete lookup). Phase 0.3 should therefore spend its
measurement budget on the GEMV/GEMM grid (covered) plus a few attention-bmm kv anchors.

## Workload spectrum (mean prefill / decode, Llama family; Qwen ≈ same)

| task | dataset | P | D | regime | matmul FLOPs (pre/dec) | matmul intensity (pre/dec) |
|---|---|---|---|---|---|---|
| LongBench-TriviaQA | long-context | 11753 | 4 | prefill-heavy | 0.87 / 0.68 | 5554 / 2.0 |
| HumanEval | code | 132 | 54 | moderate | 1.00 / 0.99 | 255 / 2.0 |
| GSM8K | QA/reason | 59 | 102 | balanced | 1.00 / 1.00 | 116 / 2.0 |
| ShareGPT | chat | 175 | 344 | decode-heavy | 1.00 / 0.99 | 334 / 2.0 |

(8B values; LongBench P≈12.2k for Qwen. Decode intensity is 2.0 across the board.)

## Figures (regenerable from `measurements/op_profile/*.json` only)

- `docs/figures/phase0.2/fig1_op_breakdown_{model}.png` — op-category share (FLOPs & bytes),
  prefill vs decode, across the four tasks. (`tools/plotting/op_breakdown.py`)
- `docs/figures/phase0.2/fig1b_model_scaling_sharegpt.png` — decode op-mix vs model size.
- `docs/figures/phase0.2/fig2_roofline_{model}.png` — predicted-side intensity vs FLOPs
  contributed, prefill (filled) vs decode (open), illustrative ridge. (`tools/plotting/roofline.py`)
- `docs/figures/phase0.2/fig2b_intensity_shift.png` — matmul prefill→decode intensity collapse,
  four models.

Each plot reads only committed JSON and re-emits PNG + PDF + SVG (`tools/plotting/_style.py`,
nature-figure style). No figure is hand-drawn.

## Artifacts

- `tools/trace_export/op_profile.py` (template + oracle + self-validation),
  `tools/trace_export/gen_op_profile.py` (emits the JSONs)
- `measurements/op_profile/{model}_{task}.json` (16) — per-sig profile + per-category/phase totals
- `measurements/op_profile/sweep_{model}.json` (4) — on-grid Layer-B scaling (all `measured=true`)
- `tools/plotting/{_style,op_breakdown,roofline}.py`, `docs/figures/phase0.2/*.{png,pdf,svg}`

## Hand-off to Phase 0.3

The profile ranks ops by execution-weighted cost and flags on-grid vs interpolated coverage.
Phase 0.3 measures per-op latency on each unit; the dominant ops (FFN/projection GEMM in
prefill, the same as memory-bound GEMV in decode, attention bmm growing with kv) get the
measurement-budget priority, and the counts here weight them into end-to-end cost.
