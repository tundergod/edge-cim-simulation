# ADR-0009: Op tiling + heterogeneous schedulers (gpu_only / npu_only / heteroinfer / tiled_hetero)

**Status:** Proposed (2026-10-23)
**Relates to:** ADR-0001 (engine), ADR-0003 (scheduler = static-first, op/tensor granularity), ADR-0004 (precision boundaries). Design inputs: `cim-mobile-soc/RESEARCH_PLAN_2026-10-23.md` §3.2–3.3 and `docs/heteroinfer-design-reference.md` §3a–3d, §6.

## Context

The research question is whether a mobile SoC with CPU+GPU+NPU+SRAM-CIM gains from splitting **every** LLM op across units (not only the linear layers HeteroInfer partitions), under a realistic cross-unit synchronisation cost and a shared memory wall. The runtime (Phase 2.1–2.5) had one op → one unit placement (`AllCim`, `CimHetero`), no tensor-level splits, no sync cost, and no multi-unit baselines. Everything here is **SIMULATED**: there is no heterogeneous-execution silicon in the repo (#52); the HeteroInfer numbers are literature anchors ([LIT]).

## Decision

### 1. Data model (`simulator/runtime/tiling.py`)

`split_node(dag, nid, dim, plan)` replaces an op by k **sub-nodes** (same category, shrunk `Workload`, own `unit`, deps = the original's deps, `node.tile = {"of", "dim", "part", "n", "frac", "src"}`) plus one **merge node** (category `"merge"`, `tile.role = "merge"`, deps = the last node of every part) that the original's successors now depend on. `split_dag` applies a `{nid: (dim, plan)}` map. Split axes per category (plan §3.2):

| dim | category | sub-shape | replicated bytes (per extra part) | merge bytes |
|---|---|---|---|---|
| `N` | matmul | output features (HeteroInfer weight-centric, Fig.7) | activation `M·K` | 0 (concat) |
| `M` | matmul | tokens (activation-centric) | weight `K·N` | 0 (concat) |
| `K` | matmul | contraction | partial output `M·N` | `k·M·N·4 + M·N·bpe` (fp32 partial-sum add) |
| `heads` | attention block | query heads, quantum = GQA group (`heads/kv_heads`) | 0 | 0 (concat) |
| `kv` | attention block | KV-chunks (flash-decoding) | Q (`H·M·hd`) per chunk for QK^T; a full partial `H·M·hd` output per chunk for S·V | `k·H·M·(hd+2)·4 + H·M·hd·bpe` (log-sum-exp merge of per-chunk max/sum/output) |
| `rows` | norm / rope / ffn / softmax / residual / non-bmm attention | rows = tokens in flight (prefill only; decode has 1 row) | 0 | 0 |

An **attention block** = the QK^T rep + every node strictly between it and its S·V partner (scale, mask, softmax) + S·V. A head/kv split copies the whole chain per part on one unit and gives each sub-pair a **fresh `pricing_group`** (R2 composite priced once per sub-pair). The block must be closed (no between-node feeds outside it) — fail-loud.

**Invariants (tests/test_tiling.py):** FLOPs conserved exactly (integer partition of the split axis; the LAST part absorbs the alignment remainder, so an NPU quantum pads inside the pricing model, never in the DAG); `Σ sub bytes + merge bytes = original bytes + replicated bytes + merge bytes` with the replicated term closed-form; acyclic; successors rewired to the merge; ids deterministic (`max id + 1`, parts first, merge last); a single-part plan is a no-op (unit set, no merge). `align = {unit: (quantum, minimum)}` on the split axis (default NPU `(32, 32)` = the systolic tile; `heads` → the GQA group).

### 2. Sync model

`config.sync_mode ∈ {fast, clfinish, none}` → per-boundary cost from `tiling.SYNC_US`: **fast** = 10 µs decode / 30 µs prefill (HeteroInfer §4.3 predictable-wait + flag polling "a few µs", + "tens of µs" GPU submission delay in prefill), **clfinish** = 400 µs (§3.1 clFinish baseline), **none** = 0; `sync_us_override` wins. **Rule:** every cross-unit boundary (producer unit ≠ consumer unit, `mem` excluded) charges one sync — on the merge node (`wl.extra["fixed_us"]`) for a gather, otherwise on the consumer (`tiling.apply_sync`, recomputed from scratch → idempotent; a consumer with several foreign producers pays once). Copy cost is **0** (UMA zero-copy pool, §4.3). The pricing contract treats `wl.extra["fixed_us"]` as an additive overhead on any unit; `merge` latency = `fixed_us`.

Consequence vs the paper's equation: an NPU-only op pays sync-in + hand-back (= `T_NPU^all + T_sync`), a GPU-only op pays nothing (the GPU is the host graph), a split pays the non-host part's sync-in (overlapped under the host part) + the merge sync.

### 3. Solver (`simulator/runtime/solver.py`) — HeteroInfer §4.4 generalised

Per op (or attention block), in topological order (producers already placed → boundary syncs known), enumerate and price:

```
single u :  T = sync_in(u) + Σ_chain max(compute, dram_bytes/BW) + [sync if u ≠ host]
split    :  T = max( max_i [sync_in(u_i) + T_i],  Σ_i dram_bytes_i / BW_agg(k) ) + sync_merge
                + merge_bytes/BW + [sync if merge_unit ≠ host]
```

= `T_total = min(max(T_GPU^p1, T_NPU^p2) + T_sync + T_copy, T_GPU^all, T_NPU^all + T_sync + T_copy)` with `T_copy = 0`, sync charged exactly as the engine charges it, and the shared-bandwidth knee (`SharedBandwidth.aggregate_GBs(k)`) bounding the parts' aggregate streaming (the §3.3 "one unit cannot saturate the SoC bandwidth" mechanism). Strategies: `<u>_only`; `weight_centric` (N, ratio grid in multiples of `1/partition_grid`, default 1/8, over every unit subset; the host unit goes last and absorbs the remainder); `hybrid` (the same at a non-standard M > 1 — the NPU pads M internally); `activation_centric` (⌊M/s⌋ chunks of a standard length s ∈ {128, 256, 512, 1024} on a static-graph unit + the remainder on another unit); `token_split` / `k_split` / `head_split` / `kv_split` / `row_split` (tiled_hetero only). Infeasible plans (alignment / minimum violations) are skipped; ties → first found. Prices come **only** from `platform.compute_us` / `platform.supports` (never computed here). Complexity per op: O(|dims| · Σ_{S⊆units, |S|≥2} C(g−1, |S|−1)) candidates (126 grid candidates for 3 units at g = 8), each priced with |chain|·|S| calls.

### 4. Schedulers (`simulator/runtime/scheduler.py`; all `pipeline=True`, all need `platform=`)

| name | placement |
|---|---|
| `gpu_only` / `npu_only` | matmul + attention + support on that unit; embedding / kv_cache → mem; sampling → cpu; no syncs |
| `heteroinfer` | RMSNorm, RoPE, attention (whole block, never split), SwiGLU, residual → GPU (§4.1 Fig.6); every matmul (incl. lm_head) → solver over {gpu-only, npu-only, weight-centric N-grid, activation-centric chunks, hybrid}; prefill vs decode by `wl.M > 1`; host = merge unit = GPU; CPU control-plane only |
| `tiled_hetero` | every category decided by the same solver over the enabled units ⊆ {gpu, npu, cim} (+ cpu with `allow_cpu_compute`) and all legal dims; `allow_cim`, `allow_cpu_compute`, `support_unit`, `matmul_dims`, `split_attention`, `split_support`, `align`, `host` options. Restricted to HeteroInfer's search space it **is** `heteroinfer` (same code path; tested by construction) |

Common tail (`_finish`): `precision.insert_conversions` (consumer pays, as `cim_hetero`), residency (`_residency`: `cpu_cache` only for the calibrated CPU-support set — a CPU matmul / bmm / merge part streams DRAM), `apply_sync`. `assign` records every decision in `dag.plans[nid] = {category, strategy, dim, parts, sizes, block, sub_ids, merge_id, T_est_us, n_candidates, ...}`; idempotent (tiled / inserted nodes keep their unit; untouched nodes re-decide deterministically; the boundary check looks through converts).

### 5. Engine / runner statistics

`events.run_dag(..., stats={})` fills per-unit busy µs + utilisation, `n_sync` / `sync_us`, `n_merge`, DRAM bytes drained, memory-active time → achieved aggregate GB/s (the bandwidth-time integral), and a resource-free critical path. `runner.run` adds (existing keys unchanged): `prefill_tok_s`, `unit_utilisation` / `prefill_unit_utilisation`, `sync_count_per_decode_token`, `sync_us_per_decode_token`, `merge_count_per_decode_token`, `prefill_sync_count` / `prefill_sync_us` / `prefill_merge_count`, `achieved_decode_bandwidth_GBs` / `achieved_prefill_bandwidth_GBs`, `decode_critical_path_us` / `prefill_critical_path_us`, `sync_mode`, `plans_summary` (counts per strategy, decode + prefill).

## Consequences / limitations (honesty)

- **No silicon.** Every heterogeneous number is simulated; `config` flags `sync_mode`, non-AllCim schedulers and NPU as `simulated`. AllCim (L4 gate) is byte-identical (verified: `validate_e2e_l4` report unchanged).
- **Static ratios, per-op greedy.** One decision per op per DAG, producers placed first; no DAG-level list scheduling, no look-ahead to the consumer's unit (the hand-back sync of a non-host single placement is assumed), no fusion of consecutive same-way splits (each split pays its own merge + sync — pessimistic for chains of row-split elementwise ops).
- **Linear sub-shape scaling is the pricing model's job.** With the HeteroInfer Table 3 anchors scaled linearly in N, the solver reproduces the 1:1 (within one grid step), 3:1 and 2:3 rows, but the two "no partition" rows ([4096,14336] M=1 → GPU-only; [4096,4096] M=128 → NPU-only) only with a **minimum sub-tensor size** (§4.4 constraint (2), value unstated in the paper; the test uses 1024 output rows). Pure linear scaling prefers a 7:1 split for the first and is within 2 µs for the second (`tests/test_solver.py`). The real scheduler defaults to NPU `(32, 32)`; a larger NPU minimum needs NPU sub-shape data.
- **Not modelled:** NPU graph-build cost (§5.3.2, 0.4–2 s per LLM run — charged only to "online-prepare" baselines in the paper), NPU operand-order / shape-ratio factors beyond what the pricing model encodes, KV-cache placement per unit, dynamic ratios, batch > 1.
- **Memory headroom comes from the knee.** With the default `knee_GBs == eff_BW` a split of a memory-bound decode matvec gains nothing and the solver keeps it single-unit; HeteroInfer-style decode gains need the multi-stream knee (SoC profile / `tunables.knee_GBs`).
- **Residency refinement.** `_residency` now keeps `cpu_cache` only for the calibrated CPU-support categories (+ scale/mask, convert); AllCim / CimHetero placements are unaffected (they never put matmul/bmm/merge on the CPU).
- **Conversions.** The int8↔fp16 rule (ADR-0004) is applied unchanged at GPU boundaries; an NPU sub-node fed by a GPU op gets a consumer-side convert, which also carries that boundary's sync.

## Addendum (2026-10-23): `scheduler.host`

`SimConfig.host` (config key `scheduler.host`, values `gpu | npu | cim | auto`, default unset = the
previous behaviour: GPU if enabled else the first enabled unit) selects the graph-owning unit of
`tiled_hetero` (merge nodes land on it; a non-host single placement pays the hand-back sync).
`HeteroInferScheduler` keeps its constructor `host="gpu"` (the constructor value wins over the config).
`auto` runs the complete per-op solve once per candidate host (every enabled compute unit) on a deep
copy of the DAG and keeps the plan with the lowest **estimated token critical path = Σ per-op solver
estimates `T_est_us`** (the per-token DAG is a chain, so the solver's own objective summed over the
ops is its critical-path estimate; the engine's pipelined time is NOT consulted — the estimate can
disagree with the engine, see `host_candidates_est_us`). The decision is per DAG (each decode kv
sample point and the prefill forward choose independently). The chosen host is recorded as
`dag.host` (runner: `host`, `prefill_host`, `host_candidates_est_us`); a re-assign reuses it
(idempotent). Single-unit baselines record their unit as `host`.

## Revisions 2026-10-23 (independent review, fixes F1–F15; branch `hetero-fixes`)

* **F1 — sync is a serial pre-op delay** in both engines and the solver: `_node_us = fixed +
  max(compute, bytes/BW)` with `fixed` from `price()["fixed_us"]`; a candidate template is cleaned of
  any previous assignment's `fixed_us`/`sync_us`/fusion tags (`solver._clean`), so a re-assign is
  idempotent and the explicit `sync_in`/hand-back/merge terms are charged exactly once.
* **F4 — launch floor per fused kernel chain.** `_finish` runs `scheduler.tag_fused_chains`: an
  elementwise node on gpu/npu with exactly one dependency whose producer is an elementwise node on the
  same unit feeding nothing else is a FOLLOWER (`wl.extra["fused_head"] = False`, 0 floor); every other
  one is a HEAD and pays the profile's `elementwise_launch_us` once. Measured on the llama-3.2-1b
  fixture DAG: 309 heads of 476 elementwise nodes per decode token (joins and fan-outs — residual adds,
  rope on q and k — stay heads). The solver prices every candidate elementwise node as a head
  (pessimistic, decided before fusion is known). `allow_cpu_compute=False` (default) is the same unit
  set the gpu_only/npu_only baselines use (CPU = control plane + sampling, HeteroInfer §4).
* **F7 — `_estimate_split` sums the parts on one unit** (serial) before taking the max over units
  (an activation-centric plan runs its standard chunks sequentially on the NPU); the knee's k = the
  number of distinct units.
* **F9 — end-to-end tests through the real `Platform("sd8gen3_cim")`** replace the pre-ADR-0008 shim:
  per-unit utilisation ≤ 1, achieved bandwidth ≤ knee, single-unit baselines at 43.0 ± 1 % in decode,
  plan strategies within each scheduler's allowed set, Σ solver T_est within 20 % of the engine
  (measured −15 % on heteroinfer/tiled_hetero: the per-op estimate ignores cross-op overlap).
* **F10/F15 — per-unit stream caps.** `SharedBandwidth(unit_caps=)` (profiles only): a unit's streams
  share min(n·eff_BW, cap_u), units are water-filled under the knee (`share()`); two streams on one unit
  → 43 GB/s, one on each of two units → 60. A `mem` node's stream (embedding gather, KV append) is
  attributed to the compute unit that consumes — else produces — its value (`events._stream_owner`),
  so gpu_only / npu_only / cim_only decode report exactly 43.0 GB/s (were 43.1–44.4). Legacy (no
  profile): the uniform fair-share, byte-identical.
* **F3 — minimum sub-tensor rule** `units.<u>.min_split_rows` → `SolverContext.align[u] = (quantum,
  min)` unless the scheduler constructor overrides `align`; `tests/test_solver.py` uses the profile's
  NPU 1024 (no GPU minimum; under the linear fake the [4096,4096] M=128 row lands one grid step from
  NPU-only — the real-platform test is authoritative).
* **F11** — `all_cim` is labelled "legacy serial, no sync charged" (docstring; runner `sync_mode:
  "n/a-legacy"` for all_cim and cim_hetero); `_attn_M` docstring corrected (wl.M IS set by
  `wl_from_row`; the helper is the out_elems-consistent value for tiling).
* **Measured effect (sd8gen3_cim, prefill 256 / ctx 1024 / decode 128, tok/s before → after):**
  1B gpu_only 14.5→21.3, npu_only 14.9→15.8, cim_only 17.2→17.0, heteroinfer 21.7→26.7, tiled_hetero
  (cim off) 21.9→29.1, (cim on) 28.3→36.9; 8B gpu_only 4.38→5.21, npu_only 2.24→2.28, cim_only
  3.81→3.80, heteroinfer 5.71→6.67, tiled (cim off) 5.90→7.31, (cim on) 6.26→8.57. Achieved decode
  bandwidth: baselines 43.0, heteroinfer ~51–52, tiled ~52–55 (knee 60: the Table-3 kernels are
  kernel-bound at 17–30 GB/s-equivalent, so a split cannot fill the knee).
* **Still open:** the solver's per-op estimate has no cross-op overlap (−15 % vs the engine); fusion is
  decided after solving; `host=auto` compares Σ T_est (ADR-0008 addendum) which now includes the fixed
  delays consistently.
