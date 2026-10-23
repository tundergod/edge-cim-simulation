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
