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

## Revisions 2026-10-23 round 2 (after matrix v1; branch `hetero`)

Fixes for the matrix-v1 anomalies 2, 3 and 5 (`cim-mobile-soc/experiments/RESULTS_v1.md`: `cim_only`
prefill beating `tiled_hetero + cim` in 10 configs, `ours (no CIM) ≥ ours + CIM` in 17, non-monotone
sweeps) — all symptoms of the per-op greedy solver never recovering a whole-DAG single-unit pipeline
and of its pessimistic hand-back accounting. Anomaly 1 (the CPU) is ADR-0008 round 2. Regression
tests: `tests/test_hetero_round2.py` (G2 cases); `tests/test_knobs.py` updated for the new candidate
record. Everything below stays SIMULATED; the legacy path is byte-identical (`validate_e2e_l4.py`
unchanged, snapshot tests green).

* **R2-1 — `host=auto` is a PORTFOLIO.** `TiledHeteroScheduler.assign` with `host=auto` now evaluates,
  through the EVENT ENGINE (the runner's ablation flags honoured), (a) the per-op solver plan once per
  candidate host (every enabled compute unit, as before) and (b) every single-unit plan — each enabled
  compute unit as the sole compute unit, support ops on it, or on the CPU when it cannot run them
  (`scheduler.single_support_unit`) — i.e. exactly the `gpu_only` / `npu_only` / `cim_only` /
  `cpu_only` placements under the same cfg (the CPU single plan exists only with `allow_cpu_compute`,
  which is what puts the CPU in the unit set). The overall fastest plan is kept (ties → first found:
  solver hosts in unit order, then singles → deterministic). **Consequence, stated explicitly: "ours ≥
  the best single-unit baseline" now holds BY CONSTRUCTION** for every DAG the runner times (each
  decode kv point and the prefill forward choose independently, so it holds for the averaged decode
  tok/s and for prefill tok/s alike). It is a property of the search, not a finding; the finding is
  *by how much* the solver plan beats the portfolio's single-unit members, which `plan_source` makes
  visible.
* **R2-2 — `plan_source` surfaced.** `dag.plan_source ∈ {"solver:<host>", "single:<unit>"}`,
  `dag.plan_pass ∈ {greedy, deferred, neighbours}` (solver plans), `dag.host_candidates = {plan_source:
  {est_us, engine_us[, greedy_engine_us, pass]}}` (keyed by plan source now, `est_us = None` for a
  single plan). Runner metrics: `plan_source`, `prefill_plan_source`, `plan_pass`, `prefill_plan_pass`;
  single-unit baselines report `single:<unit>`, `heteroinfer` `solver:gpu`, the legacy schedulers
  `None`. `host_candidates_est_us` keeps its key (shape changed).
* **R2-3 — solver refinement passes (`refine=True`).** After the per-op greedy pass (hand-back sync
  charged on every non-host single placement = the paper's `T_NPU^all + T_sync`, `SolverContext.handback
  = "always"`), the same DAG is re-solved twice from scratch: (i) **deferred** — every cross-unit boundary
  is priced ONCE, at its consumer: a non-host placement whose consumers the solver will still decide
  pays no hand-back now (the consumer pays its own `sync_in` later, with the producer's unit known); a
  FIXED consumer (mem / sampling / a pinned support block) or a terminal block keeps the hand-back; (ii)
  **neighbours** — as (i) but an already-decided consumer (the deferred pass's unit per op; a split → its
  merge unit = host) also charges the hand-back when it sits elsewhere, undoing an isolated off-host
  move. The engine times the three plans and the fastest is kept (ties → the earlier pass;
  `greedy_engine_us` recorded). Test: three chained matmuls each 12.5 µs cheaper on the NPU with a 10 µs
  sync — the greedy keeps all three on the GPU host (npu = 10 + 12.5 + 10 > 25 per op), the refined plan
  is one NPU run with ONE sync-in/out pair (2 syncs on the entry/exit converts, engine 57.5 vs 75 µs).
  Why not a strict fixed-neighbour local search: from the all-host plan no single op can move (its
  neighbours are on the host), so that search cannot leave the greedy's local minimum — deferred
  charging is what forms the run. The registered `tiled_hetero` has `refine=True`; the constructor
  default is `False` so `TiledHeteroScheduler(...)` restricted to HeteroInfer's search space still
  reduces to `heteroinfer` by construction, and `heteroinfer` itself stays the paper's per-op solver
  (no refinement). With `host=auto` the refinement runs on the engine-best solver host only (cost: 2
  extra solves, not 2 × |units|).
* **R2-4 — idempotency.** A DAG this scheduler already planned (`dag.planned_by == name`) is not
  re-solved on re-assign: only the common tail (`_finish`: conversions, unit bytes, residency, fusion
  tags, sync) is re-applied, which is idempotent; `host`, `plan_source`, `plan_pass`, `host_candidates`
  are kept. (Before, untouched nodes were re-decided "deterministically"; with the refinement passes that
  would not reproduce a refined plan.)
* **R2-5 — `cpu_only`** (`SingleUnitScheduler("cpu")`, `required_units = ("cpu",)`): the llama.cpp-style
  all-CPU baseline, SIMULATED; and `tag_fused_chains` now tags CPU elementwise nodes too, which only
  matters on a profile with `units.cpu.gemm.kernel_floor_us > 0` (ADR-0008 round 2).
* **Cost.** `host=auto`: |units| greedy solves + 2 refinement solves + |units| single plans, every
  candidate run through the engine (llama-3.1-8b, 4 units, cim on: ≈ 1.6 × the v1 time).
* **Still open.** The per-op estimate still has no cross-op overlap (the engine, not the estimate,
  ranks the portfolio); the refinement is a two-pass local improvement with no optimality claim;
  sweeps may remain non-monotone where portfolio members swap (the portfolio guarantees ≥ every member,
  not monotonicity in a knob); fusion is tagged after solving; the solver prices a split's consumer
  boundary at the merge (host), not per sub-node.
* **Measured effect (sd8gen3_cim, ctx 1024 / prefill 256 / decode 128, sync fast; decode tok/s / prefill
  tok/s, matrix v1 → round 2; `tiled_hetero` = host auto + allow_cpu_compute; includes the ADR-0008
  round-2 CPU refit):** 1B gpu_only 21.31/300 → same, npu_only 15.76/1419 → same, cim_only 16.97/4071 →
  14.87/3590 (its CPU support ops now pay the kernel floor), cpu_only — → 16.52/105, heteroinfer
  26.74/1369 → same, tiled (cim off) 43.60/1594 → 36.46/1569 (`solver:gpu` / prefill `solver:npu`), tiled
  (cim on) 47.30/4420 → 42.66/4090 (`solver:npu` / prefill `solver:cpu`); 8B gpu_only 5.21/52, npu_only
  2.28/197, heteroinfer 6.67/236 unchanged, cim_only 3.80/1004 → 3.58/944, cpu_only — → 3.80/17.6, tiled
  (cim off) 10.82/275 → 9.62/271 (`solver:gpu` / `solver:cpu`), tiled (cim on) 10.79/1059 → 10.11/1011
  (`solver:gpu` / `solver:cpu`). At these two configs `ours + CIM > ours (no CIM)` in decode (anomaly 3)
  and `ours + CIM ≥ cim_only` in prefill (anomaly 2) now hold; every winning plan is a solver plan, so the
  portfolio's single-unit members did not win here — the guarantee is what matters for the sweeps.
  Decode CPU utilisation of "ours" 0.60–0.76 → 0.53–0.70.

## Revisions 2026-10-26 (cooperation; branch `hetero`)

User request after matrix v3 (`cim-mobile-soc/experiments/RESULTS_v3.md`): in prefill "ours + CIM" equals
`cim_only` (the portfolio picks the all-CIM plan; GPU/NPU utilisation 0.1–0.3) and in decode the CIM is a
bytes stream that re-programs every beyond-capacity GEMV (ADR-0008 R3). Two mechanisms so the GPU/NPU work
while the CIM works. Both SIMULATED (no silicon); the legacy path is byte-identical (`validate_e2e_l4.py`
unchanged, snapshot tests green); `heteroinfer` unchanged (no CIM in its unit set, `prefill_chunks` default
1). Regression tests: `tests/test_cooperation.py`.

* **C1 — capacity-aware weight split** (`solver._capacity_candidates`, strategy `capacity_split`, dim N).
  For a matmul with the CIM in the unit set the CIM takes the output columns that FIT its residency budget,
  `N_cim = min(N, floor(budget / K))` aligned DOWN to the CIM core width (512; the combined 2048-wide tile
  would zero out every K = 4096 op at the 8.16 M knee), and the remainder `N − N_cim` goes to one other unit
  or to the ratio grid over the other units (host last), priced/merged like any N-split (`tile_parts` sizes
  the CIM part exactly). The CIM part is within capacity, so the M1 model prices it by the resident law with
  no R3 program term. Budget = `Platform.cim_weight_budget_elems` = knee × s (the R3 resident capacity) ×
  the profile parameter `units.cim.weight_budget_frac` (default 1.0; overridable per run by
  `platform.cim_weight_budget_frac`, provenance-flagged; later work may reserve part for KV; 0 disables the
  candidate = the pre-C1 solver); None on the legacy path and on fakes without the attribute → no candidate.
  - **Decode**: the weights are pinned ACROSS tokens — ONE global budget per token DAG. Two passes per
    host (`TiledHeteroScheduler._assign_host`): pass 1 solves the DAG with every matmul allowed to request
    up to the cap and records, per op, `saving = T_best_regular − T_best_capacity` (`SolverContext.cim_requests`;
    `T_best_regular` = the best NON-capacity candidate, which may itself be a plain CIM placement — a
    deviation from the brief's "without CIM", chosen so the density measures the marginal value of the
    pin); pass 2 allocates the budget by decreasing saving density (`scheduler._allocate_cim_budget`: full
    request when it fits the remainder, else the aligned-down columns the remainder affords; ties → lower
    id; no positive saving → nothing) and re-solves with each op offered exactly its allocation (absent →
    no capacity candidate: the normal candidates). An op that fits entirely is offered as a RESIDENT
    single placement besides the plain `cim_only`. The CIM part is tagged `wl.extra["cim_resident"]`:
    `tiling.unit_bytes` streams NO weight bytes for it (only the activation rows), in the solver's estimate
    and in the engine alike (`apply_unit_bytes`). The refinement passes reuse the greedy pass's allocation.
    The per-op estimate has no cross-op overlap, so a pinned plan can lose under the engine: the pass-2
    plan is timed against the same solve with nothing pinned and the faster is kept (ties → unpinned = the
    pre-C1 plan; `dag.cim_pin_engine_us`), so "with C1 ≥ without C1" holds per host by construction.
    Recorded: `dag.plans[nid]["cim_resident_elems"]` (and `cim_split_elems`), `dag.cim_budget_elems`,
    `dag.cim_budget_used`, `dag.cim_alloc`; runner `cim_weight_budget_elems`, `cim_budget_used_decode`,
    `cim_resident_matmuls_decode`. The one-time pin write (K·N_cim bytes once per generation) is NOT
    charged in the per-token latency (amortised over the decode tokens). Honesty: the plain CIM candidates
    keep R3's per-op `pin_first` assumption (every op pretends its first knee × s elements are pinned while
    still streaming all its weights); only the capacity split is budget-consistent — the two coexist in
    one plan, the engine picks.
  - **Prefill**: the budget is per op (an op may use up to cap); the CIM part's weights are written once
    per forward (its DRAM bytes metered by the pool, the R3 program term 0 since the part fits); never
    tagged resident. An op that fits entirely adds nothing (= `cim_only`).
  - Cost: two extra solves + one engine run per host in decode (pass 1 + the unpinned comparison). Measured effect (sd8gen3_cim, 4 cores, host auto,
    ctx 1024 / prefill 256 / decode 128): the budget is 8.16 M elements = 0.7 % of llama-3.2-1b's and
    0.1 % of llama-3.1-8b's weights, so decode gains are within 0.2 % (1B: the layer-0 Q projection
    pinned whole, 4.19 M elements; 8B: the layer-0 K projection, 4.19 M) — see the smoke table in the
    worker report / RESULTS_v4. The mechanism is in place for larger arrays (`cim_n_cores`) and the KV
    reservation.
* **C2 — prefill token-chunk pipelining** (`workload.build_prefill_dag`, `scheduler.prefill_chunks ∈ {1, 2,
  4, 8, "auto"}`, default 1 = the one-forward DAG, byte-identical). The forward is built as C chunk sub-DAGs
  of `P // C` rows (the last absorbs the remainder; `chunk_rows`), each from `build_token_dag` at its own
  row count (per-row ops exact → GEMM FLOPs conserved EXACTLY), with every attention block widened to
  `kv_c = rows of chunks ≤ c` (QK^T / S.V / scale / mask / softmax rescaled; block-causal). The fixture has
  no KV-append node in prefill, so chunk c's QK^T depends on the K-side rope outputs of chunks < c (told from
  the Q chain by the projection width equal to the V projection's; both when indistinguishable) and its S.V
  on their V projections; all other ops depend only on their own chunk. Ids chunk-major; nodes carry
  `wl.extra["chunk"]` / `["op_index"]`. **Bytes conserved** (the brief's requirement): a matmul of chunk c > 0
  carries `chunk_weights_shared` — its weights were streamed by chunk 0 and are ASSUMED to stay on the unit
  between consecutive chunks (`unit_bytes` streams its activation rows only, on every unit); the assumption
  is checked post hoc: the engine records the per-unit dispatch order (`stats["unit_order"]`) and
  `workload.weight_order_violations` counts the chunk-c > 0 matmuls whose unit ran a different GEMM since
  chunk c − 1's same GEMM (runner `prefill_weight_order_violations` / `_checked`). The whole chunked DAG is
  scheduled by the same per-op solver (every chunk re-solved: the attention blocks differ in kv; plans may
  differ per chunk) and the per-host portfolio; TTFT = the engine latency of the whole chunked DAG. `auto` =
  the runner's engine-ranked portfolio over C ∈ {1, 2, 4, 8} (ties → the smaller C), keeping the per-host
  portfolio inside each; `dag.prefill_chunks`, runner `prefill_chunks`, `prefill_chunks_config`,
  `prefill_chunk_candidates_us`. Decode untouched.
  - **Modelling differences, stated**: (i) the C = 1 DAG prices DENSE attention (the trace fixture's P × P
    bmms) while a chunked DAG prices the block-causal total Σ_c H·M_c·hd·kv_c (= (C+1)/2C of dense) — a
    model difference, not a scheduler gain; (ii) the shared-weight assumption is optimistic for the CIM:
    a beyond-capacity GEMM still pays the R3 re-programming TIME per chunk while its bytes are counted
    once; (iii) each chunk is a separate call: the CIM prefill tile law's per-call term `a` = 40.8 µs per
    2048² tile (the array programming + call overhead the Card measured), the NPU per-matmul floor, the
    GPU/NPU elementwise launch floors and the per-boundary syncs are all paid C times.
  - **Measured effect**: (iii) dominates. sd8gen3_cim, 4 cores, host auto, `tiled_hetero`, engine latency
    of the chunked forward: llama-3.2-1b P = 256: C = 1 67.0 ms (3821 tok/s), 2 73.7, 4 96.6, 8 141.1
    (CIM utilisation 0.34 → 0.50 / 0.70 / 0.89, GPU 0.07 → 0.52 at C = 4 — the units ARE busier, on
    inflated work: 232 tiles × 40.8 µs = 9.5 ms of per-call CIM cost per extra chunk, plus launch floors and
    syncs, against ≈ 5 ms of GPU work to hide); llama-3.1-8b P = 1024: C = 1 374 ms, 2 423, 4 712 (all
    `single:cim`; 1664 tiles × 40.8 µs = 68 ms per extra chunk + the per-chunk R3 program terms). The
    shared-weight assumption is violated for most chunk GEMMs (order check 135/146 at 1B C = 2, 213/225 at
    8B C = 2: the CIM interleaves GEMMs of different chunks), so the true chunked cost is higher still.
    `auto` therefore keeps C = 1 on every smoke config (RESULTS_v3's prefill numbers stand). Reading: the
    activation-centric chunking HeteroInfer uses for a weight-streaming NPU does not transfer to a
    weight-stationary CIM, whose per-call array programming is exactly what chunking multiplies; a CIM-side
    gain needs weight-stationary chunk ordering (all chunks of one GEMM back-to-back, which serialises the
    pipeline) or a larger array — open.
* **Engine**: `run_dag` iterates `started − done` in its finish passes (the running nodes; `finish` is
  order-independent — same results, ~2× faster on 2.5 k-node DAGs) and records `stats["unit_order"]`
  (both engines). `tiling.tile_parts(chain=)` lets the solver pass its `_block` once per op instead of
  recomputing it per candidate. Numerics unchanged (full suite + L4 identical).
* **Still open**: the solver's per-op estimate has no cross-chunk overlap (the engine ranks the chunk
  portfolio); the engine dispatches ready nodes lowest-id-first with no cross-chunk priority, so the host unit
  runs chunk c's long attention before chunk c+1's short norm that would unblock the CIM (on the test fake with
  equal CIM/GPU work C = 4 hides 10 % of 50 % possible); the budget allocation is greedy by density (no knapsack optimality claim) and computed once
  per host in the greedy pass; `weight_budget_frac` < 1 (KV reservation) is a parameter without a KV
  consumer yet; the CIM attention model has no residency notion.
